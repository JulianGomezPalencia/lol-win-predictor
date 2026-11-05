"""
Feature building: turn the exported tables into one row per match, ready for a model.

Everything here is computed from each player's games BEFORE the target match (player_history),
so nothing the model sees comes from the match itself or from later games.

Two levels:
  player_features()  one row per (match, player): recent form, lane fit, champion comfort...
  match_features()   one row per match: blue minus red, as a team and lane against lane

The prediction is meant to be made after champion select. By default (build()) the position in the
match is Riot's post-game teamPosition, treated as known; that can't be checked directly, and
train.position_sensitivity measures how much the results depend on it. With inferred_positions=True
the position is inferred from spells and earlier games, the only information a real prediction has.

Lane against lane is the new idea after the EDA: on their own, absolute numbers (a 60 % win rate)
said nothing about who wins, so we compare each player with the opponent in the same position.

The frozen 10-variable set (FINAL_FEATURES) adds three variables to seven existing team-level ones: mean
recent kills, games in the last 24 hours and the champion-by-position pool win rate. They are built only when
build.build(final_features=True, pool=...) asks for them; the default output (29 variables) is unchanged.

Run it:   python -m src.features.build        # reads data/export/*.csv, writes data/features/partidas.csv
Tests:    pytest tests/test_features.py -v
"""
from pathlib import Path

import numpy as np
import warnings

import pandas as pd

from src.features import champion_pool, roles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = PROJECT_ROOT / "data" / "export"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "partidas.csv"

BLUE, RED = 100, 200
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

# Per-player numbers that get averaged per team (blue minus red).
PLAYER_VARS = ["winrate_10", "en_linea_habitual", "partidas_con_campeon", "racha",
               "log_dias_desde_ultima", "kda_medio", "partidas_previas", "sin_historial"]
# Per-player numbers compared lane against lane.
LANE_VARS = ["winrate_10", "partidas_con_campeon", "en_linea_habitual", "kda_medio"]

# The three variables added for the frozen set: per player, then averaged per team (blue minus red).
NEW_PLAYER_VARS = ["kills_medio", "partidas_24h", "winrate_campeon_posicion"]
# The frozen model's variables, in this order, and the same without the weakest (the sensitivity check).
FINAL_FEATURES = ["dif_winrate_10", "dif_partidas_con_campeon", "dif_racha", "dif_log_dias_desde_ultima",
                  "dif_kda_medio", "dif_partidas_previas", "dif_sin_historial",
                  "dif_kills_medio", "dif_partidas_24h", "dif_winrate_campeon_posicion"]
SENSITIVITY_FEATURES = FINAL_FEATURES[:9]
DAY_MS = 86_400_000


def racha_actual(wins: list[int]) -> int:
    """Current streak, newest game first: +3 = three wins in a row, -2 = two losses in a row."""
    if not wins:
        return 0
    first, length = wins[0], 0
    for win in wins:
        if win != first:
            break
        length += 1
    return length if first == 1 else -length


def _usual_position(positions: pd.Series) -> str | float:
    """The position the player repeats most. Games come newest first, so a tie goes to the
    position among the tied ones that the player played most recently."""
    known = positions[positions.fillna("") != ""]
    if known.empty:
        return np.nan
    return known.value_counts(sort=False).idxmax()  # sort=False keeps first-seen (newest) order


def player_features(history: pd.DataFrame, participants: pd.DataFrame, matches: pd.DataFrame,
                    pool: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """One row per participant of each match that has its history, with the pre-game variables.

    Remakes in the history are ignored. A player without usable history (brand-new, only remakes,
    or blocked by Riot's firewall: the export can't tell these apart, ~1 % of players) keeps NaN in
    the variables and gets sin_historial = 1. Such a player is not counted as off-role.
    With `pool` (see champion_pool.py) the column winrate_campeon_posicion is added: the pool win rate of the
    player's champion in the player's (inferred) position, from pool games before the match start.
    """
    done = matches[matches["status"] == "history_done"]
    target_start = done.set_index("match_id")["game_start_ts"]

    h = history[history["target_match_id"].isin(target_start.index) & (history["remake"] == 0)]
    # Leakage guard (the collector already enforces it; checked again here on purpose).
    h = h[h["game_start_ts"] < h["target_match_id"].map(target_start)]
    h = h.sort_values(["target_match_id", "player_id", "history_rank"])  # newest first
    h = h.assign(kda=(h["kills"] + h["assists"]) / h["deaths"].clip(lower=1))

    rows = []
    for (match_id, player_id), games in h.groupby(["target_match_id", "player_id"], sort=False):
        rows.append({
            "match_id": match_id,
            "player_id": player_id,
            "partidas_previas": len(games),
            "winrate_10": games["win"].mean(),
            "linea_habitual": _usual_position(games["team_position"]),
            "campeones_previos": games["champion_id"].tolist(),
            "racha": racha_actual(games["win"].tolist()),
            "dias_desde_ultima": (target_start[match_id] - games["game_start_ts"].iat[0]) / 86_400_000,
            "kda_medio": games["kda"].mean(),
            "kills_medio": games["kills"].mean(),
            "partidas_24h": int((games["game_start_ts"] > target_start[match_id] - DAY_MS).sum()),
        })
    per_player = pd.DataFrame(rows, columns=["match_id", "player_id", "partidas_previas", "winrate_10",
                                             "linea_habitual", "campeones_previos", "racha",
                                             "dias_desde_ultima", "kda_medio", "kills_medio", "partidas_24h"])

    feats = participants[participants["match_id"].isin(target_start.index)].merge(
        per_player, on=["match_id", "player_id"], how="left")
    # An all-empty history leaves these as object columns: make them numeric so the maths below works.
    numeric = ["winrate_10", "racha", "dias_desde_ultima", "kda_medio", "kills_medio", "partidas_24h"]
    feats[numeric] = feats[numeric].astype(float)
    feats["partidas_previas"] = feats["partidas_previas"].fillna(0).astype(int)
    feats["sin_historial"] = (feats["partidas_previas"] == 0).astype(int)
    feats["en_linea_habitual"] = np.where(
        feats["linea_habitual"].isna(), np.nan,
        (feats["team_position"] == feats["linea_habitual"]).astype(float))
    feats["partidas_con_campeon"] = [
        prev.count(champion) if isinstance(prev, list) else np.nan
        for prev, champion in zip(feats["campeones_previos"], feats["champion_id"])]
    feats["log_dias_desde_ultima"] = np.log1p(feats["dias_desde_ultima"])
    if pool is not None:
        look = champion_pool.ChampionPool(pool)
        feats["winrate_campeon_posicion"] = [
            look.champion_position_rate(champion, position, target_start[match_id])
            for match_id, champion, position in zip(feats["match_id"], feats["champion_id"], feats["team_position"])]
    return feats.drop(columns=["campeones_previos"])


def match_features(player_feats: pd.DataFrame, matches: pd.DataFrame, final_features: bool = False) -> pd.DataFrame:
    """One row per match: blue minus red for every variable, as team averages and lane by lane.

    final_features=True also gives dif_ of NEW_PLAYER_VARS (player_features needs a pool for the last one).
    A team's value is the mean over its players, skipping players whose value is missing.
    """
    team_vars = PLAYER_VARS + (NEW_PLAYER_VARS if final_features else [])
    team = player_feats.groupby(["match_id", "team_id"])[team_vars].mean().unstack("team_id")
    out = pd.DataFrame({f"dif_{var}": team[(var, BLUE)] - team[(var, RED)] for var in team_vars})

    off_role = (player_feats["en_linea_habitual"] == 0).groupby(
        [player_feats["match_id"], player_feats["team_id"]]).sum().unstack("team_id")
    out["dif_fuera_de_posicion"] = off_role[BLUE] - off_role[RED]

    lanes = player_feats.pivot_table(index="match_id", columns=["team_id", "team_position"],
                                     values=LANE_VARS, aggfunc="first", dropna=False)
    for position in POSITIONS:
        for var in LANE_VARS:
            blue = lanes.get((var, BLUE, position))
            red = lanes.get((var, RED, position))
            name = f"{position.lower()}_dif_{var}"
            out[name] = (blue - red) if blue is not None and red is not None else np.nan

    info = matches.set_index("match_id")[["blue_win", "seed_tier", "game_start_ts", "patch"]]
    return info.join(out, how="inner").reset_index()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model's input columns (everything that is a blue-minus-red difference)."""
    return [c for c in df.columns if c.startswith("dif_") or "_dif_" in c]


def inferred_participants(participants: pd.DataFrame, history: pd.DataFrame, matches: pd.DataFrame,
                          spells: pd.DataFrame, champion_positions: dict | None = None) -> pd.DataFrame:
    """Copy of `participants` whose team_position is INFERRED before the match, not Riot's value.

    Uses only the summoner spells, the player's positions in games strictly before the match
    (remakes and blanks ignored) and, when `champion_positions` is given, the champion played.
    Riot's team_position is never read, so the result cannot depend on how the match went. Every match gets inferred roles; player_features later drops matches that
    are not history_done. Players without spell data get no spell evidence (a real jungler can then lose
    the role), so a warning is printed when any participant has none.
    """
    start = matches.set_index("match_id")["game_start_ts"]
    h = history[(history["remake"] == 0) & history["target_match_id"].isin(start.index)]
    h = h[h["game_start_ts"] < h["target_match_id"].map(start)]  # strictly before the match
    h = h[h["team_position"].isin(POSITIONS)].sort_values(["target_match_id", "player_id", "history_rank"])
    past = {key: list(g) for key, g in h.groupby(["target_match_id", "player_id"])["team_position"]}
    picked = {(m, p): tuple(int(x) for x in (a, b) if pd.notna(x))
              for m, p, a, b in zip(spells["match_id"], spells["player_id"],
                                    spells["spell1_id"], spells["spell2_id"])}
    no_spells = sum(1 for m, p in zip(participants["match_id"], participants["player_id"])
                    if not picked.get((m, p)))
    if no_spells:
        warnings.warn(f"{no_spells} participantes sin hechizos: sus roles inferidos serán menos fiables. "
                      "Vuelve a exportar (scripts.export_dataset) con las partidas en caché.")
    result = participants.copy()
    result["team_position"] = ""
    for (match_id, _team), group in participants.groupby(["match_id", "team_id"]):
        team = [{"key": player, "spell_ids": picked.get((match_id, player), ()),
                 "history_positions": past.get((match_id, player), []), "champion_id": champion}
                for player, champion in zip(group["player_id"], group["champion_id"])]
        assigned = roles.infer_positions(team, champion_positions)
        result.loc[group.index, "team_position"] = [assigned[p] for p in group["player_id"]]
    return result


def build(export_dir: Path = EXPORT_DIR, inferred_positions: bool = False, *, pool: pd.DataFrame | None = None,
          champion_positions: dict | None = None, final_features: bool = False) -> pd.DataFrame:
    """Read the exported CSVs and return the match-level feature table.

    inferred_positions=False (default): positions are Riot's post-game teamPosition (as in the notebook).
    inferred_positions=True: positions come from roles.infer_positions (spells + earlier games), the
    only information a real prediction has. Needs participant_spells.csv (scripts.export_dataset).
    champion_positions (inferred positions only): table from champion_pool.champion_position_shares; the
    champion then also counts in role inference.
    final_features=True: also add the three variables of the frozen set (dif_kills_medio, dif_partidas_24h,
    dif_winrate_campeon_posicion); needs `pool` (champion_pool.build_pool). Default output is unchanged.
    """
    if final_features and pool is None:
        raise ValueError("final_features=True necesita un pool de campeones (champion_pool.build_pool).")
    export_dir = Path(export_dir)
    matches = pd.read_csv(export_dir / "matches.csv", dtype={"patch": str})  # keep "16.20", not 16.2
    participants = pd.read_csv(export_dir / "participants.csv")
    history = pd.read_csv(export_dir / "player_history.csv")
    if inferred_positions:
        spells_path = export_dir / "participant_spells.csv"
        if not spells_path.exists():
            raise FileNotFoundError(
                f"Falta {spells_path}, necesario para inferir posiciones. "
                "Créalo con: python -m scripts.export_dataset")
        participants = inferred_participants(participants, history, matches, pd.read_csv(spells_path),
                                             champion_positions)
    return match_features(player_features(history, participants, matches, pool if final_features else None),
                          matches, final_features)


def main():
    table = build()
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURES_PATH, index=False)
    print(f"{len(table)} matches, {len(feature_columns(table))} features -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()
