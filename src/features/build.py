"""
Feature building: turn the exported tables into one row per match, ready for a model.

Everything here is computed from each player's games BEFORE the target match (player_history),
so nothing the model sees comes from the match itself or from later games.

Two levels:
  player_features()  one row per (match, player): recent form, lane fit, champion comfort...
  match_features()   one row per match: blue minus red, as a team and lane against lane

Lane against lane is the new idea after the EDA: on their own, absolute numbers (a 60 % win rate)
said nothing about who wins, so we compare each player with the opponent in the same position.

Run it:   python -m src.features.build        # reads data/export/*.csv, writes data/features/partidas.csv
Tests:    pytest tests/test_features.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd

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
    """The position the player repeats most (ties -> alphabetical, to stay deterministic)."""
    known = positions[positions.fillna("") != ""]
    if known.empty:
        return np.nan
    return known.mode().iat[0]


def player_features(history: pd.DataFrame, participants: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per participant of each match that has its history, with the pre-game variables.

    Remakes in the history are ignored. A player without usable history (brand-new, or blocked by
    Riot's firewall) keeps NaN in the variables and gets sin_historial = 1.
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
        })
    per_player = pd.DataFrame(rows, columns=["match_id", "player_id", "partidas_previas", "winrate_10",
                                             "linea_habitual", "campeones_previos", "racha",
                                             "dias_desde_ultima", "kda_medio"])

    feats = participants[participants["match_id"].isin(target_start.index)].merge(
        per_player, on=["match_id", "player_id"], how="left")
    feats["partidas_previas"] = feats["partidas_previas"].fillna(0).astype(int)
    feats["sin_historial"] = (feats["partidas_previas"] == 0).astype(int)
    feats["en_linea_habitual"] = np.where(
        feats["linea_habitual"].isna(), np.nan,
        (feats["team_position"] == feats["linea_habitual"]).astype(float))
    feats["partidas_con_campeon"] = [
        prev.count(champion) if isinstance(prev, list) else np.nan
        for prev, champion in zip(feats["campeones_previos"], feats["champion_id"])]
    feats["log_dias_desde_ultima"] = np.log1p(feats["dias_desde_ultima"])
    return feats.drop(columns=["campeones_previos"])


def match_features(player_feats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match: blue minus red for every variable, as team averages and lane by lane."""
    team = player_feats.groupby(["match_id", "team_id"])[PLAYER_VARS].mean().unstack("team_id")
    out = pd.DataFrame({f"dif_{var}": team[(var, BLUE)] - team[(var, RED)] for var in PLAYER_VARS})

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


def build(export_dir: Path = EXPORT_DIR) -> pd.DataFrame:
    """Read the exported CSVs and return the match-level feature table."""
    matches = pd.read_csv(export_dir / "matches.csv", dtype={"patch": str})  # keep "16.20", not 16.2
    participants = pd.read_csv(export_dir / "participants.csv")
    history = pd.read_csv(export_dir / "player_history.csv")
    return match_features(player_features(history, participants, matches), matches)


def main():
    table = build()
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURES_PATH, index=False)
    print(f"{len(table)} matches, {len(feature_columns(table))} features -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()
