"""
Frozen 10-variable model, prediction of played and live matches, and the forward test.

Order of use (see README):
  python -m src.models.predict --freeze              # ONCE: fit the candidate on the 920 matches of the saved 22-variable
                                                     #   model, keep that model as modelo_base_22.joblib, freeze the test
  python -m src.models.predict --evaluate-new        # candidate vs base on matches that are NOT in the 1,206 development
                                                     #   matches and were played after the last of the 920 (never refits)
  python -m src.models.predict --train               # only after a test result: refit on every usable match, save modelo_final
  python -m src.models.predict --evaluate-final      # second test of the shipped bundle on new matches; never refits
  python -m src.models.predict LA1_123456            # probability that blue wins that (played) match
  python -m src.models.predict --live "Name#TAG"     # a game in progress (spectator-v5), BEFORE it ends
  python -m src.models.predict --score-log           # compare the logged live predictions with the real results

The model. Logistic regression (the existing C grid and tier weights) on the 10 variables of build.FINAL_FEATURES.
The bundle (version 2) also holds the champion pool and the champion-by-position shares it was built with, the
ids of the development matches and a fitted 9-variable model (a stated sensitivity check). Bundles without
version 2 are refused by check_bundle, except as the base of --evaluate-new.

Roles. Riot assigns teamPosition after the game and the live endpoint gives no position, so the model uses
INFERRED roles everywhere (src/features/roles.py: summoner spells, earlier games and the champion played).
Training, played-match prediction and live prediction share one path (score_game), so they always agree.

Live limits: the game must be ranked solo/duo (queue 420) and in progress; a hidden or firewall-blocked
player has no history; collecting data (run_pipeline) and --live must not run at the same time, because
each process has its own rate limiter.

--freeze, --evaluate-new, --evaluate-final and --train read data/export and write only to data/models. None of them calls the
Riot API or opens data/db/lol.sqlite, and none writes to data/export.
Tests:    pytest tests/test_predict.py -v
"""
import argparse
import csv
import filecmp
import json
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import requests
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.features import build, champion_pool
from src.models import train

# src.agents / src.riot_client are imported only where a match is downloaded: their config needs
# RIOT_API_KEY, and --train makes no API calls (tests/test_predict.py checks this without a key).

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODEL_PATH = MODELS_DIR / "modelo_final.joblib"
BASE_NAME, CANDIDATE_NAME = "modelo_base_22.joblib", "modelo_candidato.joblib"
TEST_RESULT_NAME, DEVELOPMENT_NAME = "prueba_final.json", "conjunto_desarrollo.csv"
LOGISTIC = "Regresión logística"
LOG_PATH = PROJECT_ROOT / "data" / "predictions" / "live_log.csv"
LOG_COLUMNS = ["game_id", "platform", "timestamp", "prob_blue", "model_trained_at"]
RANKED_SOLO_QUEUE = 420
MIN_FINISHED_FOR_AUC = 10
BUNDLE_VERSION = 2
BASE_FEATURE_COUNT, BASE_MATCH_COUNT, DEVELOPMENT_COUNT = 22, 920, 1206  # what --freeze expects to find
LOW_POWER_N = 1000
SECOND_RESULT_NAME = "segunda_prueba.json"
SECOND_CUTOFF_TS = int(datetime(2026, 11, 5, 18, tzinfo=timezone.utc).timestamp() * 1000)
RETRAIN_MESSAGE = ("El modelo guardado es de una versión anterior (sin las 10 variables congeladas) o usa posiciones "
                   "de Riot. Congélalo con: python -m src.models.predict --freeze (una vez); después de la prueba, "
                   "reentrena con: python -m src.models.predict --train")

HISTORY_COLUMNS = ["target_match_id", "player_id", "history_rank", "history_match_id", "game_start_ts",
                   "game_duration", "remake", "win", "champion_id", "champion_name", "team_position",
                   "kills", "deaths", "assists"]
META_COLUMNS = ["match_id", "blue_win", "seed_tier", "game_start_ts"]


def _fit_logistic(table: pd.DataFrame, features: list[str]):
    search = train.logistic_model()
    search.fit(table[features], table["blue_win"], model__sample_weight=train.tier_weights(table["seed_tier"]))
    return search


def fit_final_model(table: pd.DataFrame, pool: pd.DataFrame, champion_positions: dict,
                    development_ids=()) -> dict:
    """Fit the frozen 10-variable logistic regression on ALL rows of `table` and describe it.

    `table` comes from build.build(inferred_positions=True, final_features=True, ...). `pool` and
    `champion_positions` are the ones that table was built with; they travel in the bundle so that a
    played or live game is scored with the same pool and role table. A 9-variable model is fitted too.
    """
    missing = [f for f in build.FINAL_FEATURES if f not in table.columns]
    if missing:
        raise ValueError(f"Faltan variables en la tabla: {missing}")
    reduced = table[META_COLUMNS + build.FINAL_FEATURES]
    features = list(build.FINAL_FEATURES)
    search = _fit_logistic(reduced, features)
    nine = _fit_logistic(reduced, list(build.SENSITIVITY_FEATURES))
    row = train.temporal_validation(reduced)["tabla"].loc[LOGISTIC]
    blue_rate = float(reduced["blue_win"].mean())
    return {
        "version": BUNDLE_VERSION,
        "model": search.best_estimator_,
        "features": features,
        "best_params": search.best_params_,
        "sensibilidad": {"model": nine.best_estimator_, "features": list(build.SENSITIVITY_FEATURES),
                         "best_params": nine.best_params_},
        "pool": pool,
        "champion_positions": champion_positions,
        "development_ids": sorted(str(i) for i in development_ids),
        "n_matches": int(len(reduced)),
        "match_ids": sorted(reduced["match_id"].astype(str)),
        "first_match_ts": int(reduced["game_start_ts"].min()),
        "last_match_ts": int(reduced["game_start_ts"].max()),
        "auc_temporal": float(row["roc_auc"]),
        "auc_ic95": (float(row["auc_ic95_bajo"]), float(row["auc_ic95_alto"])),
        "roles": "inferred",
        "blue_win_rate": blue_rate,
        "majority_blue_win": int(blue_rate >= 0.5),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }


def save_model(bundle: dict, path: Path = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path: Path = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No hay modelo guardado en {path}. Ejecuta primero: python -m src.models.predict --freeze")
    bundle = joblib.load(path)
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(f"El modelo se guardó con scikit-learn {bundle.get('sklearn_version')} y ahora hay "
                      f"{sklearn.__version__}; si falla o parece raro, vuelve a entrenar con --train.")
    return bundle


def check_bundle(bundle: dict) -> None:
    """Refuse anything that is not a version 2 bundle with inferred roles."""
    if bundle.get("version") != BUNDLE_VERSION or bundle.get("roles") != "inferred":
        raise ValueError(RETRAIN_MESSAGE)


# --- One path for every prediction: players + their histories -> probability ----------------


def players_from_match(match: dict) -> list[dict]:
    """Players of a match-v5 dict. Riot's teamPosition is NOT read."""
    return [{"key": p["puuid"], "team_id": p["teamId"], "champion_id": p["championId"],
             "champion_name": p["championName"], "spell_ids": (p["summoner1Id"], p["summoner2Id"])}
            for p in match["info"]["participants"]]


def features_for_game(game_id: str, start_ts: int, players: list[dict], histories: dict[str, list[dict]],
                      blue_win: int = 0, patch: str = "", pool: pd.DataFrame | None = None,
                      champion_positions: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One-row feature table and the inferred role of each player. Pure: no API, no files.

    `players`: dicts with key, team_id, champion_id, champion_name, spell_ids.
    `histories` maps key -> rows from history_builder.player_history (missing or empty = no history).
    `pool` and `champion_positions` are the model's (bundle["pool"], bundle["champion_positions"]); with a pool
    the three variables of the frozen set are built too.
    Roles come from build.inferred_participants, the feature logic from build.player_features /
    build.match_features: nothing is reimplemented here, so training and prediction cannot drift apart.
    `blue_win` only fills the label column; no feature reads it.
    """
    matches = pd.DataFrame([{"match_id": game_id, "status": "history_done", "seed_tier": "UNKNOWN",
                             "game_start_ts": start_ts, "patch": patch, "blue_win": blue_win}])
    participants = pd.DataFrame([{"match_id": game_id, "player_id": p["key"], "team_id": p["team_id"],
                                  "team_position": "", "champion_id": p["champion_id"],
                                  "champion_name": p["champion_name"]} for p in players])
    spells = pd.DataFrame([{"match_id": game_id, "player_id": p["key"],
                            "spell1_id": (list(p["spell_ids"]) + [None, None])[0],
                            "spell2_id": (list(p["spell_ids"]) + [None, None])[1]} for p in players])
    rows = [{"target_match_id": game_id, "player_id": key, **game}
            for key, games in histories.items() for game in games]
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    participants = build.inferred_participants(participants, history, matches, spells, champion_positions)
    table = build.match_features(build.player_features(history, participants, matches, pool), matches,
                                 final_features=pool is not None)
    return table, dict(zip(participants["player_id"], participants["team_position"]))


def score_game(game_id: str, start_ts: int, players: list[dict], histories: dict[str, list[dict]],
               bundle: dict, blue_win: int = 0) -> dict:
    """Probability that blue wins, from players and histories. Shared by played and live prediction."""
    check_bundle(bundle)
    table, roles = features_for_game(game_id, start_ts, players, histories, blue_win,
                                     pool=bundle["pool"], champion_positions=bundle["champion_positions"])
    missing = [f for f in bundle["features"] if f not in table.columns]
    if missing:
        raise ValueError(f"El modelo guardado usa variables que ya no existen: {missing}. Vuelve a entrenar con --train.")
    prob_blue = float(bundle["model"].predict_proba(table[bundle["features"]])[0, 1])
    return {"prob_blue": prob_blue, "prob_red": 1 - prob_blue, "roles": roles,
            "jugadores_sin_historial": sum(1 for p in players
                                          if all(game["remake"] for game in histories.get(p["key"], [])))}


def in_training_data(match_id: str, bundle: dict) -> bool:
    """True if the model was fitted on this match."""
    return match_id in bundle["match_ids"]


def predict_match(match_id: str, bundle: dict) -> dict:
    """Probability that blue wins an already-played match (downloads it and the players' histories)."""
    check_bundle(bundle)
    from src.agents import history_builder, match_fetcher
    from src.riot_client import endpoints
    match = endpoints.get_match(match_id)
    reason = match_fetcher.invalid_reason(match)
    if reason:
        raise ValueError(f"{match_id} no se puede usar: {reason}")
    start = match["info"]["gameStartTimestamp"]
    histories = {p["puuid"]: history_builder.player_history(p["puuid"], start)
                 for p in match["info"]["participants"]}
    blue_win = match_fetcher.summarize(match)["blue_win"]
    result = score_game(match_id, start, players_from_match(match), histories, bundle, blue_win)
    return {"match_id": match_id, **result, "blue_win": blue_win,
            "en_entrenamiento": in_training_data(match_id, bundle)}


# --- Live games ---------------------------------------------------------------------------


class LiveError(Exception):
    """A clear, expected reason why no live prediction can be made (not in a game, wrong queue...)."""


def parse_riot_id(text: str) -> tuple[str, str]:
    name, sep, tag = text.rpartition("#")
    if not sep or not name or not tag:
        raise LiveError('Escribe el Riot ID completo, por ejemplo "Nombre#TAG".')
    return name, tag


def champion_names(participants_csv: Path | None = None) -> dict[int, str]:
    """champion id -> name, from the exported participants (empty if the file is missing)."""
    path = participants_csv or build.EXPORT_DIR / "participants.csv"
    try:
        pairs = pd.read_csv(path, usecols=["champion_id", "champion_name"]).drop_duplicates("champion_id")
        return dict(zip(pairs["champion_id"].astype(int), pairs["champion_name"]))
    except (OSError, ValueError):
        return {}


def log_prediction(game_id, platform: str, prob_blue: float, bundle: dict, path: Path | None = None) -> bool:
    """Append one row to the live log. False (and no row) if that game is already logged."""
    path = Path(path or LOG_PATH)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            if any(r["game_id"] == str(game_id) and r["platform"] == platform for r in csv.DictReader(f)):
                return False
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([game_id, platform, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         f"{prob_blue:.6f}", bundle.get("trained_at", "")])
    return True


def predict_live(riot_id: str, bundle: dict, progress=print, log_path: Path | None = None,
                 names: dict[int, str] | None = None) -> dict:
    """Predict the game `riot_id` is playing now. Raises LiveError when there is nothing to predict."""
    check_bundle(bundle)
    from src.agents import history_builder
    from src.riot_client import config, endpoints
    name, tag = parse_riot_id(riot_id)
    try:
        puuid = endpoints.get_puuid(name, tag)
    except requests.HTTPError as error:
        if endpoints.is_not_found(error):
            raise LiveError(f"No existe el Riot ID {riot_id}.")
        raise
    if endpoints.is_firewall_blocked(puuid):
        raise LiveError("El firewall de Riot bloquea las consultas sobre este jugador; no se puede predecir su partida.")
    game = endpoints.get_active_game(puuid)
    if game is None:
        raise LiveError(f"{riot_id} no está en una partida ahora mismo.")
    if game.get("gameQueueConfigId") != RANKED_SOLO_QUEUE:
        raise LiveError(f"La partida es de la cola {game.get('gameQueueConfigId')}, no ranked solo/duo "
                        f"({RANKED_SOLO_QUEUE}); el modelo solo vale para ranked solo/duo.")
    raw = game.get("participants", [])
    if len(raw) != 10:
        raise LiveError(f"La partida tiene {len(raw)} jugadores, no 10.")
    start_ms = game.get("gameStartTime") or int(time.time() * 1000)  # 0 while the game is loading
    players, histories = [], {}
    for i, p in enumerate(raw, start=1):
        key = p.get("puuid") or f"oculto_{i}"
        players.append({"key": key, "team_id": p["teamId"], "champion_id": p["championId"],
                        "champion_name": "", "spell_ids": (p.get("spell1Id"), p.get("spell2Id"))})
        progress(f"Historial del jugador {i}/10...")
        hidden = not p.get("puuid") or endpoints.is_firewall_blocked(key)
        histories[key] = [] if hidden else history_builder.player_history(key, start_ms)
    game_id = game.get("gameId")
    platform = game.get("platformId") or config.PLATFORM.upper()
    result = score_game(f"{platform}_{game_id}", start_ms, players, histories, bundle)
    names = champion_names() if names is None else names
    teams = {}
    for p in players:
        label = names.get(p["champion_id"]) or f"campeón {p['champion_id']}"
        teams.setdefault(p["team_id"], []).append((result["roles"][p["key"]], label))
    order = {pos: n for n, pos in enumerate(build.POSITIONS)}
    teams = {team: sorted(members, key=lambda m: order[m[0]]) for team, members in teams.items()}
    logged = log_prediction(game_id, platform, result["prob_blue"], bundle, log_path)
    return {"game_id": game_id, "platform": platform, "teams": teams, "logged": logged,
            "prob_blue": result["prob_blue"], "prob_red": result["prob_red"],
            "jugadores_sin_historial": result["jugadores_sin_historial"]}


# --- Forward test of the frozen model ------------------------------------------------------


def score_log(log_path: Path | None = None, get_match=None, bundle: dict | None = None) -> dict:
    """Compare the logged live predictions with the real results (finished games only).

    With `bundle`, only rows logged by that model (model_trained_at == bundle["trained_at"]) are scored,
    so the result is one frozen model's; rows from other models are counted in "otro_modelo".
    """
    path = Path(log_path or LOG_PATH)
    if not path.exists():
        return {"n": 0, "pendientes": 0, "descartadas": 0, "accuracy": None, "auc": None, "logged": 0,
                "otro_modelo": 0}
    if get_match is None:
        from src.riot_client import endpoints
        get_match = endpoints.get_match
    from src.agents import match_fetcher
    from src.riot_client import endpoints
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    other_model = 0
    if bundle is not None:
        mine = [r for r in rows if r.get("model_trained_at") == bundle.get("trained_at", "")]
        other_model = len(rows) - len(mine)
        rows = mine
    probs, outcomes, pending, discarded = [], [], 0, 0
    for row in rows:
        try:
            match = get_match(f"{row['platform']}_{row['game_id']}")
        except requests.HTTPError as error:
            if endpoints.is_not_found(error):
                pending += 1  # not finished (or not published yet): try again later
                continue
            raise
        if match_fetcher.invalid_reason(match) in ("remake", "not 10 players", "wrong queue"):
            discarded += 1
            continue
        probs.append(float(row["prob_blue"]))
        outcomes.append(match_fetcher.summarize(match)["blue_win"])
    n = len(probs)
    y, p = np.array(outcomes), np.array(probs)
    both = n >= MIN_FINISHED_FOR_AUC and len(np.unique(y)) == 2
    return {"n": n, "logged": len(rows), "pendientes": pending, "descartadas": discarded,
            "otro_modelo": other_model, "accuracy": float(accuracy_score(y, p >= 0.5)) if n else None,
            "auc": float(roc_auc_score(y, p)) if both else None}


# --- Development set, freeze and forward test ------------------------------------------------


class PredictError(Exception):
    """A clear, expected reason why a command cannot run (shown as is, no traceback)."""


def _read_export(export_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    export_dir = Path(export_dir)
    matches = pd.read_csv(export_dir / "matches.csv", dtype={"patch": str})
    return matches, pd.read_csv(export_dir / "player_history.csv")


def _pool_and_shares(history: pd.DataFrame, match_ids) -> tuple[pd.DataFrame, dict]:
    """Pool and role shares made ONLY from the history windows of `match_ids`."""
    pool = champion_pool.build_pool(history[history["target_match_id"].isin(set(match_ids))])
    return pool, champion_pool.champion_position_shares(pool)


def _table_for(export_dir: Path, pool: pd.DataFrame, shares: dict) -> pd.DataFrame:
    return build.build(export_dir, inferred_positions=True, pool=pool, champion_positions=shares, final_features=True)


def freeze(export_dir: Path = build.EXPORT_DIR, models_dir: Path = MODELS_DIR) -> dict:
    """Run once. Keep the saved 22-variable model as the base, fit the candidate on the same matches.

    Everything is checked before anything is written. Reads data/export, writes only to models_dir.
    """
    export_dir, models_dir = Path(export_dir), Path(models_dir)
    candidate_path, final_path, base_path = (models_dir / CANDIDATE_NAME, models_dir / "modelo_final.joblib",
                                             models_dir / BASE_NAME)
    if candidate_path.exists():
        raise PredictError(f"Ya existe {candidate_path}: --freeze solo se ejecuta una vez (la prueba debe seguir "
                           "siendo la misma). Para empezar de cero hay que borrar el candidato Y restaurar el modelo base "
                           f"({base_path.name}) como {final_path.name}, a mano.")
    try:
        base = load_model(final_path)
    except FileNotFoundError as error:
        raise PredictError(str(error))
    if (base.get("version") is not None or len(base.get("features", [])) != BASE_FEATURE_COUNT
            or len(base.get("match_ids", [])) != BASE_MATCH_COUNT):
        raise PredictError(f"{final_path} no es el modelo de {BASE_FEATURE_COUNT} variables entrenado con "
                           f"{BASE_MATCH_COUNT} partidas: no se congela nada.")
    fitted_ids = {str(i) for i in base["match_ids"]}
    matches, history = _read_export(export_dir)
    development = sorted(matches.loc[matches["status"] == "history_done", "match_id"].astype(str))
    if len(development) != DEVELOPMENT_COUNT or not fitted_ids <= set(development):
        raise PredictError(f"El conjunto de desarrollo debe tener exactamente {DEVELOPMENT_COUNT} partidas "
                           f"history_done e incluir las {BASE_MATCH_COUNT} del modelo base; hay {len(development)} "
                           f"y {len(fitted_ids - set(development))} de las {BASE_MATCH_COUNT} no están. "
                           "El export cambió: no se congela.")
    if base_path.exists() and not filecmp.cmp(final_path, base_path, shallow=False):
        raise PredictError(f"{base_path} ya existe y es distinto de {final_path}: no se sobrescribe.")
    models_dir.mkdir(parents=True, exist_ok=True)
    if not base_path.exists():
        shutil.copyfile(final_path, base_path)
    if not filecmp.cmp(final_path, base_path, shallow=False):
        raise PredictError("La copia del modelo base no coincide byte a byte; no se continúa.")

    pool, shares = _pool_and_shares(history, fitted_ids)
    table = _table_for(export_dir, pool, shares)
    fit_table = table[table["match_id"].astype(str).isin(fitted_ids)]
    if len(fit_table) != len(fitted_ids):
        raise PredictError("No se pudo construir la tabla de las partidas de ajuste; no se congela.")
    bundle = fit_final_model(fit_table, pool, shares, development_ids=development)
    save_model(bundle, candidate_path)
    save_model(bundle, final_path)
    with open(models_dir / DEVELOPMENT_NAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["match_id", "uso"])
        writer.writerows((i, "ajuste" if i in fitted_ids else "seleccion") for i in development)
    return bundle


def _metrics(y, proba) -> dict:
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)
    spread = len(np.unique(y)) == 2 and len(np.unique(proba)) > 1
    return {"accuracy": float(accuracy_score(y, pred)), "accuracy_ic95": train.bootstrap_accuracy_ci(y, pred),
            "auc": float(roc_auc_score(y, proba)) if spread else None,
            "auc_ic95": train.bootstrap_auc_ci(y, proba) if spread else None,
            "log_loss": float(log_loss(y, proba, labels=[0, 1]))}


def evaluate_new(base: dict, candidate: dict, base_table: pd.DataFrame, candidate_table: pd.DataFrame) -> dict:
    """Forward test of the frozen candidate against the saved base. Nothing is refitted.

    Test set: matches that are not in candidate["development_ids"] and were played after the last match
    the models were fitted on. Both tables must contain the same matches; the base table is built with
    inferred roles but WITHOUT the champion, exactly as the base was trained.
    """
    check_bundle(candidate)
    if base.get("roles") != "inferred":
        raise ValueError("El modelo base debe usar posiciones inferidas.")
    development = set(candidate["development_ids"])
    cutoff = candidate["last_match_ts"]
    ids = candidate_table["match_id"].astype(str)
    in_development = ids.isin(development)
    too_early = ~in_development & (candidate_table["game_start_ts"] <= cutoff)
    result = {"n": 0, "excluidas_desarrollo": int(in_development.sum()), "excluidas_por_fecha": int(too_early.sum())}
    test = candidate_table[~in_development & ~too_early].sort_values(["game_start_ts", "match_id"], kind="stable")
    if test.empty:
        return result
    test_ids = test["match_id"].astype(str)
    if not set(test_ids) <= set(base_table["match_id"].astype(str)):
        raise ValueError("Las tablas del modelo base y del candidato no tienen las mismas partidas.")
    base_test = base_table.set_index(base_table["match_id"].astype(str)).loc[test_ids]
    if (base_test["blue_win"].to_numpy() != test["blue_win"].to_numpy()).any():
        raise ValueError("Las tablas del modelo base y del candidato no tienen el mismo resultado en las mismas partidas.")
    for bundle, table in ((base, base_test), (candidate, test), (candidate["sensibilidad"], test)):
        missing = [f for f in bundle["features"] if f not in table.columns]
        if missing:
            raise ValueError(f"Faltan variables en la tabla de prueba: {missing}")
    y = test["blue_win"].to_numpy()
    proba = {
        "base": base["model"].predict_proba(base_test[base["features"]])[:, 1],
        "candidato": candidate["model"].predict_proba(test[candidate["features"]])[:, 1],
        "nueve": candidate["sensibilidad"]["model"].predict_proba(test[candidate["sensibilidad"]["features"]])[:, 1],
        "constante": np.full(len(y), candidate["blue_win_rate"]),
    }
    models = {name: _metrics(y, p) for name, p in proba.items()}
    day = lambda ts: datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d")
    differences = {
        "log_loss": train.paired_log_loss_difference(y, proba["base"], proba["candidato"]),
        "auc": (train.paired_auc_difference(y, proba["candidato"], proba["base"]) if len(np.unique(y)) == 2
                else {"diferencia": None, "ic95": None}),  # one class only: no AUC to compare
        "accuracy": train.paired_accuracy_difference(y, proba["candidato"], proba["base"]),
        "log_loss_nueve": train.paired_log_loss_difference(y, proba["base"], proba["nueve"]),
    }
    low, high = differences["log_loss"]["ic95"]
    if low > 0:
        verdict = (f"El candidato es mejor que el modelo base: la pérdida logarítmica baja {differences['log_loss']['diferencia']:.4f} "
                   f"(IC 95 %: {low:.4f} a {high:.4f}), por encima de 0.")
    else:
        verdict = (f"No se demuestra que el candidato sea mejor que el modelo base: la diferencia de pérdida logarítmica es "
                   f"{differences['log_loss']['diferencia']:+.4f} (IC 95 %: {low:+.4f} a {high:+.4f}) y el intervalo no queda por encima de 0.")
    result.update({
        "n": int(len(test)), "desde": day(test["game_start_ts"].min()), "hasta": day(test["game_start_ts"].max()),
        "por_tier": {str(k): int(v) for k, v in test["seed_tier"].value_counts().sort_index().items()},
        "modelos": models, "diferencias": differences, "veredicto": verdict,
        "poca_potencia": bool(len(test) < LOW_POWER_N), "mejor": bool(low > 0),
    })
    return result


def evaluate_final(final: dict, candidate: dict, final_table: pd.DataFrame, candidate_table: pd.DataFrame) -> dict:
    """Second preregistered test of the shipped bundle; prediction only, never fit."""
    check_bundle(final)
    check_bundle(candidate)
    for table in (final_table, candidate_table):
        if table.match_id.astype(str).duplicated().any():
            raise ValueError("Las tablas contienen partidas duplicadas.")
    if set(final_table.match_id.astype(str)) != set(candidate_table.match_id.astype(str)):
        raise ValueError("Las tablas no tienen las mismas partidas.")
    aligned = candidate_table.set_index(candidate_table.match_id.astype(str)).loc[final_table.match_id.astype(str)]
    if not np.array_equal(aligned.blue_win.to_numpy(), final_table.blue_win.to_numpy()):
        raise ValueError("Las tablas no tienen el mismo resultado.")
    ids = final_table.match_id.astype(str)
    fitted = ids.isin(set(map(str, final["match_ids"])))
    development = ~fitted & ids.isin(set(map(str, final["development_ids"])))
    early = ~fitted & ~development & ~(final_table.game_start_ts > max(final["last_match_ts"], SECOND_CUTOFF_TS))
    result = dict(n=0, excluidas_ajuste=int(fitted.sum()), excluidas_desarrollo=int(development.sum()),
                  excluidas_por_fecha=int(early.sum()))
    test = final_table[~fitted & ~development & ~early].sort_values(["game_start_ts", "match_id"], kind="stable")
    if test.empty:
        return result
    other = candidate_table.set_index(candidate_table.match_id.astype(str)).loc[test.match_id.astype(str)]
    for bundle, table in ((final, test), (candidate, other)):
        missing = [f for f in bundle["features"] if f not in table]
        if missing:
            raise ValueError(f"Faltan variables en la tabla de prueba: {missing}")
    y = test.blue_win.to_numpy()
    probabilities = dict(final=final["model"].predict_proba(test[final["features"]])[:, 1],
                         candidato=candidate["model"].predict_proba(other[candidate["features"]])[:, 1],
                         constante=np.full(len(y), final["blue_win_rate"]))
    difference = train.paired_log_loss_difference(y, probabilities["constante"], probabilities["final"])
    better = difference["ic95"][0] > 0
    day = lambda t: datetime.fromtimestamp(t/1000, timezone.utc).isoformat()
    selective = {}
    for name, p in probabilities.items():
        frame = train.selective_accuracy(y, p).astype(object)
        selective[name] = frame.where(pd.notna(frame), None).to_dict("records")
    result.update(modelo_trained_at=final.get("trained_at"), n=len(test), desde=day(test.game_start_ts.min()), hasta=day(test.game_start_ts.max()),
                  por_tier={str(k): int(v) for k, v in test.seed_tier.value_counts().sort_index().items()},
                  modelos={k: _metrics(y, p) for k, p in probabilities.items()},
                  calibracion={k: train.calibration_summary(y, p) for k, p in probabilities.items()},
                  selectiva=selective,
                  diferencias={"log_loss": difference}, mejor=bool(better), poca_potencia=len(test) < LOW_POWER_N,
                  veredicto=("El modelo final es mejor que la constante: el IC 95 % de la reducción de pérdida logarítmica queda por encima de 0."
                             if better else "No se demuestra que el modelo final sea mejor que la constante: el IC 95 % no queda por encima de 0."))
    return result


def evaluate_final_command(export_dir: Path = build.EXPORT_DIR, models_dir: Path = MODELS_DIR, out=print) -> dict:
    """Read exports and each model's frozen pool, append only to segunda_prueba.json."""
    export_dir, models_dir = Path(export_dir), Path(models_dir)
    try:
        final = load_model(models_dir / "modelo_final.joblib")
        candidate = load_model(models_dir / CANDIDATE_NAME)
        check_bundle(final)
        check_bundle(candidate)
        tables = [_table_for(export_dir, b["pool"], b["champion_positions"]) for b in (final, candidate)]
        r = evaluate_final(final, candidate, *tables)
    except (FileNotFoundError, ValueError) as error:
        raise PredictError(str(error))
    earlier = len(_read_test_runs(models_dir, SECOND_RESULT_NAME))
    out(f"Segunda prueba: ejecuciones anteriores {earlier}.")
    out(f"Excluidas por ajuste: {r['excluidas_ajuste']}; por desarrollo (sin contar ajuste): {r['excluidas_desarrollo']}; por fecha: {r['excluidas_por_fecha']}.")
    if not r["n"]:
        out("0 partidas de prueba. No se escribió nada. Exporta de nuevo cuando termine la recolección.")
        return r
    out(f"n = {r['n']}; fechas UTC: {r['desde']} a {r['hasta']}")
    out("Por liga: " + ", ".join(f"{t} {n}" for t, n in r["por_tier"].items()))
    if r["poca_potencia"]:
        out(f"AVISO: menos de {LOW_POWER_N} partidas; poca potencia para una mejora pequeña (referencia: 0.0075 en 1767).")
    names = {"final": f"Modelo final ({len(final['features'])} variables)",
             "candidato": "Candidato congelado (solo referencia de deriva)", "constante": "Constante (tasa del entrenamiento)"}
    interval = lambda ci, fmt: "sin IC" if ci is None else f"{fmt.format(ci[0])} a {fmt.format(ci[1])}"
    for key, label in names.items():
        m = r["modelos"][key]
        auc = "ROC AUC no calculable" if m["auc"] is None else f"ROC AUC {m['auc']:.3f} ({interval(m['auc_ic95'], '{:.3f}')})"
        out(f"{label}: accuracy {m['accuracy']:.1%} ({interval(m['accuracy_ic95'], '{:.1%}')}); {auc}; log loss {m['log_loss']:.4f}")
        c = r["calibracion"][key]
        parts = []
        for name, label_c in (("brier", "Brier"), ("ece", "ECE (10 intervalos)"), ("intercept", "intercepto"), ("slope", "pendiente")):
            e = c[name]["estimate"]
            parts.append(f"{label_c} " + ("no calculable" if e is None else f"{e:.3f} ({interval(c[name]['ic95'], '{:.3f}')})"))
        out("  Calibración: " + "; ".join(parts))
        for row in r["selectiva"][key]:
            acc = "sin partidas" if row["accuracy"] is None else f"accuracy {row['accuracy']:.1%} ({row['lo']:.1%} a {row['hi']:.1%})"
            out(f"  Confianza >= {row['threshold']:.0%}: {row['n']} partidas ({row['coverage']:.1%}); {acc}")
    d = r["diferencias"]["log_loss"]
    out(f"Constante menos modelo final, log loss: {d['diferencia']:+.4f} (IC 95 %: {d['ic95'][0]:+.4f} a {d['ic95'][1]:+.4f}); positivo = mejor el modelo final.")
    out("Candidato: referencia de deriva solamente. " + r["veredicto"])
    record_test_run(r, models_dir, SECOND_RESULT_NAME)
    out(f"Guardado en {models_dir / SECOND_RESULT_NAME}; repetir con los mismos datos no añade evidencia.")
    return r


def _read_test_runs(models_dir: Path, file_name: str = TEST_RESULT_NAME) -> list:
    path = Path(models_dir) / file_name
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("runs", [])


def record_test_run(result: dict, models_dir: Path = MODELS_DIR, file_name: str = TEST_RESULT_NAME) -> int:
    """Append a nonempty run to its own result file; return the earlier run count."""
    runs = _read_test_runs(models_dir, file_name)
    if not result["n"]:
        return len(runs)
    keep = ("n", "desde", "hasta", "por_tier", "excluidas_desarrollo", "excluidas_por_fecha", "modelos",
            "diferencias", "veredicto", "excluidas_ajuste", "calibracion", "selectiva", "mejor", "poca_potencia", "modelo_trained_at")
    runs.append({"fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"), **{k: result[k] for k in keep if k in result}})
    path = Path(models_dir) / file_name
    path.write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(runs) - 1


def evaluate_new_command(export_dir: Path = build.EXPORT_DIR, models_dir: Path = MODELS_DIR, out=print) -> dict:
    """Load base and candidate (never modelo_final), build both tables, test, print and record."""
    export_dir, models_dir = Path(export_dir), Path(models_dir)
    try:
        base = load_model(models_dir / BASE_NAME)
        candidate = load_model(models_dir / CANDIDATE_NAME)
        check_bundle(candidate)
    except (FileNotFoundError, ValueError) as error:
        raise PredictError(str(error))
    try:
        base_table = build.build(export_dir, inferred_positions=True)
        candidate_table = _table_for(export_dir, candidate["pool"], candidate["champion_positions"])
    except FileNotFoundError as error:
        raise PredictError(str(error))
    r = evaluate_new(base, candidate, base_table, candidate_table)
    out(f"Excluidas {r['excluidas_desarrollo']} partidas del conjunto de desarrollo (ya se usaron para elegir las variables) "
        f"y {r['excluidas_por_fecha']} jugadas antes de que terminara el ajuste.")
    if r["n"] == 0:
        out("0 partidas de prueba (fuera del conjunto de desarrollo y posteriores al ajuste). Recolecta más datos, "
            "vuelve a exportar cuando la recolección haya terminado (ver README) y repite. No se escribió nada.")
        return r
    out(f"n = {r['n']} partidas de prueba, del {r['desde']} al {r['hasta']}")
    out("Por liga: " + ", ".join(f"{t} {n}" for t, n in r["por_tier"].items()))
    if r["poca_potencia"]:
        out(f"AVISO: con menos de {LOW_POWER_N} partidas la comparación tiene poca potencia: una diferencia real "
            "pequeña puede no verse.")
    names = {"base": f"Base ({len(base['features'])} variables)", "candidato": f"Candidato ({len(candidate['features'])} variables)",
             "nueve": "Sensibilidad (9 variables)", "constante": "Constante (tasa del entrenamiento)"}
    for key, label in names.items():
        m = r["modelos"][key]
        a_low, a_high = m["accuracy_ic95"]
        auc = ("ROC AUC no calculable" if m["auc"] is None else
               f"ROC AUC {m['auc']:.3f} ({m['auc_ic95'][0]:.3f} a {m['auc_ic95'][1]:.3f})")
        out(f"{label}: accuracy {m['accuracy']:.1%} ({a_low:.1%} a {a_high:.1%}); {auc}; log loss {m['log_loss']:.4f}")
    d = r["diferencias"]
    out("Diferencias emparejadas, candidato frente a base (positivo = mejor el candidato):")
    out(f"  log loss: {d['log_loss']['diferencia']:+.4f} (IC 95 %: {d['log_loss']['ic95'][0]:+.4f} a {d['log_loss']['ic95'][1]:+.4f})")
    if d["auc"]["diferencia"] is None:
        out("  ROC AUC: no calculable (todas las partidas de prueba tienen el mismo resultado)")
    else:
        out(f"  ROC AUC: {d['auc']['diferencia']:+.4f} (IC 95 %: {d['auc']['ic95'][0]:+.4f} a {d['auc']['ic95'][1]:+.4f})")
    out(f"  accuracy: {d['accuracy']['diferencia']:+.4f} (IC 95 %: {d['accuracy']['ic95'][0]:+.4f} a {d['accuracy']['ic95'][1]:+.4f})")
    out(f"Sensibilidad (9 variables frente a base), log loss: {d['log_loss_nueve']['diferencia']:+.4f} "
        f"(IC 95 %: {d['log_loss_nueve']['ic95'][0]:+.4f} a {d['log_loss_nueve']['ic95'][1]:+.4f}); no es una segunda oportunidad.")
    out(f"Veredicto (criterio fijado antes de la prueba): {r['veredicto']}")
    earlier = record_test_run(r, models_dir)
    out(f"Guardado en {models_dir / TEST_RESULT_NAME}. Ejecuciones anteriores de esta prueba: {earlier}"
        + ("; repetirla con los mismos datos no añade evidencia." if earlier else "."))
    return r


def train_command(export_dir: Path = build.EXPORT_DIR, models_dir: Path = MODELS_DIR, force: bool = False,
                  out=print) -> dict:
    """Refit on every usable match with a pool from all of them; saves only modelo_final.joblib."""
    export_dir, models_dir = Path(export_dir), Path(models_dir)
    if not _read_test_runs(models_dir):
        if not force:
            raise PredictError(f"No hay resultado de la prueba ({models_dir / TEST_RESULT_NAME}). Ejecuta antes "
                               "--evaluate-new; reentrenar antes contaminaría la prueba. (--force lo omite.)")
        out("ADVERTENCIA: --force sin resultado de la prueba. Reentrenar con todas las partidas CONTAMINA la prueba "
            "independiente: este modelo ya no se podrá evaluar con las partidas usadas.")
    try:
        matches, history = _read_export(export_dir)
        done = matches.loc[matches["status"] == "history_done", "match_id"].astype(str)
        pool, shares = _pool_and_shares(history, done)
        table = _table_for(export_dir, pool, shares)
    except FileNotFoundError as error:
        raise PredictError(str(error))
    candidate_path = models_dir / CANDIDATE_NAME
    development = load_model(candidate_path)["development_ids"] if candidate_path.exists() else []
    bundle = fit_final_model(table, pool, shares, development_ids=development)
    path = save_model(bundle, models_dir / "modelo_final.joblib")
    low, high = bundle["auc_ic95"]
    out(f"Modelo guardado en {path}")
    out(f"{bundle['n_matches']} partidas, {len(bundle['features'])} variables, {bundle['best_params']}, posiciones inferidas")
    out(f"ROC AUC en validación temporal: {bundle['auc_temporal']:.3f} (IC 95 %: {low:.3f} a {high:.3f})")
    return bundle


# --- Commands -----------------------------------------------------------------------------


def _weak_signal_warning(bundle: dict, models_dir: Path = MODELS_DIR) -> str:
    low, high = bundle["auc_ic95"]
    text = ("Aviso: es una señal débil. En validación temporal el modelo tiene ROC AUC "
            f"{bundle['auc_temporal']:.3f} (IC 95 %: {low:.3f} a {high:.3f}); 0.5 equivale a adivinar. "
            "Esa cifra es optimista: las variables se eligieron con esas mismas partidas.")
    second = [run for run in _read_test_runs(models_dir, SECOND_RESULT_NAME)
              if run.get("modelo_trained_at") == bundle.get("trained_at")]
    if second:
        last = second[-1]
        auc = last["modelos"]["final"]["auc"]
        return text + (f" Segunda prueba independiente de este modelo ({last['fecha'][:10]}, n = {last['n']}): "
                       + (f"ROC AUC {auc:.3f}." if auc is not None else "ROC AUC no calculable."))
    runs = _read_test_runs(models_dir)
    if not runs:
        return text + " Todavía no hay una prueba independiente."
    last = runs[-1]
    auc = last["modelos"]["candidato"]["auc"]
    return text + (f" Última prueba independiente del candidato congelado, no de este modelo reentrenado ({last['fecha'][:10]}, n = {last['n']}): "
                   + (f"ROC AUC {auc:.3f}." if auc is not None else "ROC AUC no calculable."))


def _safe_error(error: Exception) -> str:
    """Error text without the URL (it can contain a puuid): only the HTTP status or a fixed message."""
    response = getattr(error, "response", None)
    if isinstance(error, requests.HTTPError) and response is not None:
        return f"Riot respondió con el código HTTP {response.status_code} (la clave puede haber caducado)."
    return "Riot o la conexión fallaron repetidamente; vuelve a intentarlo más tarde."


def _load_checked() -> dict:
    try:
        bundle = load_model()
        check_bundle(bundle)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
    return bundle


def _predict_command(match_id: str) -> None:
    from src.riot_client.client import RiotUnavailableError
    bundle = _load_checked()
    print("Descargando la partida y el historial de los 10 jugadores (puede tardar un par de minutos)...")
    try:
        result = predict_match(match_id, bundle)
    except (ValueError, requests.HTTPError, RiotUnavailableError) as error:
        raise SystemExit(f"No se pudo predecir {match_id}: "
                         f"{error if isinstance(error, ValueError) else _safe_error(error)}")
    real = "azul" if result["blue_win"] else "rojo"
    if result["en_entrenamiento"]:
        print("ADVERTENCIA: el modelo se ajustó con esta partida. La probabilidad es dentro de la muestra "
              "y NO es una predicción real: el ROC AUC de abajo no la describe.")
    print(f"Probabilidad de que gane el equipo azul: {result['prob_blue']:.1%}")
    print(f"Resultado real: ganó el equipo {real}")
    print(f"Jugadores sin historial: {result['jugadores_sin_historial']}")
    print(_weak_signal_warning(bundle))


def _live_command(riot_id: str) -> None:
    from src.riot_client.client import RiotUnavailableError
    bundle = _load_checked()
    print("Buscando la partida en curso (el historial de los 10 jugadores tarda de 1 a 3 minutos)...")
    try:
        result = predict_live(riot_id, bundle)
    except LiveError as error:
        raise SystemExit(str(error))
    except (requests.HTTPError, RiotUnavailableError) as error:
        raise SystemExit(f"No se pudo predecir la partida en vivo: {_safe_error(error)}")
    for team_id, label in ((100, "Equipo azul"), (200, "Equipo rojo")):
        print(f"{label}:")
        for position, champion in result["teams"].get(team_id, []):
            print(f"  {position:<8} {champion}")
    print(f"Probabilidad de que gane el equipo azul: {result['prob_blue']:.1%}")
    print(f"Probabilidad de que gane el equipo rojo: {result['prob_red']:.1%}")
    print(f"Jugadores sin historial: {result['jugadores_sin_historial']}")
    print("Posiciones inferidas (hechizos + campeón + partidas anteriores); no son las de Riot.")
    print(_weak_signal_warning(bundle))
    print("Guardada en data/predictions/live_log.csv." if result["logged"]
          else "Esta partida ya estaba en el registro: no se añadió otra fila.")


def _score_log_command() -> None:
    from src.riot_client.client import RiotUnavailableError
    bundle = _load_checked()
    try:
        r = score_log(bundle=bundle)
    except (requests.HTTPError, RiotUnavailableError) as error:
        raise SystemExit(f"No se pudo comprobar el registro: {_safe_error(error)}")
    print(f"Modelo evaluado: entrenado el {bundle.get('trained_at', '?')}; "
          f"filas de otros modelos (no evaluadas): {r['otro_modelo']}")
    print(f"Partidas registradas (de este modelo): {r['logged'] - r['otro_modelo']}; terminadas y evaluadas: {r['n']}; "
          f"sin terminar: {r['pendientes']}; descartadas (remake): {r['descartadas']}")
    if r["n"] == 0:
        print("Todavía no hay partidas terminadas que evaluar.")
        return
    print(f"n = {r['n']}, accuracy = {r['accuracy']:.1%}")
    if r["auc"] is None:
        print(f"Muy pocas partidas para calcular el ROC AUC (hacen falta al menos {MIN_FINISHED_FOR_AUC} y ambas clases).")
    else:
        print(f"ROC AUC = {r['auc']:.3f}")


def _freeze_command() -> None:
    bundle = _guarded(freeze)
    low, high = bundle["auc_ic95"]
    print(f"Congelado el {bundle['trained_at']}.")
    print(f"Modelo base conservado en {MODELS_DIR / BASE_NAME}; candidato en {MODELS_DIR / CANDIDATE_NAME} "
          f"(y copiado a {MODEL_PATH.name} hasta el reentrenamiento final).")
    print(f"Ajuste: {bundle['n_matches']} partidas ({len(bundle['features'])} variables, {bundle['best_params']}); "
          f"desarrollo: {len(bundle['development_ids'])} partidas; pool: {len(bundle['pool'])} partidas previas; "
          f"campeones con tabla de posiciones: {len(bundle['champion_positions'])}.")
    print(f"ROC AUC en validación temporal (optimista): {bundle['auc_temporal']:.3f} (IC 95 %: {low:.3f} a {high:.3f})")
    print(f"Lista de desarrollo en {MODELS_DIR / DEVELOPMENT_NAME}.")


def _guarded(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except PredictError as error:
        raise SystemExit(str(error))


def main():
    parser = argparse.ArgumentParser(description="Freeze, test and train the 10-variable model; predict a played "
                                                 "or live match.")
    parser.add_argument("match_id", nargs="?", help="e.g. LA1_123456")
    parser.add_argument("--freeze", action="store_true",
                        help="once: keep the 22-variable model as base and fit the candidate on its 920 matches")
    parser.add_argument("--evaluate-new", action="store_true",
                        help="test the frozen candidate against the base on matches outside the development set")
    parser.add_argument("--evaluate-final", action="store_true", help="segunda prueba del modelo final sin reentrenar")
    parser.add_argument("--train", action="store_true",
                        help="refit on every usable match and save modelo_final (after the test)")
    parser.add_argument("--force", action="store_true", help="with --train: skip the test-result check (contaminates the test)")
    parser.add_argument("--live", metavar="NAME#TAG", help="predict the game this player is in now")
    parser.add_argument("--score-log", action="store_true", help="compare logged live predictions with results")
    args = parser.parse_args()
    if args.evaluate_final and any((args.freeze, args.train, args.live, args.score_log, args.evaluate_new, args.match_id)):
        parser.error("--evaluate-final debe ejecutarse solo; no puede combinarse con entrenamiento ni otros comandos")
    if args.freeze:
        _freeze_command()
    elif args.train:
        _guarded(train_command, force=args.force)
    elif args.live:
        _live_command(args.live)
    elif args.score_log:
        _score_log_command()
    elif args.evaluate_final:
        _guarded(evaluate_final_command)
    elif args.evaluate_new:
        _guarded(evaluate_new_command)
    elif args.match_id:
        _predict_command(args.match_id)
    else:
        parser.error("give a match id, --freeze, --evaluate-new, --evaluate-final, --train, --live or --score-log")


if __name__ == "__main__":
    main()
