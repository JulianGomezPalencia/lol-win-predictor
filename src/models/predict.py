"""
Final model and prediction: played matches, live games, and a forward test of the frozen model.

  python -m src.models.predict --train               # fit on data/export, save data/models/modelo_final.joblib
  python -m src.models.predict LA1_123456            # probability that blue wins that (played) match
  python -m src.models.predict --live "Name#TAG"     # a game in progress (spectator-v5), BEFORE it ends
  python -m src.models.predict --score-log           # compare the logged live predictions with the real results
  python -m src.models.predict --evaluate-new        # frozen model vs matches collected AFTER it was trained

Roles. Riot assigns teamPosition after the game and the live endpoint gives no position, so the model
uses INFERRED roles everywhere (src/features/roles.py: summoner spells + earlier games). Training,
played-match prediction and live prediction share the same path (score_game), so they always agree.
The saved model is the logistic regression without the position variables, fitted on all matches with
tier weights; its walk-forward ROC AUC is stored in the bundle and shown with every prediction.
Bundles saved before this change (no "roles": "inferred") are refused: retrain with --train.

Live limits: the game must be ranked solo/duo (queue 420) and in progress; a hidden or firewall-blocked
player has no history; collecting data (run_pipeline) and --live must not run at the same time, because
each process has its own rate limiter.

Never touches data/db/lol.sqlite (db.connect would migrate it) and never writes to data/export.
Tests:    pytest tests/test_predict.py -v
"""
import argparse
import csv
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

from src.features import build
from src.models import train

# src.agents / src.riot_client are imported only where a match is downloaded: their config needs
# RIOT_API_KEY, and --train makes no API calls (tests/test_predict.py checks this without a key).

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "modelo_final.joblib"
LOGISTIC = "Regresión logística"
LOG_PATH = PROJECT_ROOT / "data" / "predictions" / "live_log.csv"
LOG_COLUMNS = ["game_id", "platform", "timestamp", "prob_blue", "model_trained_at"]
RANKED_SOLO_QUEUE = 420
MIN_FINISHED_FOR_AUC = 10
RETRAIN_MESSAGE = ("El modelo guardado usa posiciones de Riot (posteriores a la partida) o es de una versión "
                   "anterior. Vuelve a entrenar con: python -m src.models.predict --train")

HISTORY_COLUMNS = ["target_match_id", "player_id", "history_rank", "history_match_id", "game_start_ts",
                   "game_duration", "remake", "win", "champion_id", "champion_name", "team_position",
                   "kills", "deaths", "assists"]


def fit_final_model(table: pd.DataFrame) -> dict:
    """Fit the logistic regression on ALL rows, without position variables, and describe it.

    `table` should come from build.build(inferred_positions=True): the bundle says so ("roles").
    """
    reduced = table.drop(columns=train.position_features(build.feature_columns(table)))
    features = build.feature_columns(reduced)
    search = train.logistic_model()
    search.fit(reduced[features], reduced["blue_win"],
               model__sample_weight=train.tier_weights(reduced["seed_tier"]))
    row = train.temporal_validation(reduced)["tabla"].loc[LOGISTIC]
    blue_rate = float(reduced["blue_win"].mean())
    return {
        "model": search.best_estimator_,
        "features": features,
        "best_params": search.best_params_,
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
        raise FileNotFoundError(f"No hay modelo guardado en {path}. Ejecuta primero: python -m src.models.predict --train")
    bundle = joblib.load(path)
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(f"El modelo se guardó con scikit-learn {bundle.get('sklearn_version')} y ahora hay "
                      f"{sklearn.__version__}; si falla o parece raro, vuelve a entrenar con --train.")
    return bundle


def check_bundle(bundle: dict) -> None:
    """Refuse a bundle trained with Riot's post-game positions (it would not match inferred roles)."""
    if bundle.get("roles") != "inferred":
        raise ValueError(RETRAIN_MESSAGE)


# --- One path for every prediction: players + their histories -> probability ----------------


def players_from_match(match: dict) -> list[dict]:
    """Players of a match-v5 dict. Riot's teamPosition is NOT read."""
    return [{"key": p["puuid"], "team_id": p["teamId"], "champion_id": p["championId"],
             "champion_name": p["championName"], "spell_ids": (p["summoner1Id"], p["summoner2Id"])}
            for p in match["info"]["participants"]]


def features_for_game(game_id: str, start_ts: int, players: list[dict], histories: dict[str, list[dict]],
                      blue_win: int = 0, patch: str = "") -> tuple[pd.DataFrame, dict]:
    """One-row feature table and the inferred role of each player. Pure: no API, no files.

    `players`: dicts with key, team_id, champion_id, champion_name, spell_ids.
    `histories` maps key -> rows from history_builder.player_history (missing or empty = no history).
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
    participants = build.inferred_participants(participants, history, matches, spells)
    table = build.match_features(build.player_features(history, participants, matches), matches)
    return table, dict(zip(participants["player_id"], participants["team_position"]))


def features_for_match(match: dict, histories: dict[str, list[dict]]) -> pd.DataFrame:
    """One-row feature table for a match-v5 dict (inferred roles). Pure: no API, no files."""
    from src.agents import match_fetcher
    summary = match_fetcher.summarize(match)
    table, _ = features_for_game(match["metadata"]["matchId"], summary["game_start_ts"],
                                 players_from_match(match), histories, summary["blue_win"], summary["patch"])
    return table


def score_game(game_id: str, start_ts: int, players: list[dict], histories: dict[str, list[dict]],
               bundle: dict, blue_win: int = 0) -> dict:
    """Probability that blue wins, from players and histories. Shared by played and live prediction."""
    check_bundle(bundle)
    table, roles = features_for_game(game_id, start_ts, players, histories, blue_win)
    missing = [f for f in bundle["features"] if f not in table.columns]
    if missing:
        raise ValueError(f"El modelo guardado usa variables que ya no existen: {missing}. Vuelve a entrenar con --train.")
    prob_blue = float(bundle["model"].predict_proba(table[bundle["features"]])[0, 1])
    return {"prob_blue": prob_blue, "prob_red": 1 - prob_blue, "roles": roles,
            "jugadores_sin_historial": sum(1 for g in histories.values() if all(game["remake"] for game in g))}


def in_training_data(match_id: str, start_ts: int, bundle: dict) -> bool:
    """True if the model was fitted on this match (or, for an old bundle, may have been)."""
    if "match_ids" in bundle:
        return match_id in bundle["match_ids"]
    return start_ts <= bundle["last_match_ts"]


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
            "en_entrenamiento": in_training_data(match_id, start, bundle)}


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


def score_log(log_path: Path | None = None, get_match=None) -> dict:
    """Compare the logged live predictions with the real results (finished games only)."""
    path = Path(log_path or LOG_PATH)
    if not path.exists():
        return {"n": 0, "pendientes": 0, "descartadas": 0, "accuracy": None, "auc": None, "logged": 0}
    if get_match is None:
        from src.riot_client import endpoints
        get_match = endpoints.get_match
    from src.agents import match_fetcher
    from src.riot_client import endpoints
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
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
            "accuracy": float(accuracy_score(y, p >= 0.5)) if n else None,
            "auc": float(roc_auc_score(y, p)) if both else None}


def evaluate_new(bundle: dict, table: pd.DataFrame) -> dict:
    """Frozen-model test: score only matches NOT in bundle["match_ids"]. The model is never refitted."""
    check_bundle(bundle)
    new = table[~table["match_id"].astype(str).isin(set(bundle["match_ids"]))]
    new = new.sort_values("game_start_ts", kind="stable")
    if new.empty:
        return {"n": 0}
    y = new["blue_win"].to_numpy()
    proba = np.clip(bundle["model"].predict_proba(new[bundle["features"]])[:, 1], 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)
    two_classes = len(np.unique(y)) == 2
    constant = np.full(len(y), np.clip(bundle["blue_win_rate"], 1e-6, 1 - 1e-6))
    day = lambda ts: datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d")
    return {
        "n": int(len(new)), "desde": day(new["game_start_ts"].min()), "hasta": day(new["game_start_ts"].max()),
        "accuracy": float(accuracy_score(y, pred)), "accuracy_ic95": train.bootstrap_accuracy_ci(y, pred),
        "auc": float(roc_auc_score(y, proba)) if two_classes else None,
        "auc_ic95": train.bootstrap_auc_ci(y, proba) if two_classes else None,
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "base_accuracy": float(np.mean(y == bundle["majority_blue_win"])),
        "base_log_loss": float(log_loss(y, constant, labels=[0, 1])),
    }


# --- Commands -----------------------------------------------------------------------------


def _weak_signal_warning(bundle: dict) -> str:
    low, high = bundle["auc_ic95"]
    return ("Aviso: es una señal débil. En validación temporal el modelo tiene ROC AUC "
            f"{bundle['auc_temporal']:.3f} (IC 95 %: {low:.3f} a {high:.3f}); 0.5 equivale a adivinar.")


def _load_checked() -> dict:
    try:
        bundle = load_model()
        check_bundle(bundle)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
    return bundle


def _train_command() -> None:
    try:
        table = build.build(inferred_positions=True)
    except FileNotFoundError as error:
        raise SystemExit(str(error))
    bundle = fit_final_model(table)
    path = save_model(bundle)
    low, high = bundle["auc_ic95"]
    print(f"Modelo guardado en {path}")
    print(f"{bundle['n_matches']} partidas, {len(bundle['features'])} variables, {bundle['best_params']}, posiciones inferidas")
    print(f"ROC AUC en validación temporal: {bundle['auc_temporal']:.3f} (IC 95 %: {low:.3f} a {high:.3f})")


def _predict_command(match_id: str) -> None:
    from src.riot_client.client import RiotUnavailableError
    bundle = _load_checked()
    print("Descargando la partida y el historial de los 10 jugadores (puede tardar un par de minutos)...")
    try:
        result = predict_match(match_id, bundle)
    except (ValueError, requests.HTTPError, RiotUnavailableError) as error:
        raise SystemExit(f"No se pudo predecir {match_id}: {error}")
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
        raise SystemExit(f"No se pudo predecir la partida en vivo: {error}")
    for team_id, label in ((100, "Equipo azul"), (200, "Equipo rojo")):
        print(f"{label}:")
        for position, champion in result["teams"].get(team_id, []):
            print(f"  {position:<8} {champion}")
    print(f"Probabilidad de que gane el equipo azul: {result['prob_blue']:.1%}")
    print(f"Probabilidad de que gane el equipo rojo: {result['prob_red']:.1%}")
    print(f"Jugadores sin historial: {result['jugadores_sin_historial']}")
    print("Posiciones inferidas (hechizos + partidas anteriores); no son las de Riot.")
    print(_weak_signal_warning(bundle))
    print("Guardada en data/predictions/live_log.csv." if result["logged"]
          else "Esta partida ya estaba en el registro: no se añadió otra fila.")


def _score_log_command() -> None:
    from src.riot_client.client import RiotUnavailableError
    try:
        r = score_log()
    except (requests.HTTPError, RiotUnavailableError) as error:
        raise SystemExit(f"No se pudo comprobar el registro: {error}")
    print(f"Partidas registradas: {r['logged']}; terminadas y evaluadas: {r['n']}; "
          f"sin terminar: {r['pendientes']}; descartadas (remake): {r['descartadas']}")
    if r["n"] == 0:
        print("Todavía no hay partidas terminadas que evaluar.")
        return
    print(f"n = {r['n']}, accuracy = {r['accuracy']:.1%}")
    if r["auc"] is None:
        print(f"Muy pocas partidas para calcular el ROC AUC (hacen falta al menos {MIN_FINISHED_FOR_AUC} y ambas clases).")
    else:
        print(f"ROC AUC = {r['auc']:.3f}")


def _evaluate_new_command() -> None:
    bundle = _load_checked()
    try:
        table = build.build(inferred_positions=True)
    except FileNotFoundError as error:
        raise SystemExit(str(error))
    r = evaluate_new(bundle, table)
    if r["n"] == 0:
        print("0 partidas nuevas: todas las partidas exportadas ya estaban en el entrenamiento. "
              "Recolecta más datos y vuelve a exportar (ver README).")
        return
    a_low, a_high = r["accuracy_ic95"]
    print(f"Modelo congelado (entrenado el {bundle.get('trained_at', '?')}), sin reajustar.")
    print(f"n = {r['n']} partidas nuevas, del {r['desde']} al {r['hasta']}")
    print(f"Accuracy: {r['accuracy']:.1%} (IC 95 %: {a_low:.1%} a {a_high:.1%})")
    if r["auc"] is None:
        print("ROC AUC: no se puede calcular (una sola clase).")
    else:
        low, high = r["auc_ic95"]
        print(f"ROC AUC: {r['auc']:.3f} (IC 95 %: {low:.3f} a {high:.3f})")
    print(f"Log loss: {r['log_loss']:.3f}")
    print(f"Siempre la clase mayoritaria del entrenamiento: accuracy {r['base_accuracy']:.1%}, "
          f"log loss {r['base_log_loss']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train the final model, predict a played or live match, "
                                                 "or test the frozen model.")
    parser.add_argument("match_id", nargs="?", help="e.g. LA1_123456")
    parser.add_argument("--train", action="store_true", help="fit on data/export and save the model")
    parser.add_argument("--live", metavar="NAME#TAG", help="predict the game this player is in now")
    parser.add_argument("--score-log", action="store_true", help="compare logged live predictions with results")
    parser.add_argument("--evaluate-new", action="store_true",
                        help="score the saved model on matches it was not trained on")
    args = parser.parse_args()
    if args.train:
        _train_command()
    elif args.live:
        _live_command(args.live)
    elif args.score_log:
        _score_log_command()
    elif args.evaluate_new:
        _evaluate_new_command()
    elif args.match_id:
        _predict_command(args.match_id)
    else:
        parser.error("give a match id, --live, --score-log, --evaluate-new or --train")


if __name__ == "__main__":
    main()
