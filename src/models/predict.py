"""
Final model and prediction for ONE already-played match.

  python -m src.models.predict --train          # fit on data/export, save data/models/modelo_final.joblib
  python -m src.models.predict LA1_123456       # probability that blue wins that match

The saved model is the logistic regression WITHOUT the position variables (a more cautious version, not leak-free:
Riot assigns teamPosition after the game, see train.position_sensitivity; the lane-against-lane variables
still use it to pair opponents), fitted on all matches
with tier weights. Its walk-forward ROC AUC is stored in the bundle and shown with every prediction.

Scope limit: only matches that were already played. The match-v5 data gives each player's position;
Riot's spectator endpoint (live games) does not, so a true live prediction would need role
inference. That is NOT built.

Never touches data/db/lol.sqlite (db.connect would migrate it) and never writes to data/export.
Tests:    pytest tests/test_predict.py -v
"""
import argparse
import warnings
from pathlib import Path

import joblib
import requests
import pandas as pd
import sklearn

from src.features import build
from src.models import train

# src.agents / src.riot_client are imported only where a match is downloaded: their config needs
# RIOT_API_KEY, and --train makes no API calls (tests/test_predict.py checks this without a key).

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "modelo_final.joblib"
LOGISTIC = "Regresión logística"

HISTORY_COLUMNS = ["target_match_id", "player_id", "history_rank", "history_match_id", "game_start_ts",
                   "game_duration", "remake", "win", "champion_id", "champion_name", "team_position",
                   "kills", "deaths", "assists"]


def fit_final_model(table: pd.DataFrame) -> dict:
    """Fit the logistic regression on ALL rows, without position variables, and describe it."""
    reduced = table.drop(columns=train.position_features(build.feature_columns(table)))
    features = build.feature_columns(reduced)
    search = train.logistic_model()
    search.fit(reduced[features], reduced["blue_win"],
               model__sample_weight=train.tier_weights(reduced["seed_tier"]))
    row = train.temporal_validation(reduced)["tabla"].loc[LOGISTIC]
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


def features_for_match(match: dict, histories: dict[str, list[dict]]) -> pd.DataFrame:
    """One-row feature table for a match-v5 dict. Pure: no API, no files.

    `histories` maps puuid -> rows from history_builder.player_history (missing or empty = no history).
    The feature logic is build.player_features / build.match_features, not reimplemented here.
    """
    from src.agents import match_fetcher
    match_id = match["metadata"]["matchId"]
    summary = match_fetcher.summarize(match)
    matches = pd.DataFrame([{"match_id": match_id, "status": "history_done", "seed_tier": "UNKNOWN", **summary}])
    participants = pd.DataFrame(
        [{"match_id": match_id, "player_id": p["puuid"], **{k: v for k, v in p.items() if k != "puuid"}}
         for p in match_fetcher.extract_participants(match)])
    rows = [{"target_match_id": match_id, "player_id": puuid, **game}
            for puuid, games in histories.items() for game in games]
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    return build.match_features(build.player_features(history, participants, matches), matches)


def in_training_data(match_id: str, start_ts: int, bundle: dict) -> bool:
    """True if the model was fitted on this match (or, for an old bundle, may have been)."""
    if "match_ids" in bundle:
        return match_id in bundle["match_ids"]
    return start_ts <= bundle["last_match_ts"]


def predict_match(match_id: str, bundle: dict) -> dict:
    """Probability that blue wins an already-played match (downloads it and the players' histories)."""
    from src.agents import history_builder, match_fetcher
    from src.riot_client import endpoints
    match = endpoints.get_match(match_id)
    reason = match_fetcher.invalid_reason(match)
    if reason:
        raise ValueError(f"{match_id} no se puede usar: {reason}")
    start = match["info"]["gameStartTimestamp"]
    histories = {p["puuid"]: history_builder.player_history(p["puuid"], start)
                 for p in match["info"]["participants"]}
    table = features_for_match(match, histories)
    missing = [f for f in bundle["features"] if f not in table.columns]
    if missing:
        raise ValueError(f"El modelo guardado usa variables que ya no existen: {missing}. Vuelve a entrenar con --train.")
    row = table[bundle["features"]]
    prob_blue = float(bundle["model"].predict_proba(row)[0, 1])
    summary = match_fetcher.summarize(match)
    return {"match_id": match_id, "prob_blue": prob_blue, "prob_red": 1 - prob_blue,
            "blue_win": summary["blue_win"],
            "jugadores_sin_historial": sum(1 for g in histories.values() if all(game["remake"] for game in g)),
            "en_entrenamiento": in_training_data(match_id, start, bundle)}


def _train_command() -> None:
    bundle = fit_final_model(build.build())
    path = save_model(bundle)
    low, high = bundle["auc_ic95"]
    print(f"Modelo guardado en {path}")
    print(f"{bundle['n_matches']} partidas, {len(bundle['features'])} variables, {bundle['best_params']}")
    print(f"ROC AUC en validación temporal: {bundle['auc_temporal']:.3f} (IC 95 %: {low:.3f} a {high:.3f})")


def _predict_command(match_id: str) -> None:
    from src.riot_client.client import RiotUnavailableError
    try:
        bundle = load_model()
    except FileNotFoundError as error:
        raise SystemExit(str(error))
    print("Descargando la partida y el historial de los 10 jugadores (puede tardar un par de minutos)...")
    try:
        result = predict_match(match_id, bundle)
    except (ValueError, requests.HTTPError, RiotUnavailableError) as error:
        raise SystemExit(f"No se pudo predecir {match_id}: {error}")
    low, high = bundle["auc_ic95"]
    real = "azul" if result["blue_win"] else "rojo"
    if result["en_entrenamiento"]:
        print("ADVERTENCIA: el modelo se ajustó con esta partida. La probabilidad es dentro de la muestra "
              "y NO es una predicción real: el ROC AUC de abajo no la describe.")
    print(f"Probabilidad de que gane el equipo azul: {result['prob_blue']:.1%}")
    print(f"Resultado real: ganó el equipo {real}")
    print(f"Jugadores sin historial: {result['jugadores_sin_historial']}")
    print("Aviso: es una señal débil. En validación temporal el modelo tiene ROC AUC "
          f"{bundle['auc_temporal']:.3f} (IC 95 %: {low:.3f} a {high:.3f}); 0.5 equivale a adivinar.")


def main():
    parser = argparse.ArgumentParser(description="Train the final model or predict one played match.")
    parser.add_argument("match_id", nargs="?", help="e.g. LA1_123456")
    parser.add_argument("--train", action="store_true", help="fit on data/export and save the model")
    args = parser.parse_args()
    if args.train:
        _train_command()
    elif args.match_id:
        _predict_command(args.match_id)
    else:
        parser.error("give a match id or --train")


if __name__ == "__main__":
    main()
