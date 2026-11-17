"""Una regresión logística fija por minuto. La CLI solo lee datos locales."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.features.timeline import STATE_FEATURES, state_at
from src.riot_client import cache

KEY_MINUTES = [5, 10, 15, 20, 25]
CURVE_MINUTES = range(1, 31)
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = "data/models/modelos_en_partida.joblib"


def train(rows):
    table = pd.DataFrame(rows)
    models, counts, skipped = {}, {}, []
    for minute in CURVE_MINUTES:
        group = table[table["minute"] == minute] if not table.empty else table
        counts[minute] = len(group)
        if group.empty or group["blue_win"].nunique() < 2:
            skipped.append(minute)
            continue
        if group["match_id"].duplicated().any():
            raise ValueError("Hay partidas repetidas dentro de un minuto.")
        model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
        model.fit(group[STATE_FEATURES], group["blue_win"].astype(int))
        models[minute] = model
    if not models:
        raise ValueError("No hay minutos con partidas de ambos resultados para entrenar.")
    return {"version": 1, "features": list(STATE_FEATURES), "models": models, "counts": counts,
            "unavailable_minutes": skipped, "match_ids": sorted(table["match_id"].unique().tolist()),
            "trained_at": datetime.now(timezone.utc).isoformat(), "recipe": "StandardScaler + LogisticRegression(C=1.0)"}


def load_bundle(path=MODEL_PATH):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    bundle = joblib.load(path)
    if bundle.get("features") != STATE_FEATURES or bundle.get("version") != 1:
        raise ValueError("El modelo en partida no corresponde a estas variables.")
    return bundle


def probability(bundle, minute, state):
    if int(minute) != minute or minute < 1:
        raise ValueError("El minuto debe ser un entero positivo.")
    minute = min(int(minute), 30)
    model = bundle["models"].get(minute)
    if model is None:
        raise ValueError(f"No hay modelo entrenado para el minuto {minute}.")
    row = pd.DataFrame([{name: state[name] for name in bundle["features"]}])
    return float(model.predict_proba(row)[0, list(model.classes_).index(1)])


def curve(bundle, timeline, pre_prob=None):
    points = [] if pre_prob is None else [{"minuto": 0, "prob_azul": float(pre_prob)}]
    for minute in CURVE_MINUTES:
        state = state_at(timeline, minute)
        if state is None:
            break
        points.append({"minuto": minute, "prob_azul": probability(bundle, minute, state)})
    return points


def _iso_ms(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("La fecha debe incluir zona horaria.")
    return date.timestamp() * 1000


def cached_rows(export_dir=None, db_path=None, before=None, after=None):
    """El corte es estricto por game_start_ts en milisegundos, nunca por descarga."""
    export = Path(export_dir or ROOT / "data/export") / "matches.csv"
    if export.exists():
        matches = pd.read_csv(export)
    else:
        path = Path(db_path or ROOT / "data/db/lol.sqlite").resolve()
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
            matches = pd.read_sql_query("SELECT match_id, blue_win, game_start_ts FROM target_matches WHERE status='history_done'", conn)
    if "status" in matches:
        matches = matches[matches["status"] == "history_done"]
    matches = matches.dropna(subset=["blue_win", "game_start_ts"])
    if before:
        matches = matches[matches["game_start_ts"] < _iso_ms(before)]
    if after:
        matches = matches[matches["game_start_ts"] > _iso_ms(after)]
    rows = []
    for record in matches.to_dict("records"):
        timeline = cache.load_timeline(record["match_id"])
        if timeline is None:
            continue
        for minute in CURVE_MINUTES:
            state = state_at(timeline, minute)
            if state is None:
                break
            rows.append({"minute": minute, "match_id": record["match_id"], "blue_win": int(record["blue_win"]), **state})
    return rows


def evaluate(bundle, rows):
    table = pd.DataFrame(rows)
    if table.empty:
        return []
    if set(table["match_id"]) & set(bundle.get("match_ids", [])):
        raise ValueError("La evaluación contiene partidas usadas para entrenar.")
    result = []
    for minute in KEY_MINUTES:
        group = table[table["minute"] == minute]
        if group.empty:
            continue
        y = group["blue_win"].astype(int)
        p = [probability(bundle, minute, r) for r in group.to_dict("records")]
        result.append({"minuto": minute, "n": len(group), "acierto": float(accuracy_score(y, [v >= .5 for v in p])),
                       "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
                       "perdida_logaritmica": float(log_loss(y, p, labels=[0, 1]))})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / MODEL_PATH)
    args = parser.parse_args(argv)
    rows = cached_rows(args.export_dir, args.db_path, args.before, args.after)
    if args.train:
        bundle = train(rows)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.model)
        print(f"Modelos guardados: {len(bundle['models'])}. Minutos sin datos suficientes: {bundle['unavailable_minutes']}.")
    else:
        print(json.dumps(evaluate(load_bundle(args.model), rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
