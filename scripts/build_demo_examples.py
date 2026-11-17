"""Genera ejemplos locales sin consultar Riot. Selección reproducible desde CSV.

El CSV contiene match_id, categoria y prob_previa (opcional). Las categorías
son acierto_con_confianza, fallo_con_confianza, remontada y azar. La corrida de
análisis las determina con predicciones fuera de muestra antes de llamar aquí.
"""
import argparse
import gzip
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.app.logic import example_from_match, replay_events
from src.features.build import player_features
from src.models import ingame
from src.riot_client import cache

QUOTAS = {"acierto_con_confianza": 4, "fallo_con_confianza": 2, "remontada": 3, "azar": 3}


def enrich_examples(examples, data):
    """Recupera eventos y medias previas de los ejemplos sin escribir en data."""
    data = Path(data)
    ids = {e['match_id_interno'] for e in examples}
    export = data / 'export'
    matches = pd.read_csv(export / 'matches.csv', dtype={'patch': str})
    matches = matches[matches.match_id.isin(ids)]
    participants = pd.read_csv(export / 'participants.csv')
    participants = participants[participants.match_id.isin(ids)]
    history = pd.read_csv(export / 'player_history.csv')
    history = history[history.target_match_id.isin(ids)]
    players = player_features(history, participants, matches)
    fields = ['winrate_10', 'kda_medio', 'partidas_con_campeon']
    means = players.groupby(['match_id', 'team_id'])[fields].mean()
    result = []
    for example in examples:
        key = example['match_id_interno']
        with gzip.open(data / 'raw/timelines' / f'{key}.json.gz', 'rt', encoding='utf-8') as stream:
            events = replay_events(json.load(stream))
        summary = {label: {field: None if pd.isna(value) else float(value)
                           for field, value in means.loc[(key, team)].items()}
                   for team, label in [(100, 'azul'), (200, 'rojo')]}
        result.append({**example, 'historial': summary, 'eventos': events})
    return result


def select_examples(candidates, seed=42):
    rng = random.Random(seed)
    chosen, used = [], set()
    for category, n in QUOTAS.items():
        eligible = sorted((r for r in candidates if r["categoria"] == category and r["match_id"] not in used), key=lambda r: r["match_id"])
        # Un mismo identificador no puede ocupar dos plazas.
        eligible = list({r["match_id"]: r for r in eligible}.values())
        if len(eligible) < n:
            raise ValueError(f"Faltan partidas para la categoría {category}.")
        picked = rng.sample(eligible, n)
        chosen.extend(picked)
        used.update(r["match_id"] for r in picked)
    return chosen


def build(candidates, bundle, seed=42):
    result = []
    for i, row in enumerate(select_examples(candidates, seed), 1):
        match, timeline = cache.load_match(row["match_id"]), cache.load_timeline(row["match_id"])
        if match is None or timeline is None:
            raise ValueError("Falta una partida o su línea de tiempo en la caché.")
        prior = row.get("prob_previa")
        prior = None if pd.isna(prior) else float(prior)
        result.append(example_from_match(match, timeline, bundle, prior, i))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--model", default=ingame.MODEL_PATH, type=Path)
    parser.add_argument("--output", default=ingame.ROOT / "data/app/ejemplos.json", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--data", type=Path, help="Añade medias por equipo desde los productos locales.")
    args = parser.parse_args(argv)
    examples = build(pd.read_csv(args.candidates).to_dict("records"), ingame.load_bundle(args.model), args.seed)
    if args.data is not None:
        examples = enrich_examples(examples, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(examples, ensure_ascii=False))
    print(f"Guardamos {len(examples)} ejemplos sin datos de jugadores.")


if __name__ == "__main__":
    main()
