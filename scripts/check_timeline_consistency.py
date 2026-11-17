"""Comprobación local de objetivos. No descarga ni modifica la caché."""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.timeline import event_counts
from src.riot_client import cache

OBJECTIVES = {"torres": "tower", "dragones": "dragon", "barones": "baron", "inhibidores": "inhibitor"}


def check(timeline_dir=None):
    result = {name: {"coinciden": 0, "total": 0} for name in OBJECTIVES}
    skipped = 0
    paths = sorted(Path(timeline_dir or cache.TIMELINE_CACHE_DIR).glob("*.json.gz"))
    for path in paths:
        match = cache.load_match(path.name.removesuffix(".json.gz"))
        if match is None:
            continue
        try:
            with gzip.open(path, "rt") as f:
                counts = event_counts(json.load(f))
        except (OSError, ValueError, KeyError):
            skipped += 1
            continue
        for name, objective in OBJECTIVES.items():
            result[name]["total"] += 1
            result[name]["coinciden"] += int(all(counts[t["teamId"]][name] == t["objectives"][objective]["kills"] for t in match["info"]["teams"]))
    return result, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/analysis/consistencia.md"))
    args = parser.parse_args()
    result, skipped = check()
    lines = ["# Consistencia de eventos", "", "Caché local disponible al ejecutar. Cada coincidencia exige que ambos equipos coincidan.", "", "| Variable | Coinciden | Total | Diferencias |", "|---|---:|---:|---:|"]
    excluded = []
    for name, row in result.items():
        n, total = row["coinciden"], row["total"]
        lines.append(f"| {name} | {n} | {total} | {total-n} |")
        if total and (total-n)/total > .02:
            excluded.append(name)
    lines += ["", f"Archivos no comprobables: {skipped}.", "", "BUILDING_KILL.teamId identifica al dueño destruido. Se acredita al equipo contrario. ELITE_MONSTER_KILL.killerTeamId identifica al equipo que mata.", "", "Exclusiones por superar el 2 %: " + (", ".join(excluded) or "ninguna") + ".", "No se entrenó ni evaluó ningún modelo con datos reales."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
