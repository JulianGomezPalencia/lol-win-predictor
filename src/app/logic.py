"""Operaciones de pantalla con salidas limitadas a datos sin jugadores."""
import json
import re
from pathlib import Path

from src.features.timeline import STATE_FEATURES, participant_teams
from src.models import ingame, predict
from src.models.explain import contributions

ROOT = Path(__file__).resolve().parents[2]
def asset_path(local, demo):
    """Prefiere el producto local y recurre a la copia pública si falta."""
    path = ROOT / "data" / local
    return path if path.is_file() or path.is_dir() else ROOT / "demo" / demo


def load_prior_model():
    import joblib
    return joblib.load(asset_path("models/modelo_final.joblib", "modelo_previo.joblib"))


def prior_calculator(features, bundle=None):
    import pandas as pd
    bundle = load_prior_model() if bundle is None else bundle
    row = pd.DataFrame([{name: features.get(name, 0.) for name in bundle["features"]}])
    probability = bundle["model"].predict_proba(row)[0]
    classes = list(bundle["model"].classes_)
    return {"prob_azul": float(probability[classes.index(1)]),
            "factores": contributions(bundle, row.iloc[0].to_dict())}


def load_ingame_model():
    return ingame.load_bundle(asset_path("models/modelos_en_partida.joblib", "modelos_en_partida.joblib"))
POSITIONS = {"TOP": "Superior", "JUNGLE": "Jungla", "MIDDLE": "Central", "BOTTOM": "Inferior", "UTILITY": "Apoyo"}


def champion_names(path=None):
    path = asset_path("static/champion_16.22.1.json", "champion_16.22.1.json") if path is None else path
    try:
        data = json.loads(Path(path).read_text())["data"]
        return {int(c["key"]): c["name"] for c in data.values()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def public_example(example, index=1, names=None):
    """Lista blanca. Etiquetas y nombres se reconstruyen, nunca se copian del JSON."""
    names = set(champion_names().values()) if names is None else set(names)
    champions = example.get("campeones", {})
    return {"etiqueta": f"Partida {index}", "gana_azul": bool(example["gana_azul"]),
            "prob_previa": None if example.get("prob_previa") is None else float(example["prob_previa"]),
            "campeones": {team: [c if c in names else "Campeón" for c in champions.get(team, [])] for team in ("azul", "rojo")},
            "curva": [{"minuto": int(p["minuto"]), "prob_azul": float(p["prob_azul"])} for p in example["curva"]],
            "eventos": [{"minuto": float(e["minuto"]), "tipo": e["tipo"], "equipo": e["equipo"]}
                        for e in example.get("eventos", []) if e.get("tipo") in ("dragón", "torre", "barón") and e.get("equipo") in ("azul", "rojo")]}


def load_examples(path=None):
    path = asset_path("app/ejemplos.json", "ejemplos.json") if path is None else path
    try:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, list):
            return []
        return [public_example(e, i) for i, e in enumerate(raw, 1)]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def calculator(bundle, minute, gold=0, kills=0, towers=0, dragons=0):
    state = dict.fromkeys(STATE_FEATURES, 0)
    state.update(dif_oro=float(gold), dif_asesinatos=float(kills), dif_torres=float(towers), dif_dragones=float(dragons))
    return ingame.probability(bundle, minute, state)


def replay_events(timeline):
    events = []
    for frame in timeline["info"]["frames"]:
        for e in frame.get("events", []):
            kind, team = None, None
            if e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
                kind, team = "torre", 300 - e["teamId"]
            elif e["type"] == "ELITE_MONSTER_KILL":
                kind, team = {"DRAGON": "dragón", "BARON_NASHOR": "barón"}.get(e.get("monsterType")), e.get("killerTeamId")
            if kind and team in (100, 200):
                events.append({"minuto": e["timestamp"] / 60_000, "tipo": kind, "equipo": "azul" if team == 100 else "rojo"})
    return events


def example_from_match(match, timeline, bundle, pre_prob=None, index=1):
    # Join teams explicitly so state_at also works without a match file on disk.
    teams = {p["participantId"]: p["teamId"] for p in match["info"]["participants"]}
    timeline = {**timeline, "info": {**timeline["info"], "participants": [
        {**p, "teamId": teams[p["participantId"]]} for p in timeline["info"]["participants"]]}}
    participant_teams(timeline)
    names = champion_names()
    example = {"etiqueta": f"Partida {index}", "match_id_interno": match["metadata"]["matchId"],
               "gana_azul": next(t["win"] for t in match["info"]["teams"] if t["teamId"] == 100),
               "prob_previa": pre_prob,
               "campeones": {label: [names.get(p["championId"], "Campeón") for p in match["info"]["participants"] if p["teamId"] == team] for team, label in [(100, "azul"), (200, "rojo")]},
               "curva": ingame.curve(bundle, timeline, pre_prob), "eventos": replay_events(timeline)}
    return example


def replay(match_id, bundle, endpoints=None):
    if not re.fullmatch(r"[A-Za-z0-9]+_[0-9]+", match_id.strip()):
        raise ValueError("Escribe un identificador de partida válido.")
    if endpoints is None:
        from src.riot_client import endpoints
    # Exactly two cache-aware endpoint operations, no history requests.
    match = endpoints.get_match(match_id.strip())
    timeline = endpoints.get_match_timeline(match_id.strip())
    return public_example(example_from_match(match, timeline, bundle))


def live(riot_id, bundle, progress=lambda _: None):
    names = champion_names()
    result = predict.predict_live(riot_id, bundle, progress=progress, names=names, persist=False)
    return {"prob_azul": result["prob_blue"], "confianza": predict._confidence_line(result["prob_blue"], bundle),
            "equipos": {label: [(POSITIONS.get(pos, "Posición por estimar"), champion if champion in names.values() else "Campeón")
                        for pos, champion in result["teams"].get(team, [])] for team, label in [(100, "azul"), (200, "rojo")]},
            "factores": [{"aporte": c["aporte"], "frase": c["frase"]} for c in contributions(bundle, result["feature_values"])[:3]]}


def findings_paths(analysis=None, figures=None):
    paths = [Path(analysis) if analysis is not None else asset_path("analysis", "analysis"),
             Path(figures) if figures is not None else asset_path("analysis/figuras", "figuras")]
    return {"tablas": sorted(paths[0].glob("*.csv")), "figuras": sorted(set(p for directory in [*paths, paths[0] / "figuras"] for p in directory.glob("*.png")))}


def read_findings_table(path):
    import pandas as pd
    table = pd.read_csv(path)
    private_columns = ("puuid", "riot", "summoner", "player", "jugador", "nombre", "match_id", "game_id")
    table = table[[c for c in table if c.lower() != "fuente" and not any(token in c.lower() for token in private_columns)]]
    for column in table.select_dtypes(include=["object", "string"]):
        table[column] = table[column].map(lambda value: "Dato omitido" if isinstance(value, str) and ("#" in value or len(value) > 64) else value)
    return table
