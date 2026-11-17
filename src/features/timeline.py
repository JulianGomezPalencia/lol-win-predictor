"""Estado por minuto, siempre como azul menos rojo. Sin acceso a red."""
from src.riot_client import cache

STATE_FEATURES = ["dif_oro", "dif_experiencia", "dif_niveles", "dif_subditos",
                  "dif_asesinatos", "dif_torres", "dif_placas", "dif_dragones",
                  "dif_larvas", "dif_heraldos", "dif_barones", "dif_inhibidores"]
MONSTERS = {"DRAGON": "dragones", "HORDE": "larvas", "RIFTHERALD": "heraldos", "BARON_NASHOR": "barones"}


def participant_teams(timeline):
    participants = timeline.get("info", {}).get("participants", [])
    teams = {int(p["participantId"]): p["teamId"] for p in participants if p.get("teamId") in (100, 200)}
    if len(teams) != 10:
        match_id = timeline.get("metadata", {}).get("matchId")
        match = cache.load_match(match_id) if match_id else None
        if match:
            raw = match["info"]["participants"]
            by_id = {p["participantId"]: p["teamId"] for p in raw}
            by_player = {p.get("puuid"): p["teamId"] for p in raw if p.get("puuid")}
            for p in participants:
                team = by_player.get(p.get("puuid"), by_id.get(p["participantId"]))
                if team in (100, 200):
                    teams[int(p["participantId"])] = team
    if len(teams) != 10 or list(teams.values()).count(100) != 5 or list(teams.values()).count(200) != 5:
        raise ValueError("No pudimos comprobar los equipos de los diez participantes.")
    return teams


def event_counts(timeline, cutoff=float("inf")):
    teams = participant_teams(timeline)
    counts = {team: dict.fromkeys(["asesinatos", "torres", "placas", "dragones", "larvas", "heraldos", "barones", "inhibidores"], 0) for team in (100, 200)}
    for frame in timeline["info"]["frames"]:
        for e in frame.get("events", []):
            if e.get("timestamp", 0) > cutoff:
                continue
            kind, team, variable = e["type"], None, None
            if kind == "CHAMPION_KILL":
                team, variable = teams.get(e.get("killerId")), "asesinatos"
            elif kind == "ELITE_MONSTER_KILL":
                team, variable = e.get("killerTeamId"), MONSTERS.get(e.get("monsterType"))
            elif kind in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
                # teamId owns the destroyed building/plate, including kills by minions.
                team = 300 - e["teamId"] if e.get("teamId") in (100, 200) else None
                variable = "placas" if kind == "TURRET_PLATE_DESTROYED" else {"TOWER_BUILDING": "torres", "INHIBITOR_BUILDING": "inhibidores"}.get(e.get("buildingType"))
            if team in counts and variable:
                counts[team][variable] += 1
    return counts


def state_at(timeline: dict, minute: int) -> dict | None:
    """Usa el cuadro de índice minute y eventos hasta su propio timestamp.

    Los cuadros de Riot llegan aproximadamente cada minuto, con algo de demora.
    El índice define el minuto y el timestamp del cuadro define el corte temporal.
    """
    if not isinstance(minute, int) or minute < 0:
        raise ValueError("El minuto debe ser un entero mayor o igual a cero.")
    frames = timeline.get("info", {}).get("frames", [])
    if minute >= len(frames):
        return None
    frame = frames[minute]
    cutoff = frame["timestamp"]
    teams = participant_teams(timeline)
    state = dict.fromkeys(STATE_FEATURES, 0)
    for key, p in frame["participantFrames"].items():
        sign = 1 if teams[int(key)] == 100 else -1
        for name, field in [("oro", "totalGold"), ("experiencia", "xp"), ("niveles", "level")]:
            state[f"dif_{name}"] += sign * p[field]
        state["dif_subditos"] += sign * (p["minionsKilled"] + p["jungleMinionsKilled"])
    counts = event_counts(timeline, cutoff)
    for variable in counts[100]:
        if f"dif_{variable}" in state:
            state[f"dif_{variable}"] = counts[100][variable] - counts[200][variable]
    return state
