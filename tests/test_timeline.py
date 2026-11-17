import copy
import pytest
from src.features import timeline
from scripts.check_timeline_consistency import check


@pytest.fixture
def synthetic_timeline():
    # Deliberately inverted numeric IDs to catch accidental 1..5 assumptions.
    participants = [{"participantId": i, "teamId": 200 if i <= 5 else 100} for i in range(1, 11)]
    frames = []
    for minute in range(4):
        frames.append({"timestamp": minute * 60000, "participantFrames": {
            str(i): {"totalGold": 500 + minute * i, "xp": minute * i * 2, "level": 1 + minute,
                     "minionsKilled": minute * i, "jungleMinionsKilled": minute}
            for i in range(1, 11)}, "events": []})
    frames[1]["events"] = [
        {"type": "CHAMPION_KILL", "killerId": 6, "victimId": 1, "timestamp": 60000},
        {"type": "BUILDING_KILL", "buildingType": "TOWER_BUILDING", "killerId": 0, "teamId": 200, "timestamp": 59000},
        {"type": "ELITE_MONSTER_KILL", "monsterType": "DRAGON", "killerTeamId": 200, "timestamp": 60001},
        {"type": "TURRET_PLATE_DESTROYED", "teamId": 200, "timestamp": 30000}]
    return {"info": {"participants": participants, "frames": frames}}


def test_state_at(synthetic_timeline):
    state = timeline.state_at(synthetic_timeline, 1)
    assert state["dif_oro"] == 25
    assert state["dif_experiencia"] == 50
    assert state["dif_subditos"] == 25
    assert state["dif_niveles"] == 0
    assert state["dif_asesinatos"] == 1
    assert state["dif_torres"] == state["dif_placas"] == 1
    assert state["dif_dragones"] == 0
    assert timeline.state_at(synthetic_timeline, 2)["dif_dragones"] == -1
    assert timeline.state_at(synthetic_timeline, 4) is None
    assert timeline.state_at(synthetic_timeline, 0)["dif_oro"] == 0


def test_teams_require_evidence(synthetic_timeline, monkeypatch):
    for p in synthetic_timeline["info"]["participants"]:
        del p["teamId"]
    with pytest.raises(ValueError, match="comprobar"):
        timeline.state_at(synthetic_timeline, 1)
    synthetic_timeline["metadata"] = {"matchId": "SYN_1"}
    monkeypatch.setattr(timeline.cache, "load_match", lambda _: {"info": {"participants": [{"participantId": i, "teamId": 100 if i <= 5 else 200} for i in range(1, 11)]}})
    assert timeline.state_at(synthetic_timeline, 1)["dif_oro"] == -25


def test_all_objectives_and_time(synthetic_timeline):
    for monster, variable in timeline.MONSTERS.items():
        synthetic_timeline["info"]["frames"][2]["events"].append({"type": "ELITE_MONSTER_KILL", "monsterType": monster, "killerTeamId": 100, "timestamp": 120000})
    synthetic_timeline["info"]["frames"][2]["events"].append({"type": "BUILDING_KILL", "buildingType": "INHIBITOR_BUILDING", "teamId": 200, "timestamp": 120000})
    state = timeline.state_at(synthetic_timeline, 2)
    assert all(state[f"dif_{v}"] == 1 for v in ["larvas", "heraldos", "barones", "inhibidores"])
    assert state["dif_dragones"] == 0
    synthetic_timeline["info"]["frames"][-1]["timestamp"] = 179000
    assert timeline.state_at(synthetic_timeline, 3) is not None
    with pytest.raises(ValueError):
        timeline.state_at(synthetic_timeline, -1)


def test_jittered_frames_use_snapshot_time_without_future_events(synthetic_timeline):
    frames = synthetic_timeline["info"]["frames"]
    for frame, timestamp in zip(frames, [0, 60030, 120061, 180149]):
        frame["timestamp"] = timestamp
    # Events between the nominal minute and its snapshot belong to this state.
    frames[1]["events"].extend([
        {"type": "CHAMPION_KILL", "killerId": 6, "timestamp": 60030},
        {"type": "CHAMPION_KILL", "killerId": 6, "timestamp": 60031},
    ])
    # Even an event stored in a later frame is selected by its own timestamp.
    frames[2]["events"].extend([
        {"type": "ELITE_MONSTER_KILL", "monsterType": "BARON_NASHOR", "killerTeamId": 100, "timestamp": 60030},
        {"type": "ELITE_MONSTER_KILL", "monsterType": "BARON_NASHOR", "killerTeamId": 100, "timestamp": 60031},
    ])
    state = timeline.state_at(synthetic_timeline, 1)
    assert state["dif_oro"] == 25
    assert state["dif_asesinatos"] == 2
    assert state["dif_dragones"] == -1
    assert state["dif_barones"] == 1
    later = timeline.state_at(synthetic_timeline, 2)
    assert later["dif_oro"] == 50
    assert later["dif_asesinatos"] == 3
    assert later["dif_barones"] == 2
    assert timeline.state_at(synthetic_timeline, 3) is not None
    assert timeline.state_at(synthetic_timeline, 4) is None


def test_no_frames_has_no_state():
    assert timeline.state_at({}, 0) is None


def test_cached_objectives_consistency():
    result, skipped = check()
    if not any(r["total"] for r in result.values()):
        pytest.skip("No hay pares de partidas y líneas de tiempo locales.")
    assert skipped == 0
    for variable, row in result.items():
        if "dif_" + variable in timeline.STATE_FEATURES:
            assert (row["total"] - row["coinciden"]) / row["total"] <= .02
