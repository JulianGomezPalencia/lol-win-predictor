import json
import joblib
import pandas as pd
import pytest
from src.models import ingame
from src.features.timeline import STATE_FEATURES


@pytest.fixture
def rows():
    return [{"match_id": f"SYN_{i}", "minute": minute, "blue_win": i % 2,
             **{name: (1 if i % 2 else -1) * (i + minute) for name in STATE_FEATURES}}
            for minute in ingame.CURVE_MINUTES for i in range(12) if not (i == 0 and minute > 10)]


@pytest.fixture
def bundle(rows):
    return ingame.train(rows)


def test_recipe_and_survivors(bundle):
    assert len(bundle["models"]) == 30
    assert bundle["counts"][10] == 12
    assert bundle["counts"][11] == 11
    assert bundle["models"][15].named_steps["model"].C == 1.0
    assert bundle["models"][15].named_steps["scale"].n_samples_seen_ == 11


def test_probability_curve_and_load(bundle, tmp_path, monkeypatch):
    state = dict.fromkeys(STATE_FEATURES, 100)
    assert .5 < ingame.probability(bundle, 5, state) <= 1
    assert ingame.probability(bundle, 60, state) == ingame.probability(bundle, 30, state)
    monkeypatch.setattr(ingame, "state_at", lambda t, m: state if m < 4 else None)
    assert [p["minuto"] for p in ingame.curve(bundle, {}, .4)] == [0, 1, 2, 3]
    assert ingame.curve(bundle, {})[0]["minuto"] == 1
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    assert ingame.load_bundle(path)["features"] == STATE_FEATURES
    with pytest.raises(ValueError):
        ingame.probability(bundle, 0, state)


def test_insufficient_and_duplicate(rows):
    with pytest.raises(ValueError):
        ingame.train([])
    one_minute = [r for r in rows if r["minute"] == 1]
    bundle = ingame.train(one_minute)
    with pytest.raises(ValueError, match="No hay modelo"):
        ingame.probability(bundle, 2, rows[0])
    with pytest.raises(ValueError, match="repetidas"):
        ingame.train(one_minute + one_minute)


def test_cli_train_and_evaluate(rows, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ingame, "cached_rows", lambda *args: rows)
    path = tmp_path / "synthetic.joblib"
    ingame.main(["--train", "--model", str(path), "--before", "2026-11-09T00:00:00+00:00"])
    assert path.exists()
    capsys.readouterr()
    with pytest.raises(ValueError, match="entrenar"):
        ingame.main(["--evaluate", "--model", str(path)])
    held_out = [{**r, "match_id": "TEST_" + r["match_id"]} for r in rows]
    monkeypatch.setattr(ingame, "cached_rows", lambda *args: held_out)
    ingame.main(["--evaluate", "--model", str(path), "--after", "2026-11-09T00:00:00+00:00"])
    assert len(json.loads(capsys.readouterr().out)) == 5


def test_cached_rows_time_filters(tmp_path, monkeypatch):
    pd.DataFrame([{"match_id": f"SYN_{i}", "game_start_ts": i * 1000, "blue_win": i % 2, "status": "history_done"} for i in range(1, 4)]).to_csv(tmp_path / "matches.csv", index=False)
    monkeypatch.setattr(ingame.cache, "load_timeline", lambda _: {})
    monkeypatch.setattr(ingame, "state_at", lambda t, m: dict.fromkeys(STATE_FEATURES, 0) if m <= 2 else None)
    rows = ingame.cached_rows(tmp_path, before="1970-01-01T00:00:03+00:00", after="1970-01-01T00:00:01+00:00")
    assert {r["match_id"] for r in rows} == {"SYN_2"}
    assert len(rows) == 2
    with pytest.raises(ValueError, match="zona"):
        ingame._iso_ms("2026-11-09")


def test_cached_rows_and_curve_with_jittered_snapshots(tmp_path, monkeypatch):
    pd.DataFrame([{"match_id": "SYN_1", "blue_win": 1, "game_start_ts": 1000}]).to_csv(
        tmp_path / "matches.csv", index=False)
    timeline = {"info": {
        "participants": [{"participantId": i, "teamId": 100 if i <= 5 else 200} for i in range(1, 11)],
        "frames": [{"timestamp": timestamp, "events": [], "participantFrames": {
            str(i): {"totalGold": 500 + m * i, "xp": m * i, "level": 1,
                     "minionsKilled": m, "jungleMinionsKilled": 0} for i in range(1, 11)}}
            for m, timestamp in enumerate([0, 60030, 120061, 180149])],
    }}
    monkeypatch.setattr(ingame.cache, "load_timeline", lambda _: timeline)
    rows = ingame.cached_rows(tmp_path)
    assert [r["minute"] for r in rows] == [1, 2, 3]
    assert [r["dif_oro"] for r in rows] == [-25, -50, -75]
    monkeypatch.setattr(ingame, "probability", lambda bundle, minute, state: .5)
    assert ingame.curve({}, timeline) == [
        {"minuto": minute, "prob_azul": .5} for minute in (1, 2, 3)]
