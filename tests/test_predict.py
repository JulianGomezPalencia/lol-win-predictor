"""
Tests for src/models/predict.py. No real API calls; files only under tmp_path.

Run with:  pytest tests/test_predict.py -v
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.features import build
from src.models import predict

START = 10 * 86_400_000
DAY = 86_400_000
POSITIONS = build.POSITIONS


def fake_match(match_id="LA1_1", queue=420, duration=1800, blue_win=True):
    participants = []
    for team in (100, 200):
        for pos in POSITIONS:
            participants.append({"puuid": f"{team}_{pos}", "teamId": team, "teamPosition": pos,
                                 "championId": 1, "championName": "A",
                                 "win": (team == 100) == blue_win})
    return {"metadata": {"matchId": match_id},
            "info": {"queueId": queue, "gameDuration": duration, "gameStartTimestamp": START,
                     "gameVersion": "16.20.1.1", "participants": participants,
                     "teams": [{"teamId": 100, "win": blue_win}, {"teamId": 200, "win": not blue_win}]}}


def hist_game(rank, win, days_before=1, kills=2, deaths=1, assists=2, start=None):
    return {"history_rank": rank, "history_match_id": f"H{rank}",
            "game_start_ts": START - days_before * DAY if start is None else start,
            "game_duration": 1800, "remake": 0, "win": win, "champion_id": 1, "champion_name": "A",
            "team_position": "TOP", "kills": kills, "deaths": deaths, "assists": assists}


def all_histories(match, blue_games=(), red_games=()):
    return {p["puuid"]: list(blue_games if p["teamId"] == 100 else red_games)
            for p in match["info"]["participants"]}


def synthetic_table(n=120, seed=0):
    rng = np.random.default_rng(seed)
    kda = rng.normal(size=n)
    data = {"match_id": [f"M{i}" for i in range(n)], "game_start_ts": np.arange(n) * 1000,
            "seed_tier": rng.choice(["GOLD", "MASTER"], size=n),
            "blue_win": (kda + rng.normal(scale=0.5, size=n) > 0).astype(int), "dif_kda_medio": kda}
    for pos in POSITIONS:
        data[f"{pos.lower()}_dif_kda_medio"] = rng.normal(size=n)
    data["dif_en_linea_habitual"] = rng.normal(size=n)
    data["dif_fuera_de_posicion"] = rng.normal(size=n)
    data["top_dif_en_linea_habitual"] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_bundle_excludes_position_variables_and_round_trips(tmp_path):
    bundle = predict.fit_final_model(synthetic_table())
    assert not [f for f in bundle["features"] if "en_linea_habitual" in f or "fuera_de_posicion" in f]
    assert "dif_kda_medio" in bundle["features"]
    assert bundle["n_matches"] == 120 and bundle["auc_ic95"][0] <= bundle["auc_ic95"][1]
    path = predict.save_model(bundle, tmp_path / "sub" / "m.joblib")
    loaded = predict.load_model(path)
    assert loaded["features"] == bundle["features"]
    X = synthetic_table()[bundle["features"]]
    assert np.allclose(loaded["model"].predict_proba(X), bundle["model"].predict_proba(X))


def test_load_model_missing_file_tells_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="--train"):
        predict.load_model(tmp_path / "nope.joblib")


def equivalent_tables(match, blue_games, red_games):
    matches = pd.DataFrame([{"match_id": "LA1_1", "status": "history_done", "seed_tier": "UNKNOWN",
                             "game_start_ts": START, "patch": "16.20", "blue_win": 1}])
    parts = pd.DataFrame([{"match_id": "LA1_1", "player_id": p["puuid"], "team_id": p["teamId"],
                           "team_position": p["teamPosition"], "champion_id": 1, "champion_name": "A",
                           "win": 1} for p in match["info"]["participants"]])
    rows = [{"target_match_id": "LA1_1", "player_id": p["puuid"], **g}
            for p in match["info"]["participants"] for g in (blue_games if p["teamId"] == 100 else red_games)]
    return build.match_features(build.player_features(pd.DataFrame(rows), parts, matches), matches)


def test_features_for_match_matches_build_and_hand_values():
    match = fake_match()
    blue = [hist_game(0, 1), hist_game(1, 1, days_before=2)]   # win rate 1.0
    red = [hist_game(0, 0), hist_game(1, 1, days_before=2)]    # win rate 0.5
    row = predict.features_for_match(match, all_histories(match, blue, red))
    expected = equivalent_tables(match, blue, red)
    assert len(row) == 1
    assert list(row.columns) == list(expected.columns)
    assert row["dif_winrate_10"].iloc[0] == pytest.approx(0.5)
    assert row["dif_racha"].iloc[0] == pytest.approx(2 - (-1))
    assert row["top_dif_winrate_10"].iloc[0] == pytest.approx(0.5)
    pd.testing.assert_frame_equal(row, expected)


def test_games_at_or_after_match_start_never_reach_features():
    match = fake_match()
    leaked = [hist_game(0, 1, start=START), hist_game(1, 1, start=START + DAY)]
    ok = [hist_game(2, 0)]
    row = predict.features_for_match(match, all_histories(match, leaked + ok, ok))
    assert row["dif_winrate_10"].iloc[0] == pytest.approx(0.0)  # only the loss counts, both teams
    assert row["dif_partidas_previas"].iloc[0] == pytest.approx(0.0)


def test_empty_histories_still_give_one_row_and_a_prediction():
    match = fake_match()
    row = predict.features_for_match(match, all_histories(match))
    assert len(row) == 1
    assert row["dif_sin_historial"].iloc[0] == 0  # all ten without history


def patch_endpoints(monkeypatch, match, histories):
    from src.agents import history_builder
    from src.riot_client import endpoints
    monkeypatch.setattr(endpoints, "get_match", lambda match_id: match)
    monkeypatch.setattr(history_builder, "player_history",
                        lambda puuid, start, *a, **k: histories.get(puuid, []))


def test_predict_match_probabilities_sum_to_one_and_count_missing_history(monkeypatch):
    bundle = predict.fit_final_model(synthetic_table())
    match = fake_match()
    histories = {"100_TOP": [hist_game(0, 1)], "200_TOP": [hist_game(0, 0)]}  # 8 players without history
    patch_endpoints(monkeypatch, match, histories)
    result = predict.predict_match("LA1_1", bundle)
    assert result["prob_blue"] + result["prob_red"] == pytest.approx(1.0)
    assert 0 <= result["prob_blue"] <= 1 and not math.isnan(result["prob_blue"])
    assert result["blue_win"] == 1
    assert result["jugadores_sin_historial"] == 8


def test_predict_match_invalid_match_raises(monkeypatch):
    bundle = predict.fit_final_model(synthetic_table())
    patch_endpoints(monkeypatch, fake_match(duration=100), {})
    with pytest.raises(ValueError, match="remake"):
        predict.predict_match("LA1_1", bundle)


def test_match_in_training_data_is_flagged(monkeypatch):
    table = synthetic_table()
    bundle = predict.fit_final_model(table)
    assert "M0" in bundle["match_ids"] and len(bundle["match_ids"]) == 120
    patch_endpoints(monkeypatch, fake_match(match_id="M0"), {})
    assert predict.predict_match("M0", bundle)["en_entrenamiento"] is True
    patch_endpoints(monkeypatch, fake_match(match_id="LA1_NEW"), {})
    assert predict.predict_match("LA1_NEW", bundle)["en_entrenamiento"] is False


def test_old_bundle_without_ids_falls_back_to_timestamps():
    bundle = {"last_match_ts": START}
    assert predict.in_training_data("X", START, bundle) is True
    assert predict.in_training_data("X", START + 1, bundle) is False


def test_remake_only_player_counts_as_without_history(monkeypatch):
    bundle = predict.fit_final_model(synthetic_table())
    remake = {**hist_game(0, 1), "remake": 1}
    patch_endpoints(monkeypatch, fake_match(), {"100_TOP": [remake]})
    assert predict.predict_match("LA1_1", bundle)["jugadores_sin_historial"] == 10


def test_missing_model_feature_raises(monkeypatch):
    bundle = predict.fit_final_model(synthetic_table())
    bundle["features"] = bundle["features"] + ["variable_que_no_existe"]
    patch_endpoints(monkeypatch, fake_match(), {})
    with pytest.raises(ValueError, match="variable_que_no_existe"):
        predict.predict_match("LA1_1", bundle)


def test_prediction_with_real_row_shape_and_empty_histories_runs_the_model():
    bundle = predict.fit_final_model(synthetic_table())
    match = fake_match()
    row = predict.features_for_match(match, all_histories(match))[bundle["features"]]
    assert 0 <= bundle["model"].predict_proba(row)[0, 1] <= 1


def test_import_and_help_work_without_riot_api_key():
    import os
    import subprocess
    import sys
    env = {**os.environ, "RIOT_API_KEY": ""}
    result = subprocess.run([sys.executable, "-m", "src.models.predict", "--help"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
