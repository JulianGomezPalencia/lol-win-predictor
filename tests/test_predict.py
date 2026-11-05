"""
Tests for src/models/predict.py. No real API calls; files only under tmp_path.

Run with:  pytest tests/test_predict.py -v
"""
import copy
import csv
import math

import numpy as np
import pandas as pd
import pytest
import requests

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
                                 "summoner1Id": 11 if pos == "JUNGLE" else 4, "summoner2Id": 14,
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


@pytest.fixture(scope="module")
def bundle():
    """One fit for the whole module (fit_final_model runs a walk-forward, which is slow)."""
    return predict.fit_final_model(synthetic_table())


def test_bundle_excludes_position_variables_and_round_trips(tmp_path, bundle):
    assert not [f for f in bundle["features"] if "en_linea_habitual" in f or "fuera_de_posicion" in f]
    assert "dif_kda_medio" in bundle["features"]
    assert bundle["roles"] == "inferred" and bundle["trained_at"]
    assert bundle["n_matches"] == 120 and bundle["auc_ic95"][0] <= bundle["auc_ic95"][1]
    path = predict.save_model(bundle, tmp_path / "sub" / "m.joblib")
    loaded = predict.load_model(path)
    assert loaded["features"] == bundle["features"]
    X = synthetic_table()[bundle["features"]]
    assert np.allclose(loaded["model"].predict_proba(X), bundle["model"].predict_proba(X))


def test_load_model_missing_file_tells_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="--train"):
        predict.load_model(tmp_path / "nope.joblib")


def test_old_bundle_without_roles_is_refused_everywhere(bundle, monkeypatch):
    old = {k: v for k, v in bundle.items() if k != "roles"}
    with pytest.raises(ValueError, match="--train"):
        predict.check_bundle(old)
    with pytest.raises(ValueError, match="--train"):
        predict.predict_match("LA1_1", old)
    with pytest.raises(ValueError, match="--train"):
        predict.evaluate_new(old, synthetic_table())
    with pytest.raises(ValueError, match="--train"):
        predict.predict_live("A#B", old)


def test_features_for_match_inferred_roles_and_hand_values():
    match = fake_match()
    blue = [hist_game(0, 1), hist_game(1, 1, days_before=2)]   # win rate 1.0
    red = [hist_game(0, 0), hist_game(1, 1, days_before=2)]    # win rate 0.5
    row = predict.features_for_match(match, all_histories(match, blue, red))
    assert len(row) == 1
    assert row["dif_winrate_10"].iloc[0] == pytest.approx(0.5)
    assert row["dif_racha"].iloc[0] == pytest.approx(2 - (-1))
    assert row["top_dif_winrate_10"].iloc[0] == pytest.approx(0.5)  # a lane exists for every position


def test_features_ignore_riots_team_position():
    match = fake_match()
    shuffled = copy.deepcopy(match)
    for p in shuffled["info"]["participants"]:
        p["teamPosition"] = "TOP"
    histories = all_histories(match, [hist_game(0, 1)], [hist_game(0, 0)])
    pd.testing.assert_frame_equal(predict.features_for_match(match, histories),
                                  predict.features_for_match(shuffled, histories))


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


def test_predict_match_probabilities_sum_to_one_and_count_missing_history(monkeypatch, bundle):
    match = fake_match()
    histories = {"100_TOP": [hist_game(0, 1)], "200_TOP": [hist_game(0, 0)]}  # 8 players without history
    patch_endpoints(monkeypatch, match, histories)
    result = predict.predict_match("LA1_1", bundle)
    assert result["prob_blue"] + result["prob_red"] == pytest.approx(1.0)
    assert 0 <= result["prob_blue"] <= 1 and not math.isnan(result["prob_blue"])
    assert result["blue_win"] == 1
    assert result["jugadores_sin_historial"] == 8


def test_predict_match_invalid_match_raises(monkeypatch, bundle):
    patch_endpoints(monkeypatch, fake_match(duration=100), {})
    with pytest.raises(ValueError, match="remake"):
        predict.predict_match("LA1_1", bundle)


def test_match_in_training_data_is_flagged(monkeypatch, bundle):
    assert "M0" in bundle["match_ids"] and len(bundle["match_ids"]) == 120
    patch_endpoints(monkeypatch, fake_match(match_id="M0"), {})
    assert predict.predict_match("M0", bundle)["en_entrenamiento"] is True
    patch_endpoints(monkeypatch, fake_match(match_id="LA1_NEW"), {})
    assert predict.predict_match("LA1_NEW", bundle)["en_entrenamiento"] is False


def test_old_bundle_without_ids_falls_back_to_timestamps():
    bundle = {"last_match_ts": START}
    assert predict.in_training_data("X", START, bundle) is True
    assert predict.in_training_data("X", START + 1, bundle) is False


def test_remake_only_player_counts_as_without_history(monkeypatch, bundle):
    remake = {**hist_game(0, 1), "remake": 1}
    patch_endpoints(monkeypatch, fake_match(), {"100_TOP": [remake]})
    assert predict.predict_match("LA1_1", bundle)["jugadores_sin_historial"] == 10


def test_missing_model_feature_raises(monkeypatch, bundle):
    broken = {**bundle, "features": bundle["features"] + ["variable_que_no_existe"]}
    patch_endpoints(monkeypatch, fake_match(), {})
    with pytest.raises(ValueError, match="variable_que_no_existe"):
        predict.predict_match("LA1_1", broken)


# --- live ---------------------------------------------------------------------------------

SECRET = "SECRETPUUID"


def live_game(queue=420, start=0, hidden=False, blocked=False):
    players = []
    for team in (100, 200):
        for i in range(5):
            p = {"puuid": f"{SECRET}_{team}_{i}", "teamId": team, "championId": 10 + i,
                 "spell1Id": 11 if i == 1 else 4, "spell2Id": 14}
            players.append(p)
    if hidden:
        del players[0]["puuid"]
    if blocked:
        players[1]["puuid"] = "ab--cd"
    return {"gameId": 999, "platformId": "LA1", "gameQueueConfigId": queue, "gameStartTime": start,
            "participants": players}


@pytest.fixture
def live_env(monkeypatch):
    from src.agents import history_builder
    from src.riot_client import endpoints
    state = {"game": live_game(), "cutoffs": [], "asked": []}
    monkeypatch.setattr(endpoints, "get_puuid", lambda name, tag: "ME")
    monkeypatch.setattr(endpoints, "get_active_game", lambda puuid: state["game"])

    def history(puuid, start, *a, **k):
        state["asked"].append(puuid)
        state["cutoffs"].append(start)
        return [hist_game(0, 1)]
    monkeypatch.setattr(history_builder, "player_history", history)
    return state


def run_live(bundle, tmp_path, **kw):
    lines = []
    result = predict.predict_live("Name#TAG", bundle, progress=lines.append, log_path=tmp_path / "log.csv",
                                  names={10: "Aatrox"}, **kw)
    return result, lines


def test_live_not_in_game(live_env, bundle, tmp_path):
    live_env["game"] = None
    with pytest.raises(predict.LiveError, match="no está en una partida"):
        run_live(bundle, tmp_path)
    assert not (tmp_path / "log.csv").exists()


def test_live_wrong_queue(live_env, bundle, tmp_path):
    live_env["game"] = live_game(queue=450)
    with pytest.raises(predict.LiveError, match="450"):
        run_live(bundle, tmp_path)


def test_live_probabilities_roles_and_no_puuid_in_output(live_env, bundle, tmp_path):
    result, lines = run_live(bundle, tmp_path)
    assert result["prob_blue"] + result["prob_red"] == pytest.approx(1.0)
    assert [pos for pos, _ in result["teams"][100]] == build.POSITIONS
    labels = dict((champion, pos) for pos, champion in result["teams"][100])
    assert set(labels) == {"Aatrox", "campeón 11", "campeón 12", "campeón 13", "campeón 14"}
    assert labels["campeón 11"] == "JUNGLE"  # the only one with Smite
    assert len(lines) == 10 and result["jugadores_sin_historial"] == 0
    printed = repr(result) + repr(lines) + (tmp_path / "log.csv").read_text()
    assert SECRET not in printed


def test_live_game_start_time_zero_uses_now_and_real_time_is_kept(live_env, bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(predict.time, "time", lambda: 1_700_000_000)
    run_live(bundle, tmp_path)
    assert set(live_env["cutoffs"]) == {1_700_000_000_000}
    live_env["cutoffs"].clear()
    live_env["game"] = live_game(start=1_600_000_000_000)
    run_live(bundle, tmp_path)
    assert set(live_env["cutoffs"]) == {1_600_000_000_000}


def test_live_hidden_and_blocked_players_have_no_history_but_are_predicted(live_env, bundle, tmp_path):
    live_env["game"] = live_game(hidden=True, blocked=True)
    result, _ = run_live(bundle, tmp_path)
    assert len(live_env["asked"]) == 8  # hidden and "--" players are never queried
    assert result["jugadores_sin_historial"] == 2
    assert 0 <= result["prob_blue"] <= 1


def test_live_same_game_twice_logs_one_row(live_env, bundle, tmp_path):
    first, _ = run_live(bundle, tmp_path)
    second, _ = run_live(bundle, tmp_path)
    assert first["logged"] is True and second["logged"] is False
    rows = list(csv.DictReader(open(tmp_path / "log.csv")))
    assert len(rows) == 1
    assert rows[0]["game_id"] == "999" and rows[0]["platform"] == "LA1"
    assert rows[0]["model_trained_at"] == bundle["trained_at"]
    assert float(rows[0]["prob_blue"]) == pytest.approx(first["prob_blue"], abs=1e-6)


def test_parse_riot_id():
    assert predict.parse_riot_id("A B#LAS") == ("A B", "LAS")
    with pytest.raises(predict.LiveError):
        predict.parse_riot_id("sin etiqueta")


# --- score-log ----------------------------------------------------------------------------


def write_log(path, probs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(predict.LOG_COLUMNS)
        for i, p in enumerate(probs):
            w.writerow([i, "LA1", "2026-11-04T00:00:00+00:00", p, "t"])


def http_404():
    return requests.HTTPError("404", response=type("R", (), {"status_code": 404})())


def test_score_log_skips_unfinished_games_and_says_too_few(tmp_path):
    write_log(tmp_path / "l.csv", [0.6, 0.4, 0.7])
    finished = {"LA1_0": fake_match(blue_win=True), "LA1_1": fake_match(blue_win=False)}

    def get_match(match_id):
        if match_id not in finished:
            raise http_404()
        return finished[match_id]
    r = predict.score_log(tmp_path / "l.csv", get_match)
    assert (r["n"], r["pendientes"], r["accuracy"], r["auc"]) == (2, 1, 1.0, None)


def test_score_log_auc_with_enough_games_and_both_classes(tmp_path):
    probs = [0.9, 0.8, 0.7, 0.6, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1, 0.35]
    write_log(tmp_path / "l.csv", probs)
    r = predict.score_log(tmp_path / "l.csv", lambda mid: fake_match(blue_win=int(mid.split("_")[1]) < 5))
    assert r["n"] == 11 and r["auc"] is not None and r["auc"] > 0.9


def test_score_log_without_a_log_file(tmp_path):
    assert predict.score_log(tmp_path / "none.csv")["n"] == 0


# --- evaluate-new -------------------------------------------------------------------------


def test_evaluate_new_excludes_training_matches_and_never_refits(bundle, monkeypatch):
    table = synthetic_table(n=160)  # same generator: first 120 ids M0..M119 ... but ids repeat for 0..119
    table["game_start_ts"] = np.arange(160) * 86_400_000
    frozen = copy.deepcopy(bundle)
    fitted = []
    monkeypatch.setattr(type(bundle["model"]), "fit", lambda self, *a, **k: fitted.append(1))
    r = predict.evaluate_new(frozen, table)
    assert r["n"] == 40 and not fitted
    assert r["accuracy_ic95"][0] <= r["accuracy"] <= r["accuracy_ic95"][1]
    assert r["auc_ic95"][0] <= r["auc"] <= r["auc_ic95"][1] and r["log_loss"] > 0
    assert r["desde"] <= r["hasta"] and 0 <= r["base_accuracy"] <= 1
    assert frozen["model"] is not bundle["model"]
    X = table[bundle["features"]]
    assert np.allclose(frozen["model"].predict_proba(X), bundle["model"].predict_proba(X))


def test_evaluate_new_with_no_new_matches(bundle):
    assert predict.evaluate_new(bundle, synthetic_table(n=120)) == {"n": 0}


def test_import_and_help_work_without_riot_api_key():
    import os
    import subprocess
    import sys
    env = {**os.environ, "RIOT_API_KEY": ""}
    result = subprocess.run([sys.executable, "-m", "src.models.predict", "--help"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
