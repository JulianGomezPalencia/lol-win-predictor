"""
Tests for src/models/predict.py. No real API calls; files only under tmp_path.

A small synthetic export (110 matches) is built once, frozen once, and shared by the tests below
(fitting is slow). The real constants (22 variables / 920 / 1,206) are patched to the synthetic sizes.

Run with:  pytest tests/test_predict.py -v
"""
import copy
import csv
import json
import math
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
import requests
import sklearn
from sklearn.pipeline import Pipeline

from src.features import build, champion_pool
from src.models import predict, train

START = 10 * 86_400_000
DAY = 86_400_000
POSITIONS = build.POSITIONS
N_FIT, N_SELECTION, N_TEST = 60, 20, 30
N_DEV = N_FIT + N_SELECTION


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


# --- a synthetic export, frozen once ------------------------------------------------------------

FIRST_TS = 1_700_000_000_000


def write_export(folder, n_matches):
    """Matches M000.. one per day, 10 players each, 3 earlier games per player (a few have none)."""
    folder.mkdir(parents=True, exist_ok=True)
    matches, participants, history, spells = [], [], [], []
    for i in range(n_matches):
        rng = np.random.default_rng(i)  # a match is the same whatever n_matches is
        ts = FIRST_TS + i * DAY
        strength = rng.normal()
        matches.append({"match_id": f"M{i:03d}", "status": "history_done", "seed_tier": ["GOLD", "MASTER"][i % 2],
                        "game_start_ts": ts, "patch": "16.20", "blue_win": int(strength + rng.normal(scale=.7) > 0),
                        "discovered_at": "x"})
        for team in (100, 200):
            for pos in POSITIONS:
                player = f"p_{i}_{team}_{pos}"
                champion = int(rng.integers(1, 7))
                participants.append({"match_id": f"M{i:03d}", "player_id": player, "team_id": team,
                                     "team_position": pos, "champion_id": champion, "champion_name": f"C{champion}",
                                     "win": 0})
                spells.append({"match_id": f"M{i:03d}", "player_id": player, "spell1_id": 11 if pos == "JUNGLE" else 4,
                               "spell2_id": 14})
                if rng.random() < .1:
                    continue  # no history
                for k in range(3):
                    sign = 1 if team == 100 else -1
                    history.append({"target_match_id": f"M{i:03d}", "player_id": player, "history_rank": k,
                                    "history_match_id": f"H{i}_{team}_{pos}_{k}",
                                    "game_start_ts": ts - int((k + 1 + rng.random()) * DAY * .9),
                                    "game_duration": 1800, "remake": 0, "win": int(rng.random() < .5 + .1 * sign * strength),
                                    "champion_id": int(rng.integers(1, 7)), "champion_name": "C",
                                    "team_position": pos, "kills": int(rng.integers(0, 12)),
                                    "deaths": int(rng.integers(0, 8)), "assists": int(rng.integers(0, 15))})
    for name, rows in (("matches", matches), ("participants", participants), ("player_history", history),
                       ("participant_spells", spells)):
        pd.DataFrame(rows).to_csv(folder / f"{name}.csv", index=False)


def write_base_model(path, export_dir, n_fit):
    """A stand-in for the saved 22-variable model: version-less bundle fitted on the first n_fit matches."""
    table = build.build(export_dir, inferred_positions=True)
    table = table.sort_values("game_start_ts").head(n_fit)
    features = [c for c in build.feature_columns(table) if table[c].notna().any()][:22]
    search = train.logistic_model()
    search.fit(table[features], table["blue_win"], model__sample_weight=train.tier_weights(table["seed_tier"]))
    bundle = {"model": search.best_estimator_, "features": features, "roles": "inferred", "n_matches": n_fit,
              "match_ids": sorted(table["match_id"]), "last_match_ts": int(table["game_start_ts"].max()),
              "blue_win_rate": float(table["blue_win"].mean()), "sklearn_version": sklearn.__version__,
              "trained_at": "2026-10-20T00:00:00+00:00", "auc_temporal": .55, "auc_ic95": (.5, .6)}
    joblib.dump(bundle, path)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Export A (development only), export B (development + new matches), a models folder before freeze."""
    root = tmp_path_factory.mktemp("world")
    write_export(root / "A", N_DEV)
    write_export(root / "B", N_DEV + N_TEST)
    (root / "models0").mkdir()
    write_base_model(root / "models0" / "modelo_final.joblib", root / "A", N_FIT)
    return root


@pytest.fixture(autouse=True)
def synthetic_constants(monkeypatch):
    monkeypatch.setattr(predict, "BASE_FEATURE_COUNT", 22)
    monkeypatch.setattr(predict, "BASE_MATCH_COUNT", N_FIT)
    monkeypatch.setattr(predict, "DEVELOPMENT_COUNT", N_DEV)


def copy_models(world, dest):
    shutil.copytree(world / "models0", dest)
    return dest


@pytest.fixture(scope="module")
def frozen(world, tmp_path_factory):
    """(models_dir, bundle) after one --freeze on export A. Module-scoped: tests must copy before changing."""
    mp = pytest.MonkeyPatch()
    mp.setattr(predict, "BASE_MATCH_COUNT", N_FIT)
    mp.setattr(predict, "DEVELOPMENT_COUNT", N_DEV)
    models = copy_models(world, tmp_path_factory.mktemp("frozen") / "models")
    bundle = predict.freeze(world / "A", models)
    mp.undo()
    return models, bundle


@pytest.fixture
def bundle(frozen):
    return frozen[1]


@pytest.fixture
def models_dir(world, frozen, tmp_path):
    """A private copy of the frozen models folder."""
    return shutil.copytree(frozen[0], tmp_path / "models")


# --- bundle ----------------------------------------------------------------------------------------


def test_bundle_is_version_2_with_the_frozen_variables_and_round_trips(tmp_path, bundle):
    assert bundle["version"] == 2 and bundle["roles"] == "inferred" and bundle["features"] == build.FINAL_FEATURES
    assert bundle["sensibilidad"]["features"] == build.SENSITIVITY_FEATURES and len(bundle["sensibilidad"]["features"]) == 9
    assert bundle["n_matches"] == N_FIT and len(bundle["development_ids"]) == N_DEV
    assert bundle["auc_ic95"][0] <= bundle["auc_ic95"][1]
    path = predict.save_model(bundle, tmp_path / "sub" / "m.joblib")
    loaded = predict.load_model(path)
    assert loaded["features"] == bundle["features"] and loaded["development_ids"] == bundle["development_ids"]
    pd.testing.assert_frame_equal(loaded["pool"], bundle["pool"])


def test_load_model_missing_file_tells_to_freeze(tmp_path):
    with pytest.raises(FileNotFoundError, match="--freeze"):
        predict.load_model(tmp_path / "nope.joblib")


def test_old_bundles_are_refused_everywhere_with_the_message(bundle):
    old = {k: v for k, v in bundle.items() if k != "version"}
    no_roles = {k: v for k, v in bundle.items() if k != "roles"}
    for refused in (old, no_roles):
        with pytest.raises(ValueError, match="--train"):
            predict.check_bundle(refused)
        with pytest.raises(ValueError, match="--train"):
            predict.predict_match("LA1_1", refused)
        with pytest.raises(ValueError, match="--train"):
            predict.predict_live("A#B", refused)


# --- train / serve parity ------------------------------------------------------------------------------


def players_and_histories(export_dir, match_id):
    participants = pd.read_csv(export_dir / "participants.csv")
    spells = pd.read_csv(export_dir / "participant_spells.csv").set_index(["match_id", "player_id"])
    history = pd.read_csv(export_dir / "player_history.csv")
    rows = participants[participants["match_id"] == match_id]
    players = [{"key": r.player_id, "team_id": r.team_id, "champion_id": r.champion_id,
                "champion_name": r.champion_name,
                "spell_ids": tuple(int(x) for x in spells.loc[(match_id, r.player_id)][["spell1_id", "spell2_id"]])}
               for r in rows.itertuples()]
    mine = history[history["target_match_id"] == match_id]
    histories = {p["key"]: mine[mine["player_id"] == p["key"]].drop(columns=["target_match_id", "player_id"])
                 .to_dict("records") for p in players}
    return players, histories


def test_batch_and_single_game_paths_give_identical_values_for_all_ten_variables(world, bundle):
    export = world / "A"
    table = build.build(export, inferred_positions=True, pool=bundle["pool"],
                        champion_positions=bundle["champion_positions"], final_features=True).set_index("match_id")
    matches = pd.read_csv(export / "matches.csv").set_index("match_id")
    saw_player_without_history = False
    for match_id in list(table.index[::11]):
        players, histories = players_and_histories(export, match_id)
        saw_player_without_history |= any(not h for h in histories.values())
        single, _ = predict.features_for_game(match_id, int(matches.loc[match_id, "game_start_ts"]), players,
                                              histories, pool=bundle["pool"],
                                              champion_positions=bundle["champion_positions"])
        np.testing.assert_allclose(single.iloc[0][build.FINAL_FEATURES].to_numpy(dtype=float),
                                   table.loc[match_id, build.FINAL_FEATURES].to_numpy(dtype=float), atol=1e-12)
    assert saw_player_without_history  # the parity includes a player with no history


def test_features_for_game_is_pure_and_ignores_riots_team_position():
    match = fake_match()
    blue = [hist_game(0, 1), hist_game(1, 1, days_before=2)]
    red = [hist_game(0, 0), hist_game(1, 1, days_before=2)]
    shuffled = copy.deepcopy(match)
    for p in shuffled["info"]["participants"]:
        p["teamPosition"] = "TOP"
    one, _ = predict.features_for_game("G", START, predict.players_from_match(match), all_histories(match, blue, red))
    two, _ = predict.features_for_game("G", START, predict.players_from_match(shuffled), all_histories(match, blue, red))
    pd.testing.assert_frame_equal(one, two)
    assert one["dif_winrate_10"].iloc[0] == pytest.approx(0.5)


def test_games_at_or_after_match_start_never_reach_features():
    match = fake_match()
    leaked = [hist_game(0, 1, start=START), hist_game(1, 1, start=START + DAY)]
    ok = [hist_game(2, 0)]
    row, _ = predict.features_for_game("G", START, predict.players_from_match(match),
                                       all_histories(match, leaked + ok, ok))
    assert row["dif_winrate_10"].iloc[0] == pytest.approx(0.0)
    assert row["dif_partidas_previas"].iloc[0] == pytest.approx(0.0)



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
    fitted = bundle["match_ids"][0]
    patch_endpoints(monkeypatch, fake_match(match_id=fitted), {})
    assert predict.predict_match(fitted, bundle)["en_entrenamiento"] is True
    patch_endpoints(monkeypatch, fake_match(match_id="LA1_NEW"), {})
    assert predict.predict_match("LA1_NEW", bundle)["en_entrenamiento"] is False
    assert predict.in_training_data(fitted, bundle) and not predict.in_training_data("LA1_NEW", bundle)


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
def live_env(monkeypatch, tmp_path):
    from src.agents import history_builder
    from src.riot_client import endpoints
    monkeypatch.setattr(endpoints, "get_solo_queue_entry", lambda key: None)
    monkeypatch.setattr(endpoints, "get_champion_mastery", lambda key, champion: None)
    monkeypatch.setattr(predict, "MODELS_DIR", tmp_path / "models")
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


def test_live_works_with_a_version_2_bundle_and_prints_no_puuid(live_env, bundle, tmp_path, capsys):
    assert bundle["version"] == 2
    result, lines = run_live(bundle, tmp_path)
    assert 0 <= result["prob_blue"] <= 1
    printed = repr(result) + repr(lines) + (tmp_path / "log.csv").read_text()
    assert SECRET not in printed


def test_live_refuses_an_old_bundle(live_env, bundle, tmp_path):
    old = {k: v for k, v in bundle.items() if k != "version"}
    with pytest.raises(ValueError, match="--train"):
        run_live(old, tmp_path)


# --- --freeze ---------------------------------------------------------------------------------------


def test_freeze_writes_everything_and_keeps_the_base_byte_for_byte(world, frozen):
    models, bundle = frozen
    assert (models / "modelo_base_22.joblib").read_bytes() == (world / "models0" / "modelo_final.joblib").read_bytes()
    assert (models / "modelo_candidato.joblib").exists() and (models / "modelo_final.joblib").exists()
    rows = list(csv.DictReader(open(models / "conjunto_desarrollo.csv")))
    assert len(rows) == N_DEV and {r["uso"] for r in rows} == {"ajuste", "seleccion"}
    assert [r["match_id"] for r in rows if r["uso"] == "ajuste"] == bundle["match_ids"]


def test_freeze_pool_and_shares_hold_only_rows_from_the_fitted_matches_and_no_player_ids(world, frozen):
    history = pd.read_csv(world / "A" / "player_history.csv")
    bundle = frozen[1]
    fitted = history[history["target_match_id"].isin(bundle["match_ids"])]
    everything = champion_pool.build_pool(history)
    assert len(bundle["pool"]) == len(champion_pool.build_pool(fitted)) < len(everything)
    assert list(bundle["pool"].columns) == champion_pool.POOL_COLUMNS
    assert not any(str(v).startswith("p_") for v in bundle["pool"].to_numpy().ravel())
    assert bundle["champion_positions"] == champion_pool.champion_position_shares(champion_pool.build_pool(fitted))


def test_freeze_refuses_a_second_run(world, frozen, models_dir):
    with pytest.raises(predict.PredictError, match="una vez"):
        predict.freeze(world / "A", models_dir)


def test_freeze_refuses_a_wrong_count_and_writes_nothing(world, tmp_path, monkeypatch):
    models = copy_models(world, tmp_path / "models")
    monkeypatch.setattr(predict, "DEVELOPMENT_COUNT", N_DEV + 1)
    with pytest.raises(predict.PredictError, match="exactamente"):
        predict.freeze(world / "A", models)
    assert sorted(p.name for p in models.iterdir()) == ["modelo_final.joblib"]


def test_freeze_refuses_a_model_that_is_not_the_22_variable_base(world, tmp_path, monkeypatch):
    models = copy_models(world, tmp_path / "models")
    monkeypatch.setattr(predict, "BASE_FEATURE_COUNT", 21)
    with pytest.raises(predict.PredictError, match="no es el modelo"):
        predict.freeze(world / "A", models)
    assert sorted(p.name for p in models.iterdir()) == ["modelo_final.joblib"]


# --- --evaluate-new -----------------------------------------------------------------------------------


def run_evaluate(export, models, monkeypatch=None):
    lines = []
    return predict.evaluate_new_command(export, models, out=lines.append), lines


def test_evaluate_new_excludes_development_matches_scores_the_same_ids_and_never_refits(world, models_dir, monkeypatch):
    loaded = []
    real_load = predict.load_model
    monkeypatch.setattr(predict, "load_model", lambda path=predict.MODEL_PATH: loaded.append(path.name) or real_load(path))

    def refit(*a, **k):
        raise AssertionError("evaluate-new must not fit anything")
    monkeypatch.setattr(Pipeline, "fit", refit)
    monkeypatch.setattr(train, "logistic_model", refit)
    before = {p.name: p.read_bytes() for p in models_dir.iterdir()}
    result, lines = run_evaluate(world / "B", models_dir)
    assert result["n"] == N_TEST and result["excluidas_desarrollo"] == N_DEV
    assert sorted(loaded) == ["modelo_base_22.joblib", "modelo_candidato.joblib"]  # never modelo_final
    assert {p.name: p.read_bytes() for p in models_dir.iterdir() if p.name != "prueba_final.json"} == before
    assert set(result["modelos"]) == {"base", "candidato", "nueve", "constante"}
    assert "poca potencia" in "\n".join(lines)  # n < 1,000
    assert result["desde"] <= result["hasta"] and sum(result["por_tier"].values()) == N_TEST
    for m in result["modelos"].values():
        assert m["accuracy_ic95"][0] <= m["accuracy"] <= m["accuracy_ic95"][1]


def test_evaluate_new_uses_old_roles_for_the_base_and_the_frozen_pool_for_the_candidate(world, models_dir, monkeypatch):
    calls = []
    real_build = build.build

    def spy(export_dir, inferred_positions=False, **kwargs):
        calls.append({"inferred": inferred_positions, **kwargs})
        return real_build(export_dir, inferred_positions, **kwargs)
    monkeypatch.setattr(build, "build", spy)
    run_evaluate(world / "B", models_dir)
    base_call = [c for c in calls if not c.get("final_features")]
    cand_call = [c for c in calls if c.get("final_features")]
    assert len(base_call) == 1 and base_call[0]["inferred"] is True
    assert "champion_positions" not in base_call[0] and "pool" not in base_call[0]  # the old role method
    assert len(cand_call) == 1 and cand_call[0]["champion_positions"] and cand_call[0]["pool"] is not None


def test_evaluate_new_with_no_test_matches_says_so_and_writes_nothing(world, models_dir):
    result, lines = run_evaluate(world / "A", models_dir)
    assert result["n"] == 0 and any("0 partidas de prueba" in l and "No se escribió nada" in l for l in lines)
    assert not (models_dir / "prueba_final.json").exists()


def test_evaluate_new_counts_every_run_and_appends(world, models_dir):
    _, first = run_evaluate(world / "B", models_dir)
    _, second = run_evaluate(world / "B", models_dir)
    assert "anteriores de esta prueba: 0" in first[-1]
    assert "anteriores de esta prueba: 1" in second[-1] and "no añade evidencia" in second[-1]
    runs = json.loads((models_dir / "prueba_final.json").read_text())["runs"]
    assert len(runs) == 2 and runs[0]["n"] == N_TEST and "veredicto" in runs[0]


def test_evaluate_new_prints_metrics_paired_differences_and_a_verdict(world, models_dir):
    _, lines = run_evaluate(world / "B", models_dir)
    text = "\n".join(lines)
    for expected in ("Base (", "Candidato (", "Sensibilidad (9 variables)", "Constante", "log loss", "ROC AUC",
                     "Diferencias emparejadas", "Veredicto"):
        assert expected in text


def test_evaluate_new_refuses_tables_that_do_not_hold_the_same_matches(world, models_dir):
    base, candidate = predict.load_model(models_dir / "modelo_base_22.joblib"), predict.load_model(models_dir / "modelo_candidato.joblib")
    cand_table = build.build(world / "B", inferred_positions=True, pool=candidate["pool"],
                             champion_positions=candidate["champion_positions"], final_features=True)
    base_table = build.build(world / "B", inferred_positions=True)
    with pytest.raises(ValueError, match="mismas partidas"):
        predict.evaluate_new(base, candidate, base_table.iloc[:-3], cand_table)


def test_evaluate_new_excludes_matches_played_before_the_fit_ended(world, models_dir):
    base, candidate = predict.load_model(models_dir / "modelo_base_22.joblib"), predict.load_model(models_dir / "modelo_candidato.joblib")
    candidate = {**candidate, "development_ids": candidate["development_ids"][:-10],  # 10 selection matches become "new"...
                 "last_match_ts": candidate["last_match_ts"]}
    cand_table = build.build(world / "B", inferred_positions=True, pool=candidate["pool"],
                             champion_positions=candidate["champion_positions"], final_features=True)
    result = predict.evaluate_new(base, candidate, build.build(world / "B", inferred_positions=True), cand_table)
    assert result["excluidas_por_fecha"] >= 0 and result["n"] == N_TEST + 10 - result["excluidas_por_fecha"]
    assert result["n"] >= N_TEST


@pytest.fixture
def quick_validation(monkeypatch):
    """--train tests check files and refusals, not the walk-forward (slow): stub it."""
    row = pd.DataFrame({"roc_auc": [.55], "auc_ic95_bajo": [.5], "auc_ic95_alto": [.6]}, index=[predict.LOGISTIC])
    monkeypatch.setattr(train, "temporal_validation", lambda table, **k: {"tabla": row})


# --- --train ----------------------------------------------------------------------------------------


def test_train_refuses_without_a_test_result(world, models_dir):
    before = (models_dir / "modelo_final.joblib").read_bytes()
    with pytest.raises(predict.PredictError, match="--evaluate-new"):
        predict.train_command(world / "B", models_dir, out=lambda *_: None)
    assert (models_dir / "modelo_final.joblib").read_bytes() == before


def test_train_force_warns_and_refits_on_every_match_touching_only_modelo_final(world, models_dir, quick_validation):
    lines = []
    current = predict.load_model(models_dir / "modelo_final.joblib")
    backup_name = f"modelo_final_{current['trained_at'][:10]}.joblib"
    before = (models_dir / "modelo_final.joblib").read_bytes()
    others = {p.name: p.read_bytes() for p in models_dir.iterdir() if p.name != "modelo_final.joblib"}
    final = predict.train_command(world / "B", models_dir, force=True, out=lines.append)
    assert any("ADVERTENCIA" in l and "CONTAMINA" in l for l in lines)
    assert final["n_matches"] == N_DEV + N_TEST and final["development_ids"] == predict.load_model(
        models_dir / "modelo_candidato.joblib")["development_ids"]
    assert (models_dir / backup_name).read_bytes() == before
    assert {p.name: p.read_bytes() for p in models_dir.iterdir()
            if p.name not in ("modelo_final.joblib", backup_name)} == others
    assert predict.load_model(models_dir / "modelo_final.joblib")["n_matches"] == N_DEV + N_TEST


def test_train_after_a_test_result_needs_no_force_and_prints_no_warning(world, models_dir, quick_validation):
    run_evaluate(world / "B", models_dir)
    tested = predict.load_model(models_dir / "modelo_final.joblib")
    record_second(models_dir, tested)
    lines = []
    result = predict.train_command(world / "B", models_dir, out=lines.append)
    assert not any("ADVERTENCIA" in l for l in lines)
    assert result["modelo_probado"]["trained_at"] == tested["trained_at"]
    assert result["modelo_probado"]["selectiva"] == SELECTIVE
    assert {k: result["modelo_probado"][k] for k in ("fecha", "n", "accuracy", "auc", "log_loss")} == {
        "fecha": "2026-11-11", "n": 1264, "accuracy": .5704, "auc": .5891, "log_loss": .68}
    assert result["modelo_probado"]["n_matches"] == tested["n_matches"]
    assert predict.load_model(models_dir / "modelo_final.joblib")["modelo_probado"] == result["modelo_probado"]


def test_weak_signal_warning_has_no_hardcoded_numbers(world, models_dir, bundle):
    assert "Todavía no hay una prueba independiente" in predict._weak_signal_warning(bundle, models_dir)
    assert f"{bundle['auc_temporal']:.3f}" in predict._weak_signal_warning(bundle, models_dir)
    assert "optimista" in predict._weak_signal_warning(bundle, models_dir)
    run_evaluate(world / "B", models_dir)
    text = predict._weak_signal_warning(bundle, models_dir)
    assert "Última prueba independiente" in text and f"n = {N_TEST}" in text


def test_import_and_help_work_without_riot_api_key():
    import os
    import subprocess
    import sys
    env = {**os.environ, "RIOT_API_KEY": ""}
    result = subprocess.run([sys.executable, "-m", "src.models.predict", "--help"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_safe_error_hides_the_url_and_puuid():
    response = requests.Response()
    response.status_code = 403
    error = requests.HTTPError("403 Client Error: Forbidden for url: https://x/active-games/by-summoner/SECRETPUUID",
                               response=response)
    assert "SECRETPUUID" not in predict._safe_error(error) and "403" in predict._safe_error(error)
    from src.riot_client.client import RiotUnavailableError
    assert "SECRETPUUID" not in predict._safe_error(RiotUnavailableError("Gave up on https://x/SECRETPUUID"))


class PredictionOnly:
    def predict_proba(self, X):
        p = X['p'].to_numpy()
        return np.column_stack([1-p, p])

    def fit(self, *args, **kwargs):
        raise AssertionError('Evaluation must never fit')


@pytest.fixture
def second_world(monkeypatch):
    cutoff = predict.SECOND_CUTOFF_TS
    bundle = dict(version=2, roles='inferred', model=PredictionOnly(), features=['p'],
                  match_ids=['fit'], development_ids=['fit', 'dev'], last_match_ts=cutoff,
                  blue_win_rate=.5, pool=pd.DataFrame(), champion_positions={},
                  auc_temporal=.55, auc_ic95=(.51, .59))
    table = pd.DataFrame(dict(match_id=['fit', 'dev', 'old', 'edge', 'new0', 'new1'],
                              game_start_ts=[cutoff+1, cutoff+1, cutoff-1, cutoff, cutoff+1, cutoff+2],
                              blue_win=[0, 1, 0, 1, 0, 1], seed_tier=['GOLD']*6,
                              p=[.1, .9, .1, .9, .1, .9]))
    calibration = train.calibration_summary
    monkeypatch.setattr(train, 'calibration_summary', lambda y, p: calibration(y, p, n_boot=10))
    return bundle, copy.deepcopy(bundle), table


def test_evaluate_final_filters_and_never_fits(second_world):
    final, candidate, table = second_world
    before = joblib.hash((final, candidate))
    result = predict.evaluate_final(final, candidate, table, table.iloc[::-1])
    assert result['n'] == 2
    assert [result[k] for k in ('excluidas_ajuste', 'excluidas_desarrollo', 'excluidas_por_fecha')] == [1, 1, 2]
    assert result['mejor'] and 'es mejor' in result['veredicto']
    assert result['calibracion']['final']['brier']['estimate'] == pytest.approx(.01)
    assert joblib.hash((final, candidate)) == before
    # The bundle cutoff can be later than the preregistered minimum.
    final['last_match_ts'] += 1
    assert predict.evaluate_final(final, candidate, table, table)['n'] == 1


def test_evaluate_final_one_class_and_negative_verdict(second_world):
    final, candidate, table = second_world
    table['blue_win'] = 1
    table['p'] = .1
    r = predict.evaluate_final(final, candidate, table, table)
    assert r['modelos']['final']['auc'] is None
    assert r['calibracion']['final']['slope']['estimate'] is None
    assert not r['mejor'] and 'No se demuestra' in r['veredicto']


@pytest.mark.parametrize('failure', ['ids', 'outcome', 'features', 'duplicate'])
def test_evaluate_final_alignment_guards(second_world, failure):
    final, candidate, table = second_world
    other = table.copy()
    if failure == 'ids': other.loc[0, 'match_id'] = 'different'
    elif failure == 'outcome': other.loc[0, 'blue_win'] = 1
    elif failure == 'features': other = other.drop(columns='p')
    else: other = pd.concat([other, other.iloc[:1]])
    with pytest.raises(ValueError):
        predict.evaluate_final(final, candidate, table, other)


def test_second_command_recording_and_empty(second_world, tmp_path, monkeypatch):
    final, candidate, table = second_world
    monkeypatch.setattr(predict, 'load_model', lambda path: final if path.name == 'modelo_final.joblib' else candidate)
    pools = []
    def table_for(export, pool, shares):
        pools.append((pool, shares))
        return table
    monkeypatch.setattr(predict, '_table_for', table_for)
    output = []
    predict.evaluate_final_command(tmp_path, tmp_path, output.append)
    predict.evaluate_final_command(tmp_path, tmp_path, output.append)
    assert len(pools) == 4 and pools[0][0] is final['pool'] and pools[1][0] is candidate['pool']
    assert 'ejecuciones anteriores 1' in '\n'.join(output)
    assert len(predict._read_test_runs(tmp_path, predict.SECOND_RESULT_NAME)) == 2
    assert {p.name for p in tmp_path.iterdir()} == {'segunda_prueba.json'}
    assert 'Segunda prueba independiente de este modelo' in predict._weak_signal_warning(final, tmp_path)
    assert 'Constante menos modelo final, log loss' in '\n'.join(output) and "{'" not in '\n'.join(output)
    refit = {**final, 'trained_at': 'otro'}  # a later refit is a different model: its warning must not claim this test
    assert 'Segunda prueba independiente' not in predict._weak_signal_warning(refit, tmp_path)
    before = (tmp_path / predict.SECOND_RESULT_NAME).read_bytes()
    table.loc[:, 'game_start_ts'] = predict.SECOND_CUTOFF_TS
    predict.evaluate_final_command(tmp_path, tmp_path, output.append)
    assert (tmp_path / predict.SECOND_RESULT_NAME).read_bytes() == before
    assert 'No se escribió nada' in output[-1]


def test_evaluate_final_cli(monkeypatch):
    import sys
    called = []
    monkeypatch.setattr(sys, 'argv', ['predict', '--evaluate-final'])
    monkeypatch.setattr(predict, 'evaluate_final_command', lambda: called.append(True))
    predict.main()
    assert called == [True]


def test_evaluate_final_rejects_missing_date(second_world):
    final, candidate, table = second_world
    table.loc[4, 'game_start_ts'] = np.nan
    r = predict.evaluate_final(final, candidate, table, table)
    assert r['n'] == 1 and r['excluidas_por_fecha'] == 3


@pytest.mark.parametrize('other', ['--train', '--freeze', '--evaluate-new', '--live', '--score-log'])
def test_second_cli_cannot_trigger_other_commands(monkeypatch, other):
    import sys
    arguments = ['predict', '--evaluate-final', other]
    if other == '--live': arguments.append('Synthetic#TEST')
    monkeypatch.setattr(sys, 'argv', arguments)
    with pytest.raises(SystemExit) as error:
        predict.main()
    assert error.value.code == 2


def test_second_command_zero_matches_creates_no_file(second_world, tmp_path, monkeypatch):
    final, candidate, table = second_world
    monkeypatch.setattr(predict, 'load_model', lambda path: final if path.name == 'modelo_final.joblib' else candidate)
    monkeypatch.setattr(predict, '_table_for', lambda *args: table.iloc[:4])
    result = predict.evaluate_final_command(tmp_path, tmp_path, out=lambda line: None)
    assert result['n'] == 0
    assert not list(tmp_path.iterdir())


def test_second_record_leaves_first_test_unchanged(second_world, tmp_path, monkeypatch):
    final, candidate, table = second_world
    sentinel = b'{"runs": [{"original": true}]}\n'
    first = tmp_path / predict.TEST_RESULT_NAME
    first.write_bytes(sentinel)
    r = predict.evaluate_final(final, candidate, table, table)
    predict.record_test_run(r, tmp_path, predict.SECOND_RESULT_NAME)
    assert first.read_bytes() == sentinel
    saved = json.loads((tmp_path / predict.SECOND_RESULT_NAME).read_text())['runs'][0]
    assert saved['calibracion'] == json.loads(json.dumps(r['calibracion']))
    assert saved['selectiva'] == r['selectiva']


# --- Confidence, safe refit, context and watch (only synthetic endpoints) -------------------

SELECTIVE = [{"threshold": t, "n": n, "accuracy": a}
             for t, n, a in ((.55, 699, .5951), (.60, 275, .6582), (.65, 84, .6786))]


def record_second(models_dir, bundle):
    record = {"modelo_trained_at": bundle["trained_at"], "n": 1264, "fecha": "2026-11-11",
              "modelos": {"final": {"accuracy": .5704, "auc": .5891, "log_loss": .68}},
              "selectiva": {"final": SELECTIVE}}
    (models_dir / predict.SECOND_RESULT_NAME).write_text(json.dumps({"runs": [record]}))


@pytest.mark.parametrize("p,level", [(.5, "pareja"), (.521, "pareja"), (.55, "baja"), (.45, "baja"),
                                    (.60, "media"), (.40, "media"), (.65, "alta"), (.35, "alta")])
@pytest.mark.parametrize("selective", [None, SELECTIVE])
def test_confidence_levels_and_exact_edges(p, level, selective):
    result = predict.confidence_summary(p, selective)
    assert result["nivel"] == level
    assert result["confianza"] == max(p, 1 - p)
    assert result["abstencion"] == (level == "pareja")
    assert result["favorito"] == (None if level == "pareja" else "azul" if p > .5 else "rojo")
    if selective and level != "pareja":
        row = max((r for r in selective if max(p, 1-p) >= r["threshold"]), key=lambda r: r["threshold"])
        assert result["acierto_historico"] == row["accuracy"]
        assert result["n_historico"] == row["n"]
    else:
        assert "acierto_historico" not in result


def test_confidence_source_and_warning(tmp_path, bundle):
    assert "segunda prueba" not in predict._confidence_line(.623, bundle, tmp_path)
    record_second(tmp_path, bundle)
    text = predict._confidence_line(.623, bundle, tmp_path)
    assert "62,3 %" in text and "65,8 % de 275" in text
    assert "Confianza media" in text
    refit = {**bundle, "modelo_probado": {"trained_at": "anterior", "n_matches": 2973,
              "fecha": "2026-11-11", "n": 1264, "accuracy": .5704, "auc": .5891,
              "log_loss": .68, "selectiva": SELECTIVE}}
    assert "de la versión probada del modelo" in predict._confidence_line(.623, refit, tmp_path)
    warning = predict._weak_signal_warning(refit, tmp_path)
    assert "2973 partidas" in warning and "0.5891" in warning and "no tiene prueba independiente propia" in warning
    assert "sin predicción" in predict._confidence_line(.521, refit, tmp_path)
    # A historical run of another bundle cannot supply confidence for this one.
    assert not predict._selectivity({**bundle, "trained_at": "otro"}, tmp_path)


def test_train_needs_matching_second_test(world, models_dir):
    run_evaluate(world / "B", models_dir)
    before = (models_dir / "modelo_final.joblib").read_bytes()
    record_second(models_dir, {"trained_at": "otro"})
    with pytest.raises(predict.PredictError, match="segunda prueba"):
        predict.train_command(world / "B", models_dir, out=lambda _: None)
    assert (models_dir / "modelo_final.joblib").read_bytes() == before


def test_train_does_not_overwrite_different_backup(world, models_dir):
    before = (models_dir / "modelo_final.joblib").read_bytes()
    current = predict.load_model(models_dir / "modelo_final.joblib")
    path = models_dir / f"modelo_final_{current['trained_at'][:10]}.joblib"
    path.write_bytes(b"otra copia")
    with pytest.raises(predict.PredictError, match="otro contenido"):
        predict.train_command(world / "B", models_dir, force=True, out=lambda _: None)
    assert path.read_bytes() == b"otra copia"
    assert (models_dir / "modelo_final.joblib").read_bytes() == before


def test_backup_reuses_identical_copy(tmp_path):
    path = tmp_path / "modelo_final.joblib"
    path.write_bytes(b"modelo")
    bundle = {"trained_at": "2026-11-09T18:09:41+00:00"}
    backup = predict._backup_model(path, bundle)
    assert predict._backup_model(path, bundle) == backup
    assert backup.read_bytes() == path.read_bytes()


def test_context_columns_private_missing_and_independent(live_env, bundle, tmp_path, monkeypatch):
    from src.riot_client import endpoints
    live_env["game"] = live_game(hidden=True, blocked=True)
    calls = []
    def entry(key):
        calls.append(key)
        if key.endswith("_2"):
            raise requests.ConnectionError("https://secret/" + key)
        return {"tier": "GOLD", "rank": "II", "leaguePoints": 50, "wins": 9, "losses": 8}
    monkeypatch.setattr(endpoints, "get_solo_queue_entry", entry)
    monkeypatch.setattr(endpoints, "get_champion_mastery", lambda *a: {"championPoints": 123})
    run_live(bundle, tmp_path)
    path = tmp_path / "live_context.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == predict.CONTEXT_COLUMNS
        rows = list(reader)
    assert len(rows) == 10 and len(calls) == 8
    assert rows[0]["tier"] == rows[1]["tier"] == rows[2]["tier"] == ""
    assert rows[2]["maestria_puntos"] == "123"
    assert rows[3]["tier"] == "GOLD" and rows[3]["segundos_de_partida"] == "0"
    assert SECRET not in path.read_text() and "Name#TAG" not in path.read_text()
    assert "puuid" not in path.read_text()
    run_live(bundle, tmp_path)
    assert len(path.read_text().splitlines()) == 11


def test_context_mastery_failure_and_elapsed(live_env, bundle, tmp_path, monkeypatch):
    from src.riot_client import endpoints
    monkeypatch.setattr(predict.time, "time", lambda: 1700000000)
    live_env["game"] = live_game(start=1699999980000)
    def fail(*a):
        raise requests.Timeout("private URL")
    monkeypatch.setattr(endpoints, "get_champion_mastery", fail)
    run_live(bundle, tmp_path)
    with (tmp_path / "live_context.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert all(r["maestria_puntos"] == "" and r["segundos_de_partida"] == "20" for r in rows)


def test_watch_skips_and_continues_until_max(live_env, bundle, tmp_path, monkeypatch):
    from src.riot_client import endpoints
    seeds = ["ab--cd", "none", "queue", "incomplete", "duplicate", "network", "valid", "never"]
    monkeypatch.setattr(predict, "_seed_players", lambda path: seeds)
    live_env["game"]["gameId"] = 99
    run_live(bundle, tmp_path)
    asked = []
    def active(key):
        asked.append(key)
        if key == "none":
            return None
        if key == "network":
            raise requests.ConnectionError("https://private/SECRETPUUID")
        game = live_game(queue=450 if key == "queue" else 420)
        if key == "incomplete":
            game["participants"].pop()
        if key == "duplicate":
            game["gameId"] = 99
        return game
    monkeypatch.setattr(endpoints, "get_active_game", active)
    lines = []
    counts = predict.watch(bundle, max_games=1, log_path=tmp_path / "log.csv", out=lines.append, names={})
    assert counts == dict(revisados=7, predichas=1, bloqueados=1, sin_partida=1, otra_cola=1,
                          incompletas=1, registradas=1, errores=1, errores_prediccion=0)
    assert "never" not in asked
    text = "\n".join(lines)
    assert "Equipo azul" in text and ("Confianza" in text or "Partida pareja" in text)
    assert SECRET not in text and "Name#TAG" not in text and "https://" not in text


def test_watch_respects_deadline_and_ctrl_c(bundle, tmp_path, monkeypatch):
    from src.riot_client import endpoints
    monkeypatch.setattr(predict, "_seed_players", lambda path: ["private"])
    ticks = iter([0, 0, 61, 61, 61])
    monkeypatch.setattr(predict.time, "monotonic", lambda: next(ticks))
    r = predict.watch(bundle, watch_minutes=1, log_path=tmp_path / "log.csv", out=lambda _: None)
    assert r["revisados"] == 0
    monkeypatch.setattr(predict.time, "monotonic", lambda: 0)
    def interrupt(key):
        raise KeyboardInterrupt
    monkeypatch.setattr(endpoints, "get_active_game", interrupt)
    lines = []
    r = predict.watch(bundle, log_path=tmp_path / "log.csv", out=lines.append)
    assert r["predichas"] == 0 and "interrumpida" in lines[0]


def test_seed_database_read_only(tmp_path):
    import sqlite3
    path = tmp_path / "seeds.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE players (puuid TEXT)")
    conn.execute("INSERT INTO players VALUES ('private')")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    assert predict._seed_players(path) == ["private"]
    assert path.read_bytes() == before
    with pytest.raises(sqlite3.OperationalError):
        predict._seed_players(tmp_path / "missing.sqlite")
    assert not (tmp_path / "missing.sqlite").exists()


def test_score_log_selectivity_loss_small_sample(tmp_path, monkeypatch, bundle, capsys):
    path = tmp_path / "log.csv"
    write_log(path, [.52, .6, .7, .3])
    r = predict.score_log(path, lambda mid: fake_match(blue_win=int(mid.split("_")[1]) != 3))
    assert r["log_loss"] == pytest.approx(-np.log([.52, .6, .7, .7]).mean())
    assert r["abstenciones"] == 1
    assert [row["n"] for row in r["selectiva"]] == [3, 3, 2]
    assert all(row["lo"] < row["accuracy"] <= row["hi"] for row in r["selectiva"])
    monkeypatch.setattr(predict, "_load_checked", lambda: bundle)
    monkeypatch.setattr(predict, "score_log", lambda **kw: r)
    predict._score_log_command()
    text = capsys.readouterr().out
    assert "demasiado pequeña" in text and "Wilson" in text and "pérdida logarítmica" in text



def test_confidence_sparse_history_uses_actual_threshold(tmp_path, bundle):
    record_second(tmp_path, bundle)
    tested = {"selectiva": SELECTIVE[:1]}
    text = predict._confidence_line(.7, {**bundle, "modelo_probado": tested}, tmp_path)
    assert "Confianza alta" in text and "55 % o más" in text and "699 partidas" in text


def test_score_log_empty_selection_and_other_models(tmp_path, bundle):
    path = tmp_path / "log.csv"
    write_log(path, [.52, .56])
    mine = {**bundle, "trained_at": "t"}
    predict.log_prediction(99, "LA1", .8, bundle, path)
    r = predict.score_log(path, lambda mid: fake_match(blue_win=1), bundle=mine)
    assert r["n"] == 2 and r["otro_modelo"] == 1 and r["logged"] == 3
    assert r["selectiva"][1]["n"] == 0 and r["selectiva"][1]["accuracy"] is None
    assert r["selectiva"][2]["lo"] is None


@pytest.mark.parametrize("args", [["--watch", "--max-games", "0"], ["--watch", "--watch-minutes", "-1"],
                                  ["--watch", "--live", "Private#TAG"], ["--watch", "--evaluate-final"]])
def test_watch_cli_rejects_invalid_arguments(monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["predict", *args])
    with pytest.raises(SystemExit) as error:
        predict.main()
    assert error.value.code == 2



def test_live_errors_do_not_echo_player(live_env, bundle, tmp_path, monkeypatch):
    from src.riot_client import endpoints
    def missing(*args):
        raise http_404()
    monkeypatch.setattr(endpoints, "get_puuid", missing)
    with pytest.raises(predict.LiveError) as error:
        run_live(bundle, tmp_path)
    assert "Name#TAG" not in str(error.value) and "http" not in str(error.value)


def test_prediction_commands_include_confidence(bundle, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(predict, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(predict, "_load_checked", lambda: bundle)
    monkeypatch.setattr(predict, "predict_match", lambda *a: {"prob_blue": .623, "prob_red": .377,
        "blue_win": 1, "en_entrenamiento": False, "jugadores_sin_historial": 0})
    predict._predict_command("LA1_1")
    assert "Confianza media (62,3 %)" in capsys.readouterr().out
    monkeypatch.setattr(predict, "predict_live", lambda *a: {"prob_blue": .521, "prob_red": .479,
        "teams": {}, "jugadores_sin_historial": 0, "logged": True})
    predict._live_command("Private#TAG")
    text = capsys.readouterr().out
    assert "Partida pareja: sin predicción" in text and "Private#TAG" not in text



def test_confidence_highest_threshold_without_outcomes(tmp_path):
    rows = [*SELECTIVE[:2], {"threshold": .65, "n": 0, "accuracy": None}]
    summary = predict.confidence_summary(.7, rows)
    assert summary["n_historico"] == 0 and summary["acierto_historico"] is None
    text = predict._confidence_line(.7, {"modelo_probado": {"selectiva": rows}}, tmp_path)
    assert "Confianza alta" in text and "acertó" not in text


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_context_publish_failure_is_retryable_and_atomic(live_env, bundle, tmp_path, monkeypatch, failure):
    from pathlib import Path
    # Keep another game's context intact when publishing the next game fails.
    live_env["game"]["gameId"] = 998
    run_live(bundle, tmp_path)
    context = tmp_path / "live_context.csv"
    before = context.read_bytes()
    live_env["game"]["gameId"] = 999
    with monkeypatch.context() as patch:
        def fail(*args):
            raise failure("fallo simulado")
        patch.setattr(Path, "replace", fail)
        with pytest.raises(failure):
            run_live(bundle, tmp_path)
    assert context.read_bytes() == before
    assert predict._logged_games(tmp_path / "log.csv") == {("LA1", "998")}
    assert {p.name for p in tmp_path.iterdir()} == {"live_context.csv", "log.csv"}
    result, _ = run_live(bundle, tmp_path)
    assert result["logged"]
    assert predict._context_games(context) == {("LA1", "998"), ("LA1", "999")}
    assert len(list(csv.DictReader(context.open()))) == 20


def test_interruption_between_context_and_log_does_not_duplicate_context(live_env, bundle, tmp_path, monkeypatch):
    from src.riot_client import endpoints
    calls = []
    monkeypatch.setattr(endpoints, "get_solo_queue_entry", lambda key: calls.append(1))
    with monkeypatch.context() as patch:
        def interrupt(*args):
            raise KeyboardInterrupt
        patch.setattr(predict, "log_prediction", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_live(bundle, tmp_path)
    assert not (tmp_path / "log.csv").exists()
    assert predict._context_games(tmp_path / "live_context.csv") == {("LA1", "999")}
    result, _ = run_live(bundle, tmp_path)
    assert result["logged"] and len(calls) == 10
    assert len(list(csv.DictReader((tmp_path / "live_context.csv").open()))) == 10


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("mode", ["live", "watch"])
def test_existing_log_with_missing_or_partial_context_is_repaired(live_env, bundle, tmp_path, monkeypatch,
                                                                  partial, mode):
    context = tmp_path / "live_context.csv"
    if partial:
        run_live(bundle, tmp_path)
        context.write_text("\n".join(context.read_text().splitlines()[:4]) + "\n")
    else:
        predict.log_prediction(999, "LA1", .6, bundle, tmp_path / "log.csv")
    original_log = (tmp_path / "log.csv").read_bytes()
    if mode == "live":
        result, _ = run_live(bundle, tmp_path)
        assert not result["logged"]
    else:
        from src.riot_client import endpoints
        monkeypatch.setattr(predict, "_seed_players", lambda _: ["first", "next"])
        def active(seed):
            if seed == "next":
                raise KeyboardInterrupt
            return live_env["game"]
        monkeypatch.setattr(endpoints, "get_active_game", active)
        counts = predict.watch(bundle, log_path=tmp_path / "log.csv", out=lambda _: None, names={})
        assert counts["registradas"] == 1 and counts["predichas"] == 0
    assert predict._context_games(context) == {("LA1", "999")}
    assert len(list(csv.DictReader(context.open()))) == 10
    assert (tmp_path / "log.csv").read_bytes() == original_log


@pytest.mark.parametrize("failure", [ValueError, predict.LiveError, RuntimeError])
def test_watch_continues_after_prediction_failure(live_env, bundle, tmp_path, monkeypatch, failure):
    original = predict.score_game
    calls = []
    def score(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise failure("https://private/" + SECRET + " Name#TAG")
        return original(*args, **kwargs)
    monkeypatch.setattr(predict, "score_game", score)
    monkeypatch.setattr(predict, "_seed_players", lambda _: ["first", "next", "never"])
    lines = []
    counts = predict.watch(bundle, max_games=1, log_path=tmp_path / "log.csv", out=lines.append, names={})
    assert counts["revisados"] == 2 and counts["predichas"] == 1
    assert counts["errores_prediccion"] == 1 and counts["errores"] == 0
    assert lines[-1].startswith("Resumen de vigilancia:")
    text = "\n".join(lines)
    assert SECRET not in text and "Name#TAG" not in text and "https://" not in text


@pytest.mark.parametrize("stage", ["active", "history", "rank", "mastery"])
def test_watch_deadline_during_download_stops_without_incomplete_log(live_env, bundle, tmp_path, monkeypatch, stage):
    from src.agents import history_builder
    from src.riot_client import client, endpoints
    now = [0.0]
    calls = []
    monkeypatch.setattr(predict.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(predict, "_seed_players", lambda _: ["first", "never"])
    def expire(*args, **kwargs):
        calls.append(1)
        now[0] = 61.0
        return live_env["game"] if stage == "active" else ([] if stage == "history" else {})
    owner, attr = {
        "active": (endpoints, "get_active_game"),
        "history": (history_builder, "player_history"),
        "rank": (endpoints, "get_solo_queue_entry"),
        "mastery": (endpoints, "get_champion_mastery"),
    }[stage]
    monkeypatch.setattr(owner, attr, expire)
    lines = []
    counts = predict.watch(bundle, watch_minutes=1, log_path=tmp_path / "log.csv", out=lines.append, names={})
    assert counts["revisados"] == 1 and counts["predichas"] == 0
    assert counts["errores"] == counts["errores_prediccion"] == 0
    assert len(calls) == 1 and not (tmp_path / "log.csv").exists()
    assert not (tmp_path / "live_context.csv").exists()
    assert lines[-1].startswith("Resumen de vigilancia:")
    assert client.check_deadline() is None  # The next --live has no watch deadline.


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("stage", ["active", "rank", "mastery"])
def test_watch_stops_when_riot_rejects_the_key(live_env, bundle, tmp_path, monkeypatch, status, stage):
    from src.riot_client import endpoints
    monkeypatch.setattr(predict, "_seed_players", lambda path: ["first", "second"])
    asked = []
    calls = []
    def query(current, result):
        calls.append(current)
        if current == stage:
            response = requests.Response()
            response.status_code = status
            raise requests.HTTPError("https://private/SECRETPUUID", response=response)
        return result
    def active(key):
        asked.append(key)
        return query("active", live_env["game"])
    monkeypatch.setattr(endpoints, "get_active_game", active)
    monkeypatch.setattr(endpoints, "get_solo_queue_entry", lambda key: query("rank", None))
    monkeypatch.setattr(endpoints, "get_champion_mastery", lambda *args: query("mastery", None))
    lines = []
    counts = predict.watch(bundle, max_games=1, log_path=tmp_path / "log.csv", out=lines.append, names={})
    assert asked == ["first"] and counts["errores"] == 0 and counts["predichas"] == 0
    assert calls == ["active", "rank", "mastery"][:["active", "rank", "mastery"].index(stage) + 1]
    assert not (tmp_path / "log.csv").exists()
    assert not (tmp_path / "live_context.csv").exists()
    text = "\n".join(lines)
    assert "rechazó la clave" in text and "Resumen de vigilancia" in text and "https://" not in text
