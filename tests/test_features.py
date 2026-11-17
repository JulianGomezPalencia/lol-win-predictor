"""
Tests for src/features/build.py, on tiny hand-made tables (no real data needed).

Run with:  pytest tests/test_features.py -v
"""
import math

import pandas as pd
import pytest

from src.features import build
from src.features.build import BLUE, RED, POSITIONS

START = 10 * 86_400_000  # target match starts on "day 10" (milliseconds)
DAY = 86_400_000


def make_matches():
    return pd.DataFrame([{"match_id": "M1", "status": "history_done", "seed_tier": "GOLD",
                          "game_start_ts": START, "patch": "16.20", "blue_win": 1}])


def make_participants():
    rows = []
    for team in (BLUE, RED):
        for pos in POSITIONS:
            rows.append({"match_id": "M1", "player_id": f"{team}_{pos}", "team_id": team,
                         "team_position": pos, "champion_id": 1, "champion_name": "A", "win": int(team == BLUE)})
    return pd.DataFrame(rows)


def game(player, rank, win, position="TOP", champion=1, days_before=1, remake=0, kills=2, deaths=1, assists=2):
    return {"target_match_id": "M1", "player_id": player, "history_rank": rank, "history_match_id": f"H{player}{rank}",
            "game_start_ts": START - days_before * DAY, "game_duration": 1800, "remake": remake, "win": win,
            "champion_id": champion, "champion_name": "A", "team_position": position,
            "kills": kills, "deaths": deaths, "assists": assists}


def make_history(extra=()):
    """Every player has two previous TOP games with champion 1, one win then one loss."""
    rows = []
    for team in (BLUE, RED):
        for pos in POSITIONS:
            player = f"{team}_{pos}"
            rows += [game(player, 0, 1, days_before=1), game(player, 1, 0, days_before=2)]
    return pd.DataFrame(rows + list(extra))


@pytest.mark.parametrize("wins, expected", [
    ([1, 1, 0, 1], 2), ([0, 0, 0], -3), ([1], 1), ([], 0), ([0, 1, 1], -1)])
def test_racha_actual(wins, expected):
    assert build.racha_actual(wins) == expected


def player_row(feats, player):
    return feats.set_index("player_id").loc[player]


def test_player_features_basic_values():
    feats = build.player_features(make_history(), make_participants(), make_matches())
    top = player_row(feats, f"{BLUE}_TOP")
    assert top["partidas_previas"] == 2
    assert top["winrate_10"] == 0.5
    assert top["racha"] == 1                      # newest game was a win
    assert top["en_linea_habitual"] == 1.0        # usual lane TOP, plays TOP
    assert top["partidas_con_campeon"] == 2
    assert top["dias_desde_ultima"] == pytest.approx(1.0)
    jungle = player_row(feats, f"{BLUE}_JUNGLE")
    assert jungle["en_linea_habitual"] == 0.0     # usual lane TOP, plays JUNGLE


def test_remakes_are_ignored():
    remake = game(f"{BLUE}_TOP", 2, 1, days_before=3, remake=1)
    feats = build.player_features(make_history([remake]), make_participants(), make_matches())
    assert player_row(feats, f"{BLUE}_TOP")["partidas_previas"] == 2


def test_games_at_or_after_the_target_are_dropped():
    """Leakage guard: a 'history' game that starts with or after the match must not count."""
    future = game(f"{BLUE}_TOP", 2, 1, days_before=0)
    feats = build.player_features(make_history([future]), make_participants(), make_matches())
    assert player_row(feats, f"{BLUE}_TOP")["partidas_previas"] == 2


def test_player_without_history_gets_nan_and_flag():
    history = make_history()
    history = history[history["player_id"] != f"{RED}_MIDDLE"]
    feats = build.player_features(history, make_participants(), make_matches())
    row = player_row(feats, f"{RED}_MIDDLE")
    assert row["sin_historial"] == 1 and row["partidas_previas"] == 0
    assert math.isnan(row["winrate_10"]) and math.isnan(row["en_linea_habitual"])


def test_lane_differences_subtract_the_right_opponent():
    history = make_history()
    # Blue MIDDLE won both previous games; red MIDDLE lost both (everyone else: 1 win, 1 loss).
    history.loc[history["player_id"] == f"{BLUE}_MIDDLE", "win"] = 1
    history.loc[history["player_id"] == f"{RED}_MIDDLE", "win"] = 0
    table = build.match_features(build.player_features(history, make_participants(), make_matches()),
                                 make_matches())
    row = table.iloc[0]
    assert row["middle_dif_winrate_10"] == pytest.approx(1.0)
    assert row["top_dif_winrate_10"] == pytest.approx(0.0)
    assert row["dif_winrate_10"] == pytest.approx(0.2)  # team average: (+0.5 - -0.5) / 5
    assert row["blue_win"] == 1


def test_off_role_count_difference():
    # Everyone's usual lane is TOP, so 4 players per team are off-role: difference 0.
    table = build.match_features(build.player_features(make_history(), make_participants(), make_matches()),
                                 make_matches())
    assert table.iloc[0]["dif_fuera_de_posicion"] == 0


def test_only_matches_with_history_are_built():
    matches = pd.concat([make_matches(), pd.DataFrame([{
        "match_id": "M2", "status": "fetched", "seed_tier": "GOLD", "game_start_ts": START,
        "patch": "16.20", "blue_win": 0}])])
    table = build.match_features(build.player_features(make_history(), make_participants(), matches), matches)
    assert table["match_id"].tolist() == ["M1"]


def test_feature_columns_are_only_differences():
    table = build.match_features(build.player_features(make_history(), make_participants(), make_matches()),
                                 make_matches())
    cols = build.feature_columns(table)
    assert "blue_win" not in cols and "game_start_ts" not in cols and "seed_tier" not in cols
    assert "dif_winrate_10" in cols and "utility_dif_kda_medio" in cols


def test_usual_lane_tie_goes_to_the_most_recent_lane():
    # Newest first: UTILITY, TOP, UTILITY, TOP -> tie 2-2, UTILITY was played most recently.
    assert build._usual_position(pd.Series(["UTILITY", "TOP", "UTILITY", "TOP"])) == "UTILITY"
    assert build._usual_position(pd.Series(["TOP", "", "UTILITY", "UTILITY"])) == "UTILITY"
    assert math.isnan(build._usual_position(pd.Series(["", None])))


# --- inferred positions -------------------------------------------------------------------


def make_spells(smite_player="100_JUNGLE"):
    rows = []
    for team in (BLUE, RED):
        for pos in POSITIONS:
            p = f"{team}_{pos}"
            rows.append({"match_id": "M1", "player_id": p, "spell1_id": 11 if p == smite_player else 4, "spell2_id": 14})
    return pd.DataFrame(rows)


def positional_history(extra=()):
    rows = []
    for team in (BLUE, RED):
        for pos in POSITIONS:
            p = f"{team}_{pos}"
            rows += [game(p, 0, 1, position=pos, days_before=1), game(p, 1, 0, position=pos, days_before=2)]
    return pd.DataFrame(rows + list(extra))


def inferred(participants, history, spells=None):
    return build.inferred_participants(participants, history, make_matches(),
                                       make_spells() if spells is None else spells)


def test_inferred_positions_recover_history_roles():
    result = inferred(make_participants().assign(team_position=""), positional_history())
    assert (result["team_position"].values == [p.split("_")[1] for p in result["player_id"]]).all()


def test_inferred_mode_ignores_riot_team_position():
    history = positional_history()
    base = inferred(make_participants(), history)
    shuffled = make_participants().sample(frac=1, random_state=0).reset_index(drop=True)
    shuffled["team_position"] = shuffled["team_position"].sample(frac=1, random_state=1).to_numpy()
    wrong = make_participants().assign(team_position="TOP")
    pd.testing.assert_series_equal(
        base.set_index("player_id")["team_position"].sort_index(),
        inferred(shuffled, history).set_index("player_id")["team_position"].sort_index())
    feats = lambda parts: build.match_features(build.player_features(history, parts, make_matches()), make_matches())
    pd.testing.assert_frame_equal(feats(base), feats(inferred(shuffled, history)))
    pd.testing.assert_frame_equal(feats(base), feats(inferred(wrong, history)))


def test_history_at_or_after_match_start_does_not_change_inferred_roles():
    base = inferred(make_participants(), positional_history())
    late = [game("100_TOP", 5 + i, 1, position="UTILITY", days_before=-i) for i in range(0, 4)]  # start >= START
    late.append({**game("100_TOP", 9, 1, position="UTILITY", days_before=1), "remake": 1})      # remake
    changed = inferred(make_participants(), positional_history(late))
    assert (changed["team_position"].values == base["team_position"].values).all()


def test_build_inferred_needs_the_spells_csv(tmp_path):
    for name, frame in (("matches", make_matches()), ("participants", make_participants()), ("player_history", make_history())):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    with pytest.raises(FileNotFoundError, match="export_dataset"):
        build.build(tmp_path, inferred_positions=True)
    default = build.build(tmp_path)  # default mode works without the new file and uses Riot's positions
    assert len(default) == 1
    make_spells().to_csv(tmp_path / "participant_spells.csv", index=False)
    inferred_table = build.build(tmp_path, inferred_positions=True)
    assert list(inferred_table.columns) == list(default.columns) and len(inferred_table) == 1


# --- the three variables of the frozen set -----------------------------------------------------

from src.features import champion_pool  # noqa: E402


def make_pool(rows=()):
    base = [{"game_start_ts": START - 5 * DAY, "champion_id": 1, "team_position": "TOP", "win": 1},
            {"game_start_ts": START - 4 * DAY, "champion_id": 1, "team_position": "TOP", "win": 0},
            {"game_start_ts": START - 3 * DAY, "champion_id": 1, "team_position": "MIDDLE", "win": 1}]
    return pd.DataFrame(base + list(rows))


def final_row(history, pool=None, matches=None, participants=None):
    matches = make_matches() if matches is None else matches
    feats = build.player_features(history, make_participants() if participants is None else participants, matches,
                                  make_pool() if pool is None else pool)
    return feats, build.match_features(feats, matches, final_features=True)


def test_default_build_is_unchanged_and_new_columns_need_final_features(tmp_path):
    matches, participants, history = make_matches(), make_participants(), make_history()
    default = build.match_features(build.player_features(history, participants, matches), matches)
    assert not [c for c in default.columns if "kills_medio" in c or "partidas_24h" in c or "campeon_posicion" in c]
    assert len(build.feature_columns(default)) == 29
    _, final = final_row(history)
    assert set(build.FINAL_FEATURES) <= set(final.columns)
    assert set(default.columns) < set(final.columns)
    pd.testing.assert_frame_equal(final[list(default.columns)], default)  # old columns keep their values
    assert build.SENSITIVITY_FEATURES == build.FINAL_FEATURES[:9]


def test_final_features_without_a_pool_is_refused(tmp_path):
    with pytest.raises(ValueError, match="pool"):
        build.build(tmp_path, final_features=True)


def test_kills_mean_and_24h_count_by_hand():
    player = f"{BLUE}_TOP"
    rows = [game(player, 0, 1, kills=4), game(player, 1, 0, kills=2, days_before=3)]
    boundary = [{**game(player, 2, 1, kills=0), "game_start_ts": START - DAY},        # exactly 24 h: not counted
                {**game(player, 3, 1, kills=0), "game_start_ts": START - DAY + 1},    # 1 ms inside: counted
                {**game(player, 4, 1, kills=0), "game_start_ts": START - DAY - 1}]    # 1 ms outside
    feats, _ = final_row(pd.DataFrame(rows + boundary))
    top = player_row(feats, player)
    assert top["kills_medio"] == pytest.approx((4 + 2 + 0 + 0 + 0) / 5)
    assert top["partidas_24h"] == 1
    plain = player_row(final_row(pd.DataFrame(rows))[0], player)
    assert plain["kills_medio"] == 3 and plain["partidas_24h"] == 0  # the 1-day-old game is exactly at the boundary
    nobody = player_row(final_row(pd.DataFrame(rows))[0], f"{RED}_TOP")
    assert math.isnan(nobody["kills_medio"]) and math.isnan(nobody["partidas_24h"]) and nobody["sin_historial"] == 1


def test_champion_position_rate_in_the_table_uses_the_players_inferred_position_and_shrinks():
    feats, _ = final_row(make_history())
    look = champion_pool.ChampionPool(make_pool())
    expected_top = look.champion_position_rate(1, "TOP", START)
    expected_mid = look.champion_position_rate(1, "MIDDLE", START)
    assert player_row(feats, f"{BLUE}_TOP")["winrate_campeon_posicion"] == pytest.approx(expected_top)
    assert player_row(feats, f"{BLUE}_MIDDLE")["winrate_campeon_posicion"] == pytest.approx(expected_mid)
    overall = 2 / 3
    champion = (2 + 20 * overall) / (3 + 20)
    assert expected_top == pytest.approx((1 + 20 * champion) / (2 + 20))


def test_new_variables_ignore_games_and_pool_rows_at_or_after_the_match_start():
    clean_feats, clean = final_row(make_history())
    late_games = [{**game(f"{BLUE}_TOP", 9, 1, kills=50), "game_start_ts": START},
                  {**game(f"{RED}_TOP", 9, 0, kills=50), "game_start_ts": START + DAY}]
    late_pool = [{"game_start_ts": START, "champion_id": 1, "team_position": "TOP", "win": 0},
                 {"game_start_ts": START + DAY, "champion_id": 1, "team_position": "TOP", "win": 0}]
    leaked_feats, leaked = final_row(make_history(late_games), make_pool(late_pool))
    pd.testing.assert_frame_equal(leaked[build.FINAL_FEATURES], clean[build.FINAL_FEATURES])


def test_no_feature_reads_the_result_of_the_target_match():
    _, base = final_row(make_history())
    flipped_matches = make_matches().assign(blue_win=0)
    flipped_players = make_participants().assign(win=lambda d: 1 - d["win"])
    _, other = final_row(make_history(), matches=flipped_matches, participants=flipped_players)
    pd.testing.assert_frame_equal(other.drop(columns=["blue_win"]), base.drop(columns=["blue_win"]))
