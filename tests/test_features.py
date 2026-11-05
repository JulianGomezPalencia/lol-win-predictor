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
