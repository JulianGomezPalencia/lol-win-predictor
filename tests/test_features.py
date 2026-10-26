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
