"""Tests for src/features/champion_pool.py (pure). Run: pytest tests/test_champion_pool.py -v"""
import pandas as pd
import pytest

from src.features import champion_pool as cp


def history_rows():
    def row(match, player, ts, champion, position, win, remake=0):
        return {"history_match_id": match, "player_id": player, "game_start_ts": ts, "champion_id": champion,
                "team_position": position, "win": win, "remake": remake, "target_match_id": "T"}
    return pd.DataFrame([
        row("G1", "p1", 100, 7, "TOP", 1), row("G1", "p1", 100, 7, "TOP", 1),   # same game seen twice: once
        row("G2", "p2", 200, 7, "TOP", 0), row("G3", "p3", 300, 7, "MIDDLE", 1),
        row("G4", "p4", 400, 7, "TOP", 1, remake=1),                            # remake: dropped
        row("G5", "p5", 500, 8, "", 1),                                         # no position: dropped
        row("G6", "p6", 600, 8, "TOP", 0)])


def test_build_pool_dedupes_drops_remakes_and_blank_positions_and_has_no_player_column():
    pool = cp.build_pool(history_rows())
    assert list(pool.columns) == cp.POOL_COLUMNS
    assert list(pool["game_start_ts"]) == [100, 200, 300, 600]


def test_shares_per_champion():
    shares = cp.champion_position_shares(cp.build_pool(history_rows()))
    assert shares[7] == {"TOP": pytest.approx(2 / 3), "MIDDLE": pytest.approx(1 / 3)}
    assert shares[8] == {"TOP": 1.0}


def test_rate_with_shrinkage_by_hand():
    look = cp.ChampionPool(cp.build_pool(history_rows()))
    t = 301  # rows at 100, 200, 300 are before t: overall 2 wins in 3, champion 7 the same
    overall = 2 / 3
    champion = (2 + 20 * overall) / (3 + 20)
    pair = (1 + 20 * champion) / (2 + 20)  # TOP: one win in two games
    assert look.overall_rate(t) == pytest.approx(overall)
    assert look.champion_rate(7, t) == pytest.approx(champion)
    assert look.champion_position_rate(7, "TOP", t) == pytest.approx(pair)
    assert look.champion_position_rate(999, "TOP", t) == pytest.approx(overall)  # unknown champion: the prior
    assert cp.ChampionPool(cp.build_pool(history_rows())).overall_rate(0) == 0.5


def test_rows_at_or_after_t_never_change_the_rate():
    pool = cp.build_pool(history_rows())
    extra = pd.DataFrame([{"game_start_ts": 301, "champion_id": 7, "team_position": "TOP", "win": 0},
                          {"game_start_ts": 9999, "champion_id": 7, "team_position": "TOP", "win": 0}])
    before = cp.ChampionPool(pool).champion_position_rate(7, "TOP", 301)
    after = cp.ChampionPool(pd.concat([pool, extra])).champion_position_rate(7, "TOP", 301)
    assert after == before  # the row AT t is excluded too
    assert cp.ChampionPool(pd.concat([pool, extra])).champion_position_rate(7, "TOP", 302) != before
