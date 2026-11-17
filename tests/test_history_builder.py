"""
Tests for Agent 4 (history builder). No real API calls.

Run with:  pytest tests/test_history_builder.py -v
"""
import sqlite3

import pytest
import requests

from src.agents import history_builder
from src.riot_client import endpoints
from src.storage import db

TARGET_START = 1_800_000_000_000  # ms


class FakeHTTPResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def history_match(match_id, puuid, start_ts, win=True, duration=1800, champion_id=7):
    other = [{"puuid": f"other{i}", "win": False, "championId": 1, "championName": "X",
              "teamPosition": "TOP", "kills": 0, "deaths": 0, "assists": 0} for i in range(9)]
    me = {"puuid": puuid, "win": win, "championId": champion_id, "championName": "Leblanc",
          "teamPosition": "MIDDLE", "kills": 5, "deaths": 2, "assists": 7,
          "gameEndedInEarlySurrender": False}
    return {
        "metadata": {"matchId": match_id},
        "info": {"gameStartTimestamp": start_ts, "gameDuration": duration, "participants": other + [me]},
    }


def test_summarize_for_player():
    summary = history_builder.summarize_for_player(history_match("LA1_9", "me", 123), "me")
    assert summary == {
        "history_match_id": "LA1_9", "game_start_ts": 123, "game_duration": 1800, "remake": 0,
        "win": 1, "champion_id": 7, "champion_name": "Leblanc", "team_position": "MIDDLE",
        "kills": 5, "deaths": 2, "assists": 7,
    }


def test_summarize_marks_remakes():
    summary = history_builder.summarize_for_player(history_match("LA1_9", "me", 123, duration=200), "me")
    assert summary["remake"] == 1


def test_summarize_player_not_in_match():
    assert history_builder.summarize_for_player(history_match("LA1_9", "me", 123), "stranger") is None


@pytest.fixture
def fake_riot(monkeypatch):
    calls = {"ids": [], "matches": {}}

    def fake_get_match_ids(puuid, count=5, start_time=None, end_time=None):
        calls["ids"].append({"puuid": puuid, "count": count, "end_time": end_time})
        return ["LA1_OLD2", "LA1_OLD1", "LA1_FUTURE", "LA1_GONE"]

    def fake_get_match(match_id):
        if match_id == "LA1_GONE":
            raise requests.HTTPError("404", response=FakeHTTPResponse(404))
        return calls["matches"][match_id]

    monkeypatch.setattr(endpoints, "get_match_ids", fake_get_match_ids)
    monkeypatch.setattr(endpoints, "get_match", fake_get_match)
    return calls


def test_player_history_asks_riot_for_games_before_the_target(fake_riot):
    fake_riot["matches"] = {
        "LA1_OLD2": history_match("LA1_OLD2", "me", TARGET_START - 5_000_000),
        "LA1_OLD1": history_match("LA1_OLD1", "me", TARGET_START - 9_000_000),
        "LA1_FUTURE": history_match("LA1_FUTURE", "me", TARGET_START + 1),
    }
    rows = history_builder.player_history("me", TARGET_START, depth=10)

    assert fake_riot["ids"] == [{"puuid": "me", "count": 10, "end_time": TARGET_START // 1000 - 1}]
    assert [r["history_match_id"] for r in rows] == ["LA1_OLD2", "LA1_OLD1"], \
        "the game after the target is dropped (leakage) and the missing one is skipped"
    assert [r["history_rank"] for r in rows] == [0, 1]
    assert all(r["game_start_ts"] < TARGET_START for r in rows)


def test_build_histories_saves_rows_and_marks_done(tmp_path, fake_riot):
    fake_riot["matches"] = {
        "LA1_OLD2": history_match("LA1_OLD2", "me", TARGET_START - 5_000_000),
        "LA1_OLD1": history_match("LA1_OLD1", "me", TARGET_START - 9_000_000),
        "LA1_FUTURE": history_match("LA1_FUTURE", "me", TARGET_START + 1),
    }
    path = tmp_path / "test.sqlite"
    conn = db.connect(path)
    db.add_target_match(conn, "LA1_T", "seed", "GOLD")
    db.save_fetched_match(
        conn, "LA1_T", {"game_start_ts": TARGET_START, "patch": "16.8", "blue_win": 1},
        [{"puuid": "me", "team_id": 100, "team_position": "MIDDLE", "champion_id": 7,
          "champion_name": "Leblanc", "win": 1}],
    )

    assert history_builder.build_histories(conn) == 1
    conn.close()

    conn = db.connect(path)  # reopen: proves everything was committed
    assert db.count_target_matches_by_status(conn) == {"history_done": 1}
    assert conn.execute("SELECT COUNT(*) FROM player_history").fetchone()[0] == 2
    assert history_builder.build_histories(conn) == 0, "nothing left to do on a re-run"


def test_build_histories_stores_the_history_of_all_10_players(tmp_path, monkeypatch):
    players = [f"p{i}" for i in range(10)]
    monkeypatch.setattr(endpoints, "get_match_ids",
                        lambda puuid, count=5, start_time=None, end_time=None: [f"LA1_{puuid}_A", f"LA1_{puuid}_B"])
    monkeypatch.setattr(endpoints, "get_match",
                        lambda match_id: history_match(match_id, match_id.split("_")[1], TARGET_START - 5_000_000))
    conn = db.connect(tmp_path / "t.sqlite")
    db.add_target_match(conn, "LA1_T", "seed", "GOLD")
    db.save_fetched_match(
        conn, "LA1_T", {"game_start_ts": TARGET_START, "patch": "16.8", "blue_win": 1},
        [{"puuid": puuid, "team_id": 100 if i < 5 else 200, "team_position": "MIDDLE", "champion_id": 7,
          "champion_name": "Leblanc", "win": int(i < 5)} for i, puuid in enumerate(players)],
    )

    assert history_builder.build_histories(conn) == 1

    rows = conn.execute("SELECT puuid, history_rank, history_match_id FROM player_history").fetchall()
    assert len(rows) == 20, "2 games for each of the 10 players"
    for puuid in players:
        own = {row["history_rank"]: row["history_match_id"] for row in rows if row["puuid"] == puuid}
        assert own == {0: f"LA1_{puuid}_A", 1: f"LA1_{puuid}_B"}


def test_save_history_rolls_back_when_an_insert_fails(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = db.connect(path)
    db.add_target_match(conn, "LA1_T", "seed", "GOLD")
    db.save_fetched_match(conn, "LA1_T", {"game_start_ts": TARGET_START, "patch": "16.8", "blue_win": 1}, [])
    good = {"puuid": "me", "history_rank": 0,
            **history_builder.summarize_for_player(history_match("LA1_A", "me", 123), "me")}
    broken = {key: value for key, value in good.items() if key != "kills"} | {"history_rank": 1}

    with pytest.raises(sqlite3.Error):
        db.save_history(conn, "LA1_T", [good, broken])
    conn.close()

    conn = db.connect(path)  # reopen: only what was committed is still there
    assert conn.execute("SELECT COUNT(*) FROM player_history").fetchone()[0] == 0, "the good row is undone too"
    assert db.count_target_matches_by_status(conn) == {"fetched": 1}, "still waiting -> retried next run"


def test_no_history_row_is_ever_after_its_target(tmp_path, fake_riot):
    """The leakage rule from the plan, checked directly on the stored data."""
    fake_riot["matches"] = {
        "LA1_OLD2": history_match("LA1_OLD2", "me", TARGET_START - 5_000_000),
        "LA1_OLD1": history_match("LA1_OLD1", "me", TARGET_START - 9_000_000),
        "LA1_FUTURE": history_match("LA1_FUTURE", "me", TARGET_START),  # same instant counts as future
    }
    conn = db.connect(tmp_path / "t.sqlite")
    db.add_target_match(conn, "LA1_T", "seed", "GOLD")
    db.save_fetched_match(conn, "LA1_T", {"game_start_ts": TARGET_START, "patch": "16.8", "blue_win": 0},
                          [{"puuid": "me", "team_id": 200, "team_position": "MIDDLE", "champion_id": 7,
                            "champion_name": "Leblanc", "win": 1}])
    history_builder.build_histories(conn)
    leaks = conn.execute("""
        SELECT COUNT(*) FROM player_history ph
        JOIN target_matches tm ON tm.match_id = ph.target_match_id
        WHERE ph.game_start_ts >= tm.game_start_ts
    """).fetchone()[0]
    assert leaks == 0


def test_expired_key_stops_the_run_instead_of_saving_a_partial_history(fake_riot, monkeypatch):
    def expired_key(match_id):
        raise requests.HTTPError("403", response=FakeHTTPResponse(403))
    monkeypatch.setattr(endpoints, "get_match", expired_key)
    with pytest.raises(requests.HTTPError):
        history_builder.player_history("me", TARGET_START)


def test_player_blocked_by_firewall_is_skipped_without_asking(monkeypatch):
    """Riot's firewall (Cloudflare) blocks PUUIDs containing "--": skip that player, keep going."""
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("a blocked player must be skipped before any request")
    monkeypatch.setattr(endpoints, "get_match_ids", must_not_be_called)
    assert history_builder.player_history("abc--def", TARGET_START) == []


def test_firewall_403_for_a_normal_player_still_stops_the_run(monkeypatch):
    """If the firewall starts blocking us for another reason, stop: don't save empty histories."""
    def blocked(puuid, count=5, start_time=None, end_time=None):
        raise requests.HTTPError("403", response=FakeHTTPResponse(403))
    monkeypatch.setattr(endpoints, "get_match_ids", blocked)
    with pytest.raises(requests.HTTPError):
        history_builder.player_history("normal-puuid", TARGET_START)


def test_riot_403_on_match_ids_still_stops_the_run(monkeypatch):
    """A 403 from Riot itself means the key is bad: stop, don't save an empty history."""
    def expired(puuid, count=5, start_time=None, end_time=None):
        raise requests.HTTPError("403", response=FakeHTTPResponse(403))
    monkeypatch.setattr(endpoints, "get_match_ids", expired)
    with pytest.raises(requests.HTTPError):
        history_builder.player_history("me", TARGET_START)
