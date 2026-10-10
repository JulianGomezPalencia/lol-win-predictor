"""
Tests for Exercise 6: get_match_ids time filters, the target_matches queue, and Agent 2.

Run with:  pytest tests/test_match_discovery.py -v
"""
import pytest

from src.agents import match_discovery
from src.riot_client import endpoints
from src.storage import db


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


# --- Part 1: get_match_ids() with optional start_time / end_time -----------------------


@pytest.fixture
def captured_params(monkeypatch):
    calls = []

    def fake_get(url, params=None):
        calls.append(params)
        return []

    monkeypatch.setattr(endpoints, "get", fake_get)
    return calls


def test_get_match_ids_default_params_unchanged(captured_params):
    endpoints.get_match_ids("p1")
    assert captured_params == [{"queue": 420, "count": 5}], \
        "without time filters, don't send startTime/endTime at all"


def test_get_match_ids_sends_start_time(captured_params):
    endpoints.get_match_ids("p1", count=3, start_time=1_700_000_000)
    assert captured_params == [{"queue": 420, "count": 3, "startTime": 1_700_000_000}]


def test_get_match_ids_sends_end_time(captured_params):
    endpoints.get_match_ids("p1", end_time=1_700_000_000)
    assert captured_params == [{"queue": 420, "count": 5, "endTime": 1_700_000_000}]


# --- Part 2: database functions --------------------------------------------------------


def test_connect_creates_target_matches_table(conn):
    tables = [r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "target_matches" in tables


def test_get_all_players(conn):
    db.upsert_player(conn, "a", "GOLD", "I", 0)
    db.upsert_player(conn, "b", "IRON", "IV", 0)
    players = db.get_all_players(conn)
    assert sorted((p["puuid"], p["tier"]) for p in players) == [("a", "GOLD"), ("b", "IRON")]


def test_add_target_match_new_is_pending(conn):
    assert db.add_target_match(conn, "LA1_1", "puuid-a", "GOLD") is True
    row = conn.execute("SELECT * FROM target_matches WHERE match_id = 'LA1_1'").fetchone()
    assert row["status"] == "pending"
    assert (row["seed_puuid"], row["seed_tier"]) == ("puuid-a", "GOLD")


def test_add_target_match_duplicate_returns_false_and_keeps_one_row(conn):
    db.add_target_match(conn, "LA1_1", "puuid-a", "GOLD")
    assert db.add_target_match(conn, "LA1_1", "puuid-b", "GOLD") is False
    assert conn.execute("SELECT COUNT(*) FROM target_matches").fetchone()[0] == 1


def test_count_target_matches_by_status(conn):
    db.add_target_match(conn, "LA1_1", "a", "GOLD")
    db.add_target_match(conn, "LA1_2", "a", "GOLD")
    assert db.count_target_matches_by_status(conn) == {"pending": 2}


# --- Part 3: the agent -----------------------------------------------------------------

NOW = 1_800_000_000  # a fixed fake "now", so the test always computes the same start_time


@pytest.fixture
def fake_riot(monkeypatch):
    """Two players share match LA1_SHARED. Records the arguments each call received."""
    calls = []
    histories = {
        "gold-player": ["LA1_SHARED", "LA1_G1"],
        "iron-player": ["LA1_SHARED", "LA1_I1", "LA1_I2"],
    }

    def fake_get_match_ids(puuid, count=5, start_time=None, end_time=None):
        calls.append({"puuid": puuid, "count": count, "start_time": start_time})
        return histories[puuid][:count]

    monkeypatch.setattr(endpoints, "get_match_ids", fake_get_match_ids)
    monkeypatch.setattr(match_discovery.time, "time", lambda: NOW + 0.75)
    return calls


@pytest.fixture
def seeded(conn):
    db.upsert_player(conn, "gold-player", "GOLD", "I", 0)
    db.upsert_player(conn, "iron-player", "IRON", "IV", 0)
    return conn


def test_discover_adds_unique_matches(seeded, fake_riot):
    new = match_discovery.discover(seeded, matches_per_player=5, days_back=14)
    assert new == 4, "LA1_SHARED appears twice but must be counted once"
    assert db.count_target_matches_by_status(seeded) == {"pending": 4}


def test_discover_uses_start_time_in_whole_seconds(seeded, fake_riot):
    match_discovery.discover(seeded, matches_per_player=2, days_back=14)
    expected = NOW - 14 * 24 * 60 * 60
    for call in fake_riot:
        assert call["start_time"] == expected
        assert isinstance(call["start_time"], int), "Riot expects whole seconds, not a float"
        assert call["count"] == 2


def test_discover_twice_adds_nothing_new(seeded, fake_riot):
    match_discovery.discover(seeded)
    assert match_discovery.discover(seeded) == 0, "re-running must be safe"
