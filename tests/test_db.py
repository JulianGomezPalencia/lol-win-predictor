"""
Tests for src/storage/db.py. Each test gets a fresh in-memory database (":memory:"),
so nothing touches your real data/db/lol.sqlite.

Run with:  pytest tests/test_db.py -v
"""
import pytest

from src.storage import db


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection  # the test runs here
    connection.close()


def test_connect_creates_players_table(conn):
    tables = [row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "players" in tables


def test_upsert_inserts_new_player(conn):
    db.upsert_player(conn, "puuid-1", "GOLD", "II", 45)
    row = conn.execute("SELECT * FROM players WHERE puuid = ?", ("puuid-1",)).fetchone()
    assert row["tier"] == "GOLD"
    assert row["division"] == "II"
    assert row["lp"] == 45
    assert row["seeded_at"]  # filled automatically


def test_upsert_updates_existing_player_instead_of_duplicating(conn):
    db.upsert_player(conn, "puuid-1", "GOLD", "II", 45)
    db.upsert_player(conn, "puuid-1", "GOLD", "I", 10)  # they climbed
    rows = conn.execute("SELECT * FROM players").fetchall()
    assert len(rows) == 1, "same puuid must not create a second row"
    assert rows[0]["division"] == "I"
    assert rows[0]["lp"] == 10


def test_upsert_handles_quotes_safely(conn):
    # A value with a quote would break SQL built with f-strings.
    db.upsert_player(conn, "it's-a-puuid", "GOLD", "II", 0)
    assert conn.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 1


def test_count_players_by_tier(conn):
    db.upsert_player(conn, "a", "GOLD", "I", 0)
    db.upsert_player(conn, "b", "GOLD", "IV", 0)
    db.upsert_player(conn, "c", "IRON", "II", 0)
    assert db.count_players_by_tier(conn) == {"GOLD": 2, "IRON": 1}


def test_count_players_by_tier_empty(conn):
    assert db.count_players_by_tier(conn) == {}
