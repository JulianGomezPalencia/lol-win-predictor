"""
Tests for the two new endpoints and the seeder agent. No real API calls.

Run with:  pytest tests/test_seeder.py -v
"""
import pytest

from src.agents import seeder
from src.riot_client import endpoints
from src.storage import db

# --- Part 1: the new endpoint functions in endpoints.py ---------------------------------


@pytest.fixture
def captured_urls(monkeypatch):
    """Replaces the `get` used by endpoints.py and records which URL was requested."""
    urls = []

    def fake_get(url, params=None):
        urls.append(url)
        return {"entries": []} if "leagues" in url else []

    monkeypatch.setattr(endpoints, "get", fake_get)
    return urls


def test_get_league_entries_url(captured_urls):
    endpoints.get_league_entries("GOLD", "II")
    assert captured_urls == [
        "https://la1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/GOLD/II"
    ], "league-v4 is a PLATFORM endpoint (la1), not a region one (americas)"


@pytest.mark.parametrize("tier, path", [
    ("CHALLENGER", "challengerleagues"),
    ("GRANDMASTER", "grandmasterleagues"),
    ("MASTER", "masterleagues"),
])
def test_get_apex_league_url(captured_urls, tier, path):
    endpoints.get_apex_league(tier)
    assert captured_urls == [
        f"https://la1.api.riotgames.com/lol/league/v4/{path}/by-queue/RANKED_SOLO_5x5"
    ]


# --- Part 2: pick_players ---------------------------------------------------------------


def entry(puuid, inactive=False, lp=0):
    return {"puuid": puuid, "inactive": inactive, "leaguePoints": lp}


def test_pick_players_takes_first_n():
    entries = [entry("a"), entry("b"), entry("c")]
    assert [e["puuid"] for e in seeder.pick_players(entries, 2)] == ["a", "b"]


def test_pick_players_skips_inactive():
    entries = [entry("a", inactive=True), entry("b"), entry("c", inactive=True), entry("d")]
    assert [e["puuid"] for e in seeder.pick_players(entries, 5)] == ["b", "d"]


def test_pick_players_with_fewer_entries_than_n():
    assert seeder.pick_players([entry("a")], 10) == [entry("a")]


# --- Part 3: seed() ---------------------------------------------------------------------


@pytest.fixture
def fake_league(monkeypatch):
    """Fake Riot: every division/league has 50 active players with unique puuids."""
    def fake_entries(tier, division):
        return [entry(f"{tier}-{division}-{i}", lp=i) for i in range(50)]

    def fake_apex(tier):
        return {"tier": tier, "entries": [entry(f"{tier}-{i}", lp=1000 + i) for i in range(50)]}

    monkeypatch.setattr(endpoints, "get_league_entries", fake_entries)
    monkeypatch.setattr(endpoints, "get_apex_league", fake_apex)


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def test_seed_saves_equal_number_per_tier(conn, fake_league):
    saved = seeder.seed(conn, players_per_division=3)
    counts = db.count_players_by_tier(conn)
    all_tiers = seeder.TIERS + seeder.APEX_TIERS
    assert set(counts) == set(all_tiers), "every tier, Iron to Challenger, should have players"
    assert all(n == 12 for n in counts.values()), f"each tier should have 4 x 3 = 12 players, got {counts}"
    assert saved == 12 * len(all_tiers)


def test_seed_stores_tier_division_and_lp(conn, fake_league):
    seeder.seed(conn, players_per_division=1)
    gold = conn.execute("SELECT * FROM players WHERE puuid = 'GOLD-II-0'").fetchone()
    assert (gold["tier"], gold["division"], gold["lp"]) == ("GOLD", "II", 0)
    chall = conn.execute("SELECT * FROM players WHERE puuid = 'CHALLENGER-0'").fetchone()
    assert (chall["tier"], chall["division"], chall["lp"]) == ("CHALLENGER", "I", 1000)
