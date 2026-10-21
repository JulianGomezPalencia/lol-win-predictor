"""
Tests for Agent 5 (player snapshot) and the rank/mastery endpoints. No real API calls.

Run with:  pytest tests/test_player_snapshot.py -v
"""
import pytest
import requests

from src.agents import player_snapshot
from src.riot_client import endpoints
from src.storage import db


class FakeHTTPResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_solo_queue_entry_picks_solo_queue(monkeypatch):
    monkeypatch.setattr(endpoints, "get", lambda url, params=None: [
        {"queueType": "RANKED_FLEX_SR", "tier": "IRON"},
        {"queueType": "RANKED_SOLO_5x5", "tier": "GOLD"},
    ])
    assert endpoints.get_solo_queue_entry("p")["tier"] == "GOLD"


def test_solo_queue_entry_unranked_is_none(monkeypatch):
    monkeypatch.setattr(endpoints, "get", lambda url, params=None: [])
    assert endpoints.get_solo_queue_entry("p") is None


def test_champion_mastery_404_is_none(monkeypatch):
    def fake_get(url, params=None):
        raise requests.HTTPError("404", response=FakeHTTPResponse(404))
    monkeypatch.setattr(endpoints, "get", fake_get)
    assert endpoints.get_champion_mastery("p", 7) is None


def test_champion_mastery_other_errors_still_raise(monkeypatch):
    def fake_get(url, params=None):
        raise requests.HTTPError("403", response=FakeHTTPResponse(403))
    monkeypatch.setattr(endpoints, "get", fake_get)
    with pytest.raises(requests.HTTPError):
        endpoints.get_champion_mastery("p", 7)


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "t.sqlite")
    db.add_target_match(connection, "LA1_T", "seed", "GOLD")
    db.save_fetched_match(
        connection, "LA1_T", {"game_start_ts": 1, "patch": "16.8", "blue_win": 1},
        [{"puuid": "ranked", "team_id": 100, "team_position": "TOP", "champion_id": 7,
          "champion_name": "Leblanc", "win": 1},
         {"puuid": "unranked", "team_id": 200, "team_position": "TOP", "champion_id": 8,
          "champion_name": "Vladimir", "win": 0}],
    )
    yield connection
    connection.close()


def test_snapshot_players(conn, monkeypatch):
    ranks = {"ranked": {"tier": "GOLD", "rank": "II", "leaguePoints": 40, "wins": 10, "losses": 8},
             "unranked": None}
    monkeypatch.setattr(endpoints, "get_solo_queue_entry", lambda puuid: ranks[puuid])
    monkeypatch.setattr(endpoints, "get_champion_mastery", lambda puuid, champ: None if puuid == "unranked" else {
        "championId": champ, "championLevel": 5, "championPoints": 21000, "lastPlayTime": 99})

    assert player_snapshot.snapshot_players(conn) == 2

    gold = conn.execute("SELECT * FROM player_snapshots WHERE puuid = 'ranked'").fetchone()
    assert (gold["tier"], gold["division"], gold["lp"]) == ("GOLD", "II", 40)
    none = conn.execute("SELECT * FROM player_snapshots WHERE puuid = 'unranked'").fetchone()
    assert none["tier"] is None and none["lp"] is None, "unranked players are stored with NULL rank"
    assert conn.execute("SELECT COUNT(*) FROM champion_mastery").fetchone()[0] == 1
    assert player_snapshot.snapshot_players(conn) == 0, "already snapshotted players are skipped"


def test_snapshot_skips_players_blocked_by_firewall(tmp_path, monkeypatch):
    connection = db.connect(tmp_path / "t.sqlite")
    db.add_target_match(connection, "LA1_T", "seed", "GOLD")
    db.save_fetched_match(
        connection, "LA1_T", {"game_start_ts": 1, "patch": "16.8", "blue_win": 1},
        [{"puuid": "ok", "team_id": 100, "team_position": "TOP", "champion_id": 7,
          "champion_name": "Leblanc", "win": 1},
         {"puuid": "bad--id", "team_id": 200, "team_position": "TOP", "champion_id": 8,
          "champion_name": "Vladimir", "win": 0}],
    )
    asked = []
    monkeypatch.setattr(endpoints, "get_solo_queue_entry", lambda puuid: asked.append(puuid))
    monkeypatch.setattr(endpoints, "get_champion_mastery", lambda puuid, champ: None)
    assert player_snapshot.snapshot_players(connection) == 1
    assert asked == ["ok"], "the blocked player is never requested"
    connection.close()


def test_is_firewall_blocked_only_for_double_dash():
    assert endpoints.is_firewall_blocked("abc--def")
    assert not endpoints.is_firewall_blocked("abc-def")
