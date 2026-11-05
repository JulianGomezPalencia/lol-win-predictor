"""Tests for get_active_game, export of spells and the run_pipeline arguments. No real API calls."""
import csv
import hashlib
import sqlite3

import pytest
import requests

from scripts import export_dataset, run_pipeline
from src.riot_client import endpoints


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def raise_http(status):
    def fail(url, **kwargs):
        raise requests.HTTPError(str(status), response=FakeResponse(status))
    return fail


def test_get_active_game_returns_game(monkeypatch):
    monkeypatch.setattr(endpoints, "get", lambda url, **k: {"gameId": 7, "url": url})
    game = endpoints.get_active_game("PUUID")
    assert game["gameId"] == 7 and "spectator/v5/active-games/by-summoner/PUUID" in game["url"]


def test_get_active_game_none_on_404(monkeypatch):
    monkeypatch.setattr(endpoints, "get", raise_http(404))
    assert endpoints.get_active_game("P") is None


def test_get_active_game_raises_on_403(monkeypatch):
    monkeypatch.setattr(endpoints, "get", raise_http(403))
    with pytest.raises(requests.HTTPError):
        endpoints.get_active_game("P")


def test_export_spells_uses_salt_and_cache_only(tmp_path, monkeypatch):
    memory = sqlite3.connect(":memory:")
    memory.execute("CREATE TABLE match_participants (match_id TEXT, puuid TEXT)")
    memory.executemany("INSERT INTO match_participants VALUES (?, ?)", [("M1", "puu1"), ("M2", "puu2")])
    monkeypatch.setattr(export_dataset, "snapshot_connection", lambda: memory)
    monkeypatch.setattr(export_dataset, "load_salt", lambda: "salt")
    cached = {"M1": {"info": {"participants": [{"puuid": "puu1", "summoner1Id": 4, "summoner2Id": 11}]}}}
    count = export_dataset.export_spells(tmp_path, load_match=cached.get)  # M2 not cached: skipped
    assert count == 1
    rows = list(csv.reader(open(tmp_path / "participant_spells.csv")))
    assert rows[0] == ["match_id", "player_id", "spell1_id", "spell2_id"]
    assert rows[1] == ["M1", "p_" + hashlib.sha256(b"saltpuu1").hexdigest()[:12], "4", "11"]
    assert [p.name for p in tmp_path.iterdir()] == ["participant_spells.csv"]


class FakeConn:
    def close(self):
        pass


def run_main(monkeypatch, argv, calls):
    monkeypatch.setattr("sys.argv", ["run_pipeline", *argv])
    monkeypatch.setattr(run_pipeline.db, "connect", lambda: FakeConn())
    monkeypatch.setattr(run_pipeline.db, "count_target_matches_by_status", lambda conn: {})
    monkeypatch.setattr(run_pipeline.match_discovery, "discover",
                        lambda conn, **kw: calls.setdefault("discover", kw) and 0)
    monkeypatch.setattr(run_pipeline.match_fetcher, "fetch_pending",
                        lambda conn, **kw: calls.setdefault("fetch", kw) and {})
    run_pipeline.main()


def test_pipeline_arguments_reach_discover_and_fetch(monkeypatch):
    calls = {}
    run_main(monkeypatch, ["--steps", "discover,fetch", "--matches-per-player", "15", "--days-back", "10",
                           "--limit", "300"], calls)
    assert calls["discover"] == {"matches_per_player": 15, "days_back": 10}
    assert calls["fetch"] == {"limit": 300}


def test_pipeline_discover_defaults_unchanged(monkeypatch):
    calls = {}
    run_main(monkeypatch, ["--steps", "discover"], calls)
    assert calls["discover"] == {"matches_per_player": 5, "days_back": 14}
