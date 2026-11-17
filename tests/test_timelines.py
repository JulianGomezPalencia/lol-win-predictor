"""Pruebas sin Riot ni datos locales: SQLite y caché temporales."""
import gzip
import json
import sqlite3

import pytest
import requests

from scripts import download_timelines as downloader
from src.riot_client import cache, client, endpoints

TIMELINE = {"metadata": {"matchId": "LA1_1"},
            "info": {"frames": [{"timestamp": 60000, "events": [{"type": "CHAMPION_KILL"}]}]}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "TIMELINE_CACHE_DIR", tmp_path / "timelines")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "matches")
    def forbidden(*args, **kwargs):
        pytest.fail("No debe haber llamadas de red reales")
    monkeypatch.setattr(requests, "get", forbidden)


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE target_matches (match_id TEXT, status TEXT, game_start_ts INTEGER)")
    yield conn
    conn.close()


def seed(conn, n):
    conn.executemany("INSERT INTO target_matches VALUES (?, 'history_done', ?)",
                     [(f"LA1_{i}", downloader.CUTOFF_TS + i + 1) for i in range(n)])


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError("private URL and key must never be printed", response=response)


def test_complete_cache_and_separate_match_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(endpoints, "get", lambda url: calls.append(url) or TIMELINE)
    cache.save_match("LA1_1", {"info": "match"})
    assert endpoints.get_match_timeline("LA1_1") == TIMELINE
    assert endpoints.get_match_timeline("LA1_1") == TIMELINE
    assert calls == [f"{endpoints.REGION_URL}/lol/match/v5/matches/LA1_1/timeline"]
    assert cache.load_match("LA1_1") == {"info": "match"}
    path = cache.timeline_cache_path("LA1_1")
    with gzip.open(path, "rt") as stream:
        assert json.load(stream) == TIMELINE
    assert list(path.parent.glob("*.tmp")) == []


@pytest.mark.parametrize("damage", [b"not gzip", b"\x1f\x8b", gzip.compress(b"{bad json")])
def test_damaged_cache_is_replaced(monkeypatch, damage):
    path = cache.timeline_cache_path("LA1_1")
    path.parent.mkdir()
    path.write_bytes(damage)
    assert cache.load_timeline("LA1_1") is None
    monkeypatch.setattr(endpoints, "get", lambda url: TIMELINE)
    assert endpoints.get_match_timeline("LA1_1") == TIMELINE
    assert cache.load_timeline("LA1_1") == TIMELINE


def test_resume_and_max_skip_cached(conn, monkeypatch):
    seed(conn, 4)
    calls = []
    monkeypatch.setattr(endpoints, "get", lambda url: calls.append(url) or TIMELINE)
    counts, code = downloader.download(conn, max_downloads=2)
    assert counts == dict(descargadas=2, en_cache=0, no_encontradas=0, errores=0)
    assert code == 0
    counts, _ = downloader.download(conn, max_downloads=1)
    assert counts["descargadas"] == 1 and counts["en_cache"] == 2
    counts, _ = downloader.download(conn)
    assert counts["descargadas"] == 1 and counts["en_cache"] == 3
    assert len(calls) == 4


def test_interrupted_save_preserves_previous_cache(monkeypatch):
    cache.save_timeline("LA1_1", TIMELINE)
    def interrupted(*args):
        raise OSError("interrupted")
    monkeypatch.setattr(cache.os, "replace", interrupted)
    with pytest.raises(OSError):
        cache.save_timeline("LA1_1", {"replacement": True})
    assert cache.load_timeline("LA1_1") == TIMELINE


def test_priority_and_only_history_done(conn, monkeypatch):
    cutoff = downloader.CUTOFF_TS
    rows = [("older", "history_done", cutoff - 10), ("new2", "history_done", cutoff + 2),
            ("new1b", "history_done", cutoff + 1), ("equal", "history_done", cutoff),
            ("new1a", "history_done", cutoff + 1), ("skip", "fetched", cutoff + 3)]
    conn.executemany("INSERT INTO target_matches VALUES (?, ?, ?)", rows)
    calls = []
    monkeypatch.setattr(downloader, "get_match_timeline", lambda mid: calls.append(mid))
    downloader.download(conn)
    assert calls == ["new1a", "new1b", "new2", "equal", "older"]


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_stops_and_hides_details(conn, monkeypatch, status):
    seed(conn, 3)
    calls = []
    def rejected(mid):
        calls.append(mid)
        raise http_error(status)
    monkeypatch.setattr(downloader, "get_match_timeline", rejected)
    lines = []
    counts, code = downloader.download(conn, out=lines.append)
    assert len(calls) == 1 and counts["errores"] == 1 and code == 1
    assert "rechazó la clave" in lines[0]
    assert "private" not in " ".join(lines) and "LA1_" not in " ".join(lines)


def test_404_and_network_errors_continue(conn, monkeypatch):
    seed(conn, 3)
    outcomes = iter([http_error(404), requests.ConnectionError("private"), TIMELINE])
    def fetch(mid):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(downloader, "get_match_timeline", fetch)
    assert downloader.download(conn)[0] == dict(descargadas=1, en_cache=0, no_encontradas=1, errores=1)


@pytest.mark.parametrize("error", [requests.ConnectionError("private"), client.RiotUnavailableError("private")])
def test_twenty_consecutive_errors_stop(conn, monkeypatch, error):
    seed(conn, 25)
    def fetch(mid):
        raise error
    monkeypatch.setattr(downloader, "get_match_timeline", fetch)
    counts, code = downloader.download(conn)
    assert counts["errores"] == 20 and code == 1


@pytest.mark.parametrize("error", [requests.ConnectionError("private"), client.RiotUnavailableError("private")])
def test_cached_matches_do_not_reset_error_streak(conn, monkeypatch, error):
    seed(conn, 50)
    for i in range(1, 50, 2):
        cache.save_timeline(f"LA1_{i}", TIMELINE)
    calls = []
    def fetch(mid):
        calls.append(mid)
        raise error
    monkeypatch.setattr(downloader, "get_match_timeline", fetch)
    counts, code = downloader.download(conn)
    assert calls == [f"LA1_{i}" for i in range(0, 40, 2)]
    assert counts == dict(descargadas=0, en_cache=19, no_encontradas=0, errores=20)
    assert code == 1


@pytest.mark.parametrize("reset", [TIMELINE, http_error(404)])
def test_success_or_404_resets_error_streak(conn, monkeypatch, reset):
    seed(conn, 39)
    outcomes = iter([requests.Timeout()] * 19 + [reset] + [requests.Timeout()] * 19)
    def fetch(mid):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(downloader, "get_match_timeline", fetch)
    counts, code = downloader.download(conn)
    assert counts["errores"] == 38 and code == 0


def test_deadline_between_matches(conn, monkeypatch):
    seed(conn, 3)
    now = [1000.0]
    monkeypatch.setattr(client.time, "monotonic", lambda: now[0])
    def fetch(mid):
        now[0] += 60
        return TIMELINE
    monkeypatch.setattr(downloader, "get_match_timeline", fetch)
    counts, code = downloader.download(conn, minutes=1)
    assert counts["descargadas"] == 1 and code == 0
    assert client.check_deadline() is None


@pytest.mark.parametrize("status", [429, 503])
def test_deadline_caps_real_client_retries(conn, monkeypatch, status):
    seed(conn, 2)
    now = [1000.0]
    calls = []
    monkeypatch.setattr(client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(client.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(client, "_last_call_time", 0)
    def get(*args, **kwargs):
        calls.append(kwargs["timeout"])
        response = requests.Response()
        response.status_code = status
        response.headers["Retry-After"] = "120"
        return response
    monkeypatch.setattr(client.requests, "get", get)
    counts, code = downloader.download(conn, minutes=.01)
    assert counts == dict(descargadas=0, en_cache=0, no_encontradas=0, errores=0)
    assert calls == [pytest.approx(.6)] and code == 0
    assert now[0] == pytest.approx(1000.6)
    assert client.check_deadline() is None
    assert not cache.TIMELINE_CACHE_DIR.exists()


def test_progress_counts_every_hundred_and_final(conn, monkeypatch):
    seed(conn, 101)
    monkeypatch.setattr(downloader, "get_match_timeline", lambda mid: TIMELINE)
    lines = []
    downloader.download(conn, out=lines.append)
    assert lines == ["descargadas=100 en_cache=0 no_encontradas=0 errores=0",
                     "descargadas=101 en_cache=0 no_encontradas=0 errores=0"]


@pytest.mark.parametrize("args", [["--max", "0"], ["--minutes", "-1"], ["--minutes", "nan"],
                                  ["--minutes", "inf"]])
def test_cli_rejects_invalid_limits(args):
    with pytest.raises(SystemExit) as error:
        downloader.main(args)
    assert error.value.code == 2


def test_cli_uses_readonly_db(tmp_path, monkeypatch):
    path = tmp_path / "missing.sqlite"
    monkeypatch.setattr(downloader, "DB_PATH", path)
    assert downloader.main([]) == 1
    assert not path.exists()
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE target_matches (match_id TEXT, status TEXT, game_start_ts INTEGER)")
    conn.close()
    assert downloader.main(["--max", "1", "--minutes", "1"]) == 0
