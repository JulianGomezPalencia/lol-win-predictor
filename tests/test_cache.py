"""
Tests for src/riot_client/cache.py and for get_match() using the cache.

`tmp_path` is a pytest feature: a brand-new empty folder for each test, deleted afterwards.
We point CACHE_DIR at it so tests never touch your real data/ folder.

Run with:  pytest tests/test_cache.py -v
"""
import gzip
import json

import pytest

from src.riot_client import cache, endpoints

FAKE_MATCH = {"metadata": {"matchId": "LA1_123"}, "info": {"gameDuration": 2006, "participants": []}}


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "matches"  # does NOT exist yet — save_match must create it
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    return cache_dir


def test_cache_path_uses_match_id_and_json_gz(temp_cache):
    assert cache.cache_path("LA1_123") == temp_cache / "LA1_123.json.gz"


def test_load_returns_none_when_not_cached(temp_cache):
    assert cache.load_match("LA1_999") is None


def test_save_then_load_returns_same_match(temp_cache):
    cache.save_match("LA1_123", FAKE_MATCH)
    assert cache.load_match("LA1_123") == FAKE_MATCH


def test_save_creates_missing_folder(temp_cache):
    assert not temp_cache.exists()
    cache.save_match("LA1_123", FAKE_MATCH)
    assert temp_cache.exists()


def test_saved_file_is_gzipped_json(temp_cache):
    cache.save_match("LA1_123", FAKE_MATCH)
    with gzip.open(temp_cache / "LA1_123.json.gz", "rt", encoding="utf-8") as f:
        assert json.load(f) == FAKE_MATCH


# --- Part 2: get_match() in endpoints.py should use the cache ---------------------------

@pytest.fixture
def fake_riot(monkeypatch):
    """Replaces the `get` that endpoints.py uses, and counts how many times Riot is 'called'."""
    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        return FAKE_MATCH

    monkeypatch.setattr(endpoints, "get", fake_get)
    return calls


def test_get_match_downloads_once_then_uses_cache(temp_cache, fake_riot):
    first = endpoints.get_match("LA1_123")
    second = endpoints.get_match("LA1_123")
    assert first == second == FAKE_MATCH
    assert len(fake_riot) == 1, "the second call should come from disk, not from Riot"


def test_get_match_reads_existing_cache_without_calling_riot(temp_cache, fake_riot):
    cache.save_match("LA1_123", FAKE_MATCH)
    assert endpoints.get_match("LA1_123") == FAKE_MATCH
    assert fake_riot == [], "match was already on disk — Riot should not be called at all"
