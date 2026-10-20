"""
Routing: every endpoint must use the right Riot host. A wrong host gives 403/404 for real,
but no other test would notice, because they all fake the API.

  regional (americas): account-v1, match-v5
  platform (la1):      league-v4, champion-mastery-v4

Run with:  pytest tests/test_routing.py -v
"""
import pytest

from src.riot_client import cache, endpoints

REGIONAL = "https://americas.api.riotgames.com/"
PLATFORM = "https://la1.api.riotgames.com/"


@pytest.fixture
def urls(monkeypatch, tmp_path):
    seen = []

    def fake_get(url, params=None):
        seen.append(url)
        if "by-riot-id" in url:
            return {"puuid": "p"}
        if "leagues/by-queue" in url:
            return {"entries": []}
        if "/ids" in url or "entries" in url:
            return []
        return {"metadata": {"matchId": "LA1_1"}, "info": {}}

    monkeypatch.setattr(endpoints, "get", fake_get)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)  # so get_match never reads the real cache
    return seen


@pytest.mark.parametrize("call, host", [
    (lambda: endpoints.get_puuid("Name", "TAG"), REGIONAL),
    (lambda: endpoints.get_match_ids("p"), REGIONAL),
    (lambda: endpoints.get_match("LA1_1"), REGIONAL),
    (lambda: endpoints.get_league_entries("GOLD", "II"), PLATFORM),
    (lambda: endpoints.get_apex_league("CHALLENGER"), PLATFORM),
    (lambda: endpoints.get_solo_queue_entry("p"), PLATFORM),
    (lambda: endpoints.get_champion_mastery("p", 7), PLATFORM),
])
def test_endpoint_uses_the_right_host(urls, call, host):
    call()
    assert len(urls) == 1 and urls[0].startswith(host), urls
