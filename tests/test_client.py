"""
Tests for src/riot_client/client.py.

These tests never touch the real Riot API. They replace `requests.get` and `time.sleep`
with fakes ("mocks") so we can pretend Riot answered 429 or 500, and check what get() does,
instantly and without spending any of your rate limit.

Run with:  pytest tests/test_client.py -v
"""
import pytest
import requests

from src.riot_client import client


class FakeResponse:
    """Pretends to be a requests.Response with a given status code and JSON body."""

    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.fixture
def fake_api(monkeypatch):
    """Replaces requests.get with a fake that returns pre-programmed responses in order.

    Once the list runs out, the last response repeats (handy for "Riot is down for good").
    Also replaces the clock: time.sleep doesn't actually wait, it records the sleep and moves
    a fake clock forward, so get() "sees" the time pass and its outage limit can be tested.
    """
    state = {"responses": [], "calls": 0, "sleeps": [], "now": 1000.0}

    def fake_get(url, **kwargs):
        response = state["responses"][min(state["calls"], len(state["responses"]) - 1)]
        state["calls"] += 1
        if isinstance(response, Exception):
            raise response
        return response

    def fake_sleep(seconds):
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(client.requests, "get", fake_get)
    monkeypatch.setattr(client.time, "sleep", fake_sleep)
    monkeypatch.setattr(client.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(client, "_last_call_time", 0.0)
    return state


def test_returns_json_on_success(fake_api):
    fake_api["responses"] = [FakeResponse(200, {"puuid": "abc"})]
    assert client.get("https://example") == {"puuid": "abc"}
    assert fake_api["calls"] == 1


def test_retries_after_429_using_retry_after_header(fake_api):
    fake_api["responses"] = [
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(200, ["LA1_1"]),
    ]
    assert client.get("https://example") == ["LA1_1"]
    assert fake_api["calls"] == 2
    assert 7 in fake_api["sleeps"], "should sleep for the number of seconds in Retry-After"


def test_retries_on_server_error(fake_api):
    fake_api["responses"] = [FakeResponse(503), FakeResponse(200, {"ok": True})]
    assert client.get("https://example") == {"ok": True}
    assert fake_api["calls"] == 2


def test_does_not_retry_on_404(fake_api):
    fake_api["responses"] = [FakeResponse(404), FakeResponse(200, {"never": "reached"})]
    with pytest.raises(requests.HTTPError):
        client.get("https://example")
    assert fake_api["calls"] == 1, "a 404 won't fix itself — don't waste requests retrying"


def test_gives_up_when_the_outage_lasts_too_long(fake_api):
    fake_api["responses"] = [FakeResponse(500)]  # Riot answers 500 forever
    with pytest.raises(client.RiotUnavailableError, match="500 server error"):
        client.get("https://example")
    assert sum(fake_api["sleeps"]) <= client.MAX_OUTAGE_SECONDS, "never waits past the limit"
    assert sum(fake_api["sleeps"]) > client.MAX_OUTAGE_SECONDS - client.MAX_BACKOFF_SECONDS - 5, \
        "keeps trying for (almost) the whole limit before giving up"


def test_rate_limiter_spaces_out_calls(fake_api, monkeypatch):
    # Pretend the clock reads 100.0s and the previous call happened at 99.5s (0.5s ago).
    monkeypatch.setattr(client.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(client, "_last_call_time", 99.5)
    client._wait_for_rate_limit()
    assert fake_api["sleeps"] == [pytest.approx(client.MIN_SECONDS_BETWEEN_CALLS - 0.5)]


def test_rate_limiter_does_not_sleep_when_enough_time_passed(fake_api, monkeypatch):
    monkeypatch.setattr(client.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(client, "_last_call_time", 90.0)
    client._wait_for_rate_limit()
    assert fake_api["sleeps"] == []


def test_retries_after_network_error(fake_api):
    fake_api["responses"] = [requests.ConnectionError("wifi dropped"), FakeResponse(200, {"ok": True})]
    assert client.get("https://example") == {"ok": True}
    assert fake_api["calls"] == 2


def test_retries_when_connection_drops_mid_answer(fake_api):
    fake_api["responses"] = [requests.exceptions.ChunkedEncodingError("reset"), FakeResponse(200, {"ok": True})]
    assert client.get("https://example") == {"ok": True}


def test_retries_when_the_answer_arrives_damaged(fake_api):
    fake_api["responses"] = [requests.exceptions.ContentDecodingError("bad gzip"), FakeResponse(200, {"ok": True})]
    assert client.get("https://example") == {"ok": True}


def test_gives_up_on_network_errors_and_says_why(fake_api):
    fake_api["responses"] = [requests.ConnectionError("Connection reset by peer")]
    with pytest.raises(client.RiotUnavailableError, match="Connection reset by peer"):
        client.get("https://example")


def test_backoff_grows_but_is_capped(fake_api, monkeypatch):
    monkeypatch.setattr(client, "_wait_for_rate_limit", lambda: None)  # only count backoff sleeps
    fake_api["responses"] = [FakeResponse(503)]
    with pytest.raises(client.RiotUnavailableError):
        client.get("https://example")
    assert fake_api["sleeps"][:8] == [1, 2, 4, 8, 16, 32, 60, 60]


def test_429_without_retry_after_backs_off(fake_api, monkeypatch):
    """A 429 from Riot's service (not its rate limiter) may come without Retry-After."""
    monkeypatch.setattr(client, "_wait_for_rate_limit", lambda: None)
    fake_api["responses"] = [FakeResponse(429)] * 3 + [FakeResponse(200, {"ok": True})]
    assert client.get("https://example") == {"ok": True}
    assert fake_api["sleeps"] == [1, 2, 4]


def test_429_with_unreadable_retry_after_does_not_crash(fake_api, monkeypatch):
    monkeypatch.setattr(client, "_wait_for_rate_limit", lambda: None)
    fake_api["responses"] = [FakeResponse(429, headers={"Retry-After": "soon"}), FakeResponse(200, {"ok": True})]
    assert client.get("https://example") == {"ok": True}
    assert fake_api["sleeps"] == [1]


def test_429_forever_gives_up(fake_api):
    fake_api["responses"] = [FakeResponse(429, headers={"Retry-After": "10"})]
    with pytest.raises(client.RiotUnavailableError, match="429"):
        client.get("https://example")
