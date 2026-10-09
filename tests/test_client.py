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

    Also replaces time.sleep so tests don't actually wait, and records every sleep.
    """
    state = {"responses": [], "calls": 0, "sleeps": []}

    def fake_get(url, **kwargs):
        response = state["responses"][state["calls"]]
        state["calls"] += 1
        return response

    monkeypatch.setattr(client.requests, "get", fake_get)
    monkeypatch.setattr(client.time, "sleep", lambda seconds: state["sleeps"].append(seconds))
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


def test_gives_up_after_max_retries(fake_api):
    fake_api["responses"] = [FakeResponse(500)] * client.MAX_RETRIES
    with pytest.raises(RuntimeError):
        client.get("https://example")
    assert fake_api["calls"] == client.MAX_RETRIES


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
