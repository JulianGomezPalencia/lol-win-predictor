"""
The one function that sends requests to the Riot API: get().

It takes care of three things so the agents don't have to:
  - rate limit: calls are spaced out so we never break the personal key's limits
  - retries: a 429 (too many requests), a 5xx (Riot having trouble) or a network error
    (Wi-Fi drop, connection reset) is temporary, so we wait and try again
  - giving up: if the trouble lasts longer than MAX_OUTAGE_SECONDS we raise
    RiotUnavailableError; the pipeline then stops cleanly and can simply be run again

Tests: pytest tests/test_client.py -v
"""
import time

import requests

from src.riot_client.config import API_KEY

HEADERS = {"X-Riot-Token": API_KEY}

# 100 requests / 120 seconds = 1 request every 1.2 seconds. We use 1.25 s: exactly 1.2 s sits
# right on the limit, and network delays could then push 101 requests into Riot's 2-minute window.
MIN_SECONDS_BETWEEN_CALLS = 1.25

# How long we keep retrying before giving up. A long run (hours) will meet Wi-Fi drops and
# short Riot outages; 30 minutes rides out most of them without anyone watching.
MAX_OUTAGE_SECONDS = 30 * 60
MAX_BACKOFF_SECONDS = 60  # waits grow 1, 2, 4, ... s, but never longer than this

# Errors that mean "the connection failed", not "Riot said no" — worth retrying.
#   ChunkedEncodingError  = the connection dropped while the answer was still arriving
#   ContentDecodingError  = the compressed answer arrived damaged
NETWORK_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
)

# Remembers when the last request was sent. Starts at 0 = "long ago".
# The leading underscore means "private to this file — don't use from outside".
_last_call_time = 0.0


class RiotUnavailableError(RuntimeError):
    """Riot (or the internet connection) kept failing for MAX_OUTAGE_SECONDS.

    Nothing is wrong with the request itself, so the run can simply be started again later.
    """


def _wait_for_rate_limit() -> None:
    """Sleep just long enough so calls are at least MIN_SECONDS_BETWEEN_CALLS apart."""
    global _last_call_time  # lets this function change the variable defined above
    now = time.monotonic()
    elapsed = now - _last_call_time
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time = time.monotonic()


def _backoff_seconds(attempt: int) -> int:
    """1, 2, 4, 8, ... seconds, but never more than MAX_BACKOFF_SECONDS."""
    return min(2 ** attempt, MAX_BACKOFF_SECONDS)


def _retry_after_seconds(response) -> int | None:
    """The wait Riot asks for in the Retry-After header, or None if it's missing or unreadable.

    Riot's own rate limiter always sends it, but a 429 from the service behind it may not.
    """
    try:
        return int(response.headers["Retry-After"])
    except (KeyError, TypeError, ValueError):
        return None


def get(url: str, params: dict | None = None) -> dict | list:
    """GET a Riot API url and return the parsed JSON, handling rate limits and retries."""
    started = time.monotonic()
    attempt = 0
    while True:
        _wait_for_rate_limit()
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        except NETWORK_ERRORS as error:
            # A network blip (Wi-Fi drop, slow answer) is temporary, like a 5xx: wait and retry.
            problem = error
            wait = _backoff_seconds(attempt)
        else:
            if response.status_code == 429:
                problem = "429 rate limited"
                wait = _retry_after_seconds(response) or _backoff_seconds(attempt)
            elif response.status_code >= 500:
                problem = f"{response.status_code} server error"
                wait = _backoff_seconds(attempt)
            else:
                response.raise_for_status()  # other 4xx: retrying won't help, let the caller decide
                return response.json()

        attempt += 1
        waited = time.monotonic() - started
        if waited + wait > MAX_OUTAGE_SECONDS:
            # Don't sleep first: the wait would end past the limit anyway.
            raise RiotUnavailableError(
                f"Gave up on {url} after {attempt} attempts over {waited / 60:.0f} min "
                f"(last problem: {problem})"
            )
        time.sleep(wait)
