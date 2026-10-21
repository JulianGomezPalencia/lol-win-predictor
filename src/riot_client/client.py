import time

import requests

from src.riot_client.config import API_KEY

HEADERS = {"X-Riot-Token": API_KEY}

# 100 requests / 120 seconds = 1 request every 1.2 seconds.
# Spacing calls evenly is the simplest way to never break the 2-minute limit.
MIN_SECONDS_BETWEEN_CALLS = 1.2
# A long run (hours) will meet Wi-Fi drops and short Riot outages. With 8 attempts and the
# waits below (1, 2, 4, 8, 16, 32, 60 s) we ride out about 2 minutes of trouble before giving up.
MAX_RETRIES = 8
MAX_BACKOFF_SECONDS = 60

# Errors that mean "the connection failed", not "Riot said no" — worth retrying.
# ChunkedEncodingError = the connection dropped while the answer was still arriving.
NETWORK_ERRORS = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)

# Remembers when the last request was sent. Starts at 0 = "long ago".
# The leading underscore means "private to this file — don't use from outside".
_last_call_time = 0.0


class RiotUnavailableError(RuntimeError):
    """Riot (or the internet connection) kept failing after every retry.

    Nothing is wrong with the request itself, so the run can simply be started again later.
    """


def _backoff_seconds(attempt: int) -> int:
    """1, 2, 4, 8, ... seconds, but never more than MAX_BACKOFF_SECONDS."""
    return min(2 ** attempt, MAX_BACKOFF_SECONDS)


def _wait_for_rate_limit() -> None:
    """Sleep just long enough so calls are at least MIN_SECONDS_BETWEEN_CALLS apart."""
    global _last_call_time  # lets this function change the variable defined above
    now = time.monotonic()
    elapsed = now - _last_call_time
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time = time.monotonic()


def get(url: str, params: dict | None = None) -> dict | list:
    """GET a Riot API url and return the parsed JSON, handling rate limits and retries."""
    last_problem = None
    for attempt in range(MAX_RETRIES):
        is_last_attempt = attempt == MAX_RETRIES - 1
        _wait_for_rate_limit()
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        except NETWORK_ERRORS as error:
            # A network blip (Wi-Fi drop, slow answer) is temporary, like a 5xx: wait and retry.
            last_problem = error
            if not is_last_attempt:
                time.sleep(_backoff_seconds(attempt))
            continue
        if response.status_code == 429:
            last_problem = "429 rate limited"
            if not is_last_attempt:
                time.sleep(int(response.headers.get("Retry-After", 1)))
            continue
        if response.status_code >= 500:
            last_problem = f"{response.status_code} server error"
            if not is_last_attempt:
                time.sleep(_backoff_seconds(attempt))
            continue
        response.raise_for_status()
        return response.json()
    raise RiotUnavailableError(f"Gave up on {url} after {MAX_RETRIES} attempts (last problem: {last_problem})")
