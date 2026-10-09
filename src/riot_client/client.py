"""
EXERCISE 2 — the single gateway to the Riot API.

Every request in this project goes through `get()` below. That way rate limiting
and retries live in ONE place instead of being copy-pasted into every function.

Rules from Riot (Personal key):
  - 20 requests per 1 second
  - 100 requests per 2 minutes   <- this one is the real bottleneck
  - If you go over, Riot answers 429 with a "Retry-After" header (seconds to wait).

How to check your work:
    pytest tests/test_client.py -v
All tests should go from FAILED to PASSED as you fill in the TODOs.
"""
import time

import requests

from src.riot_client.config import API_KEY

HEADERS = {"X-Riot-Token": API_KEY}

# 100 requests / 120 seconds = 1 request every 1.2 seconds.
# Spacing calls evenly is the simplest way to never break the 2-minute limit.
MIN_SECONDS_BETWEEN_CALLS = 1.2
MAX_RETRIES = 3

# Remembers when the last request was sent. Starts at 0 = "long ago".
# The leading underscore means "private to this file — don't use from outside".
_last_call_time = 0.0


def _wait_for_rate_limit() -> None:
    """Sleep just long enough so calls are at least MIN_SECONDS_BETWEEN_CALLS apart."""
    global _last_call_time  # lets this function change the variable defined above
    # TODO:
    #   1. now = time.monotonic()   (a clock in seconds that never jumps backwards)
    #   2. elapsed = how many seconds since _last_call_time
    #   3. if elapsed is less than MIN_SECONDS_BETWEEN_CALLS, time.sleep() the difference
    #   4. set _last_call_time to time.monotonic() (the moment we're about to send)
    now = time.monotonic()
    elapsed = now - _last_call_time
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time = time.monotonic()


def get(url: str, params: dict | None = None) -> dict | list:
    """GET a Riot API url and return the parsed JSON, handling rate limits and retries."""
    # TODO: try up to MAX_RETRIES times. Inside a `for attempt in range(MAX_RETRIES):` loop:
    #
    #   1. call _wait_for_rate_limit()
    #   2. response = requests.get(url, headers=HEADERS, params=params, timeout=10)
    #
    #   3. if response.status_code == 429:       (too many requests)
    #         wait = int(response.headers.get("Retry-After", 1))
    #         time.sleep(wait), then `continue` (go to the next attempt)
    #
    #   4. if response.status_code >= 500:       (Riot's server had a problem — usually temporary)
    #         time.sleep(2 ** attempt)           (1s, 2s, 4s... = "exponential backoff")
    #         `continue`
    #
    #   5. otherwise: response.raise_for_status() and return response.json()
    #      (a 404/403 is OUR problem — retrying won't fix it, so we fail immediately)
    #
    # After the loop (all attempts used up), raise an error, e.g.:
    #   raise RuntimeError(f"Gave up on {url} after {MAX_RETRIES} attempts")

   
    for attempt in range(MAX_RETRIES):
        _wait_for_rate_limit()
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 1))
            time.sleep(wait)
            continue
        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Gave up on {url} after {MAX_RETRIES} attempts")

    

