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
    now = time.monotonic()
    elapsed = now - _last_call_time
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time = time.monotonic()


def get(url: str, params: dict | None = None) -> dict | list:
    """GET a Riot API url and return the parsed JSON, handling rate limits and retries."""
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

    

