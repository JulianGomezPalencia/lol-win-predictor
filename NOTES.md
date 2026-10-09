# Notes / TODO

Things to fix or improve later. Check items off as they're done.

## Fixes

- [ ] **`client.get()` wastes a sleep on the last attempt** (`src/riot_client/client.py`)
  When the final attempt gets a 429 or 5xx, `get()` still sleeps (Retry-After or backoff)
  before the loop ends and raises `RuntimeError`, so that wait is pointless.
  Idea: only sleep if there's another attempt left (`attempt < MAX_RETRIES - 1`).
  Add a test first: with `MAX_RETRIES` responses of 500, the last one should not trigger a sleep.

## Ideas
