# Notes / TODO

Things to fix or improve later. Check items off as they're done.

## Fixes

- [x] **`client.get()` wastes a sleep on the last attempt** — fixed, with a test. `get()` now also retries network errors (Wi-Fi drops, timeouts). (`src/riot_client/client.py`)
  When the final attempt gets a 429 or 5xx, `get()` still sleeps (Retry-After or backoff)
  before the loop ends and raises `RuntimeError`, so that wait is pointless.
  Idea: only sleep if there's another attempt left (`attempt < MAX_RETRIES - 1`).
  Add a test first: with `MAX_RETRIES` responses of 500, the last one should not trigger a sleep.

## Data design

- [x] **Unranked match participants** — done in Agent 5: `player_snapshots` rank columns allow NULL. (still in placements / no ranked games this season)
  Seed players are always ranked (league-v4 only lists ranked players), so `players.lp NOT NULL`
  is correct — checked: 400 players, 0 nulls, 18 with lp = 0 (valid, just placed / demoted).
  But the other 9 players in each match may be unranked: league-v4 by-puuid returns `[]` for them.
  When we store participants' rank/LP (Agent 4), those columns must allow NULL.
  For the model, consider an `is_unranked` feature (1/0) — being unranked is itself a signal
  (placements, returning player, smurf).

- [ ] **Target matches are unbalanced across tiers** (found after the first discovery run, 954 matches)
  Seeds are balanced (40 players/tier), but high-elo players play much more in 14 days:
  MASTER 160, GRANDMASTER 159, CHALLENGER 149, DIAMOND 129 … BRONZE 42.
  Almost half the dataset comes from Master+, which is ~1% of the real player base.
  Options: cap matches per seed_tier when training (sample equally), and/or seed more
  low-elo players. Check this again before training the model.

## From the code review (2026-10-20)

Reviewer 1 reviewed the whole repo; Reviewer 2 verified each finding against the code.
Confirmed fixes were applied. These were judged real but not urgent:

- [ ] **Check history completeness before feature engineering.** A target is marked `history_done`
  whatever number of rows it got. Fewer than 10 per player is legitimate (new accounts), but run:
  `SELECT target_match_id, COUNT(*) FROM player_history GROUP BY 1 ORDER BY 2` and look at the low end.
- [ ] **Decide whether remakes count toward the 10 history games.** Right now a remake takes a slot
  (it's stored with `remake = 1`). Option: request more than 10 IDs and keep the first 10 non-remakes.
- [ ] **History games only store the player's own side.** By design (see schema.sql), the other 9
  players of a history game aren't in `match_participants`. The full JSON is cached, so this can be
  backfilled if a feature ever needs it.
- [ ] **Champion mastery gap.** Agent 5 snapshots each player once. If a later discovery run adds
  matches where they play a different champion, that champion's mastery is never fetched.
- [ ] **Store why a match is invalid.** `mark_match_invalid` drops the reason (remake / wrong queue /
  missing position / not found). Needs a new column: `CREATE TABLE IF NOT EXISTS` won't add it to
  the existing database, so it needs an `ALTER TABLE target_matches ADD COLUMN invalid_reason TEXT`.
- [ ] **More tests:** `save_history` rolling back when an insert fails; a history test with all
  10 participants instead of 1.
- **One process at a time** (documented in README): the 1.2 s spacing lives inside each process,
  so two agents running at once would exceed 100 requests / 2 minutes.

- [x] **Riot's firewall blocks some players** (found 2026-10-21). Cloudflare, in front of the API,
  answers an HTML 403 for any URL whose PUUID contains `--` (it looks like an SQL comment), and
  URL-encoding the dashes doesn't help. 111 of 5,602 participants (2 %) are affected. Agents 4 and 5 now
  skip those players (no history / no snapshot); a 403 from Riot itself still stops the run.
  For the model: treat them like players with no previous games.

## Ideas
