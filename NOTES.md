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
- **One process at a time** (documented in README): the 1.25 s spacing lives inside each process,
  so two agents running at once would exceed 100 requests / 2 minutes.

- [x] **Riot's firewall blocks some players** (found 2026-10-21). Cloudflare, in front of the API,
  answers an HTML 403 for any URL whose PUUID contains `--` (it looks like an SQL comment), and
  URL-encoding the dashes doesn't help. 111 of 5,602 participants (2 %) are affected. Agents 4 and 5 now
  skip those players before asking (`endpoints.is_firewall_blocked(puuid)`), and print how many
  were skipped. Any other 403 — Riot's own, or the firewall blocking us for another reason —
  still stops the run, so empty histories are never saved by mistake.
  For the model: their history is unknown (not "no games"); add an `is_history_missing` flag.

- [x] **History run crashed on network drops** (found 2026-10-21). `Gave up on .../matches/LA1_...
  after 3 attempts`: the connection was being reset (`Connection reset by peer`), and 3 attempts
  only covered ~3 seconds. `get()` now keeps retrying network errors, 5xx and 429 for up to
  30 minutes (`MAX_OUTAGE_SECONDS`, waits 1, 2, 4 … 60 s), also retries answers cut off or damaged
  in transit, and reads a missing/odd Retry-After safely. If it still fails it raises
  `RiotUnavailableError`, and `run_pipeline` stops with a clear message: just run it again.

## First model results (2026-10-26)

920 matches, 29 blue-minus-red features (team averages + lane against lane). Chronological split
736 train / 184 test; the test set covers less than a day (2026-10-09 23:13 to 2026-10-10 16:34).

| | Test accuracy (95 % CI) | Walk-forward accuracy, 765 preds | Walk-forward ROC AUC (95 % CI) |
|---|---|---|---|
| Always majority class | 53.3 % (46–60) | 51.6 % | — |
| Logistic regression (C=0.01) | 56.0 % (49–63) | 54.6 % | 0.571 (0.532–0.611) |
| Random forest (depth 6) | 52.2 % (45–60) | 53.2 % | 0.559 (0.518–0.597) |

- Weak but real ranking signal: walk-forward AUC 0.571, above 0.5 in 4 of 5 rounds (0.501 in the 4th).
- It does NOT turn into better predictions: logistic minus baseline accuracy, paired on the same
  matches, is +3.0 pp (95 % CI −1.3 to +7.1); log loss and Brier equal a constant prediction.
- The walk-forward's last round overlaps the final test set: more precise, not independent.
- Strongest variables: days since the last game (longer break -> slightly fewer wins), team KDA
  difference, support-vs-support KDA. Some coefficients have the "wrong" sign (support win rate): noise.
- The random forest doesn't beat the logistic regression: too little data to find interactions.
- Review (2026-10-26) items left as documented choices: tier weights only in the fit (weighted
  scoring picks the same hyperparameters); sin_historial can't tell firewall-blocked players from new
  ones (~1 %); prediction assumes champion select is over (champion and position known).

## Ideas

- [ ] **More data.** The matches cover two weeks only. Re-running `--steps discover` later adds
  newer matches (and a longer test period); the history step is the slow part (~110 calls/match).
- [ ] **Rank at match time.** Riot has no historical rank endpoint. A proxy could be the average
  rank of the seed player's league, or LP gained/lost in the history games (not in match-v5).
