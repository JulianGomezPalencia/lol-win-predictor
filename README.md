# LoL Win Predictor

Software Engineering project at Universidad Industrial de Santander (UIS).

Predicts the probability that a League of Legends ranked match will be won **before it starts**, using
each player's recent history: their last 10 results, whether they're in their usual lane, and whether
they're playing a champion they're comfortable on.

## Setup

```bash
python3 -m venv .venv            # create an isolated Python environment
source .venv/bin/activate        # activate it (do this every time you open a new terminal)
pip install -r requirements.txt  # install the libraries
cp .env.example .env             # then paste your Riot API key into .env
```

## Collecting the data

```bash
python -m scripts.run_pipeline --steps seed,discover   # once: pick players and their matches
python -m scripts.run_pipeline                         # fetch, history, snapshot (resumable)
python -m scripts.run_pipeline --steps history --limit 50   # only some steps, capped
pytest                                                 # run all tests
```

Each agent can also run alone, e.g. `python -m src.agents.match_fetcher`. The default command only
picks up the work that's still left, so if a run stops (Ctrl+C, crash, laptop asleep), run it again.
`seed` and `discover` aren't in the default because re-running them grows the dataset instead of resuming it.

Run **one agent at a time**: the rate limiter lives inside each process, so two running together
would go over Riot's limit (100 requests / 2 minutes).

| # | Agent | What it does | Writes to |
|---|---|---|---|
| 1 | `seeder` | Samples 40 players from every tier, Iron → Challenger | `players` |
| 2 | `match_discovery` | Their ranked matches from the last 14 days → the job queue | `target_matches` (`pending`) |
| 3 | `match_fetcher` | Downloads each match, drops remakes / wrong queue / missing lanes | `match_participants`, status `fetched` or `invalid` |
| 4 | `history_builder` | Each player's last 10 ranked games **before** the match (no future leakage) | `player_history`, status `history_done` |
| 5 | `player_snapshot` | Today's rank + champion mastery of each participant (NULL rank if unranked) | `player_snapshots`, `champion_mastery` |

## Project layout

| Folder | What goes there |
|---|---|
| `src/riot_client/` | The only code that talks to the Riot API: rate limiting, retries (up to 30 min of outage), match cache |
| `src/storage/` | SQLite database: `schema.sql` (tables) and `db.py` (queries) |
| `src/agents/` | The five data-collection agents |
| `scripts/` | Runnable scripts: `run_pipeline.py`, `hello_riot.py` |
| `tests/` | Automated tests (no real API calls) |
| `data/` | Downloaded matches and the database (not in git) |
| `notebooks/` | Data exploration |

## Roadmap

- [x] Exercise 1: first API calls (`scripts/hello_riot.py`)
- [x] Shared Riot client with rate limiting and caching
- [x] Database + data collection agents
- [ ] Full data collection run (history step takes the longest)
- [ ] Feature engineering
- [ ] Model training

---
This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone
officially involved in producing or managing Riot Games properties. Riot Games and all associated properties
are trademarks or registered trademarks of Riot Games, Inc.
