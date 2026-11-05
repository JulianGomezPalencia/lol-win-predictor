# LoL Win Predictor

Software Engineering project at Universidad Industrial de Santander (UIS).

Predicts the probability that a League of Legends ranked match will be won, using each player's recent
history from before the match: their last 10 results, whether they're in their usual lane, and whether
they're playing a champion they're comfortable on. The saved model scores matches that were already played
(see "Training the model").

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

## Training the model

    python -m scripts.export_dataset   # database -> data/export/*.csv (anonymized player IDs)
    python -m src.features.build       # -> data/features/partidas.csv, one row per match
    python -m src.models.train         # baselines, logistic regression, random forest + metrics, also per rank band
    python -m src.models.predict --train          # fit the final model (inferred roles), save data/models/modelo_final.joblib
    python -m src.models.predict LA1_123456       # probability that blue won that (already played) match
    python -m src.models.predict --live "Name#TAG"   # a game in progress, before it ends
    python -m src.models.predict --score-log      # logged live predictions vs the real results
    python -m src.models.predict --evaluate-new   # frozen model vs matches collected after training

**Roles.** Riot assigns `teamPosition` after the game and the live endpoint (spectator-v5) gives no
position. So `src/features/roles.py` infers each team's roles from what is known before the match: the
summoner spells (Smite = jungle) and the positions of the player's earlier games, solved as a
one-to-one assignment. The model is trained on the same inferred roles, so training and prediction use
one method (`build.build(inferred_positions=True)`, which needs `data/export/participant_spells.csv`;
`python -m scripts.export_dataset` writes it from the cached match JSON, with no API calls). The default
`build.build()` still uses Riot's positions, so the notebook and its numbers do not change.

The final model is the logistic regression without the 7 position variables. `--train` needs no Riot
API key; the prediction commands do. A model saved before this change (Riot positions) is refused with a
"retrain with --train" message. If the match was part of the training data the command prints a warning,
because that probability is in-sample. The signal is weak (see NOTES.md).

**`--live` limits.** The player must be in a ranked solo/duo game (queue 420) that has started loading;
otherwise the command says so and stops. A hidden player or one blocked by Riot's firewall (`--` in the
puuid) has no history but is still predicted. Reading the 10 histories takes about 1 to 3 minutes. Each
game adds one row to `data/predictions/live_log.csv` (a game predicted twice is logged once);
`--score-log` later downloads the results of finished games and prints n, accuracy and, with at least 10
games of both outcomes, the ROC AUC. **Do not run collection (`run_pipeline`) and `--live` at the same
time**: each process has its own rate limiter, so together they go over Riot's limit.

**`--evaluate-new`** loads the saved model WITHOUT refitting and scores only matches that are not in its
training set (`bundle["match_ids"]`): the independent test of a frozen model. With no new matches it says so.

## Collecting more data

The discover step asks for the newest `--matches-per-player` (default 5) matches of each player in the
last `--days-back` (default 14) days. Taking only the newest five piles the matches into the last days
(197 and 171 matches on the final two days; the test set covers under one day). Asking for more matches
per player spreads them over the period. `--limit` caps `fetch`, `history` and `snapshot`.

    cp data/db/lol.sqlite data/db/lol_antes_de_ampliar.sqlite   # 1. backup of the database before growing it
    caffeinate -i .venv/bin/python -m scripts.run_pipeline --steps discover --matches-per-player 15 --days-back 14
        # 2. queue more matches (1 call per player); caffeinate keeps the Mac awake
    caffeinate -i .venv/bin/python -m scripts.run_pipeline --steps fetch,history --limit 300
        # 3. download and validate up to 300 queued matches, then their players' histories (the slow part)
    .venv/bin/python -m scripts.export_dataset
        # 4. database -> CSV files, including participant_spells.csv; new matches sort after the old ones
    .venv/bin/python -m src.models.predict --evaluate-new
        # 5. test the frozen model on the new matches

Measured cost (an estimate from the first collection): about 64 calls and 1.3 minutes per match, so 300
matches take about 6.5 hours and 1,000 about 22 hours. Run it as the only process using the API key.

`notebooks/entrega3_modelo.ipynb` runs the same steps with charts (and in Google Colab).

## Project layout

| Folder | What goes there |
|---|---|
| `src/riot_client/` | The only code that talks to the Riot API: rate limiting, retries (up to 30 min of outage), match cache |
| `src/storage/` | SQLite database: `schema.sql` (tables) and `db.py` (queries) |
| `src/agents/` | The five data-collection agents |
| `src/features/` | Pre-game variables per player and per match (blue minus red, lane against lane); `roles.py` infers positions before the game |
| `src/models/` | Chronological split, baselines, models and metrics (`train.py`); final model and prediction (`predict.py`) |
| `scripts/` | Runnable scripts: `run_pipeline.py`, `export_dataset.py`, `hello_riot.py` |
| `tests/` | Automated tests (no real API calls) |
| `data/` | Downloaded matches and the database (not in git) |
| `notebooks/` | Data exploration (`entrega2_eda.ipynb`) and the model (`entrega3_modelo.ipynb`) |

## Roadmap

- [x] Exercise 1: first API calls (`scripts/hello_riot.py`)
- [x] Shared Riot client with rate limiting and caching
- [x] Database + data collection agents
- [x] Full data collection run: 920 matches with the history of all 10 players
- [x] Feature engineering
- [x] Model training (first version: weak signal, see NOTES.md)
- [x] Sensitivity check on the position variables (see NOTES.md)
- [x] Final model + `predict` command for played matches, per-tier check (see NOTES.md)
- [x] Role inference from spells + history; live prediction (`--live`), live log, `--score-log`, `--evaluate-new`
- [ ] More data: a longer period, especially for the test set (recipe above; then `--evaluate-new`)
- [ ] Forward test: run `--live` on real games, then `--score-log`

---
This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone
officially involved in producing or managing Riot Games properties. Riot Games and all associated properties
are trademarks or registered trademarks of Riot Games, Inc.
