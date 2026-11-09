# LoL Win Predictor

Software Engineering project at Universidad Industrial de Santander (UIS).

Predicts the probability that a League of Legends ranked match will be won, using each player's recent
history from before the match: their last 10 results, whether they're in their usual lane, and whether
they're playing a champion they're comfortable on. The saved 10-variable model also predicts ranked games in progress from pre-game information.

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
    python -m src.models.predict --freeze         # historical, completed ONCE: keep the 22-variable model as the base, fit the 10-variable candidate
    python -m src.models.predict --evaluate-new   # historical, completed: frozen candidate vs base on matches collected after the freeze
    python -m src.models.predict --train          # historical step, completed: refit on 2,973 matches
    python -m src.models.predict --evaluate-final # FUTURE, after collection stops: test the shipped final model, never refit
    python -m src.models.predict LA1_123456       # probability that blue won that (already played) match
    python -m src.models.predict --live "Name#TAG"   # a game in progress, before it ends
    python -m src.models.predict --score-log      # logged live predictions vs the real results

**Current state (2026-11-09):** the first forward test and the refit on 2,973 matches are complete.
Candidate accuracy 56.5 %, AUC 0.577, log loss 0.6847; base 53.4 %, 0.549, 0.6917. The candidate won
by the pre-registered paired log-loss criterion. Those results belong to the candidate fitted on 920,
not to the shipped final model fitted on all 2,973.

**Next order:** collect NEW matches → stop collection → keep the 2026-11-09 export (`cp -R data/export data/export_2026-11-09`; the notebook reproduces the delivered numbers only from it) → `python -m scripts.export_dataset` →
`python -m src.models.predict --evaluate-final`. **Do not run the test while collection is writing the DB.**
No further retraining is planned. The historical `--train` guard remains: it requires a first-test
result in `prueba_final.json` unless `--force` is given. `--freeze` was run once on 2026-11-05 and refuses
another freeze. Do not rerun `--evaluate-new` to test the shipped model.

**Roles.** Riot assigns `teamPosition` after the game and the live endpoint (spectator-v5) gives no
position. So `src/features/roles.py` infers each team's roles from what is known before the match: the
summoner spells (Smite = jungle) and the positions of the player's earlier games, solved as a
one-to-one assignment. The model is trained on the same inferred roles, so training and prediction use
one method (`build.build(inferred_positions=True)`, which needs `data/export/participant_spells.csv`;
`python -m scripts.export_dataset` writes it from the cached match JSON, with no API calls). The default
`build.build()` still uses Riot's positions; the final notebook explicitly requests inferred positions.

The final model is a logistic regression on 10 frozen team-difference variables (`build.FINAL_FEATURES`; the
three newest are mean recent kills, games in the last 24 hours and the champion-in-position pool win rate).
The champion played also counts in role inference (a table made from the frozen pool). `--freeze`,
`--evaluate-new`, `--evaluate-final` and `--train` need no Riot API key; the prediction commands do. A model saved before this
change (22 variables, no pool; it is kept as `modelo_base_22.joblib`) is refused with a
message telling the user to run `--freeze` first and `--train` after the test. If the match was part of the training data the command prints a warning,
because that probability is in-sample. The signal is weak (see NOTES.md).

**`--live` limits.** The player must be in a ranked solo/duo game (queue 420) that has started loading;
otherwise the command says so and stops. A hidden player or one blocked by Riot's firewall (`--` in the
puuid) has no history but is still predicted. Reading the 10 histories takes about 1 to 3 minutes. Each
game adds one row to `data/predictions/live_log.csv` (a game predicted twice is logged once);
`--score-log` later downloads the results of finished games and prints n, accuracy and, with at least 10
games of both outcomes, the ROC AUC. **Do not run collection (`run_pipeline`) and `--live` at the same
time**: each process has its own rate limiter, so together they go over Riot's limit.

**`--evaluate-new`** loads `modelo_base_22.joblib` and `modelo_candidato.joblib` WITHOUT refitting (it never reads
`modelo_final.joblib`) and scores only matches that are not in the 1,206 development matches and were played after
the last fitted one. It prints base, candidate, a 9-variable sensitivity model and a constant, the paired
differences (log loss, AUC, accuracy) and a plain verdict: the candidate is better only if the 95 % interval of
(base minus candidate) log loss is above 0. Each run is appended to `data/models/prueba_final.json`; with no
new matches it says so and writes nothing. Repeating it on the same data adds no evidence.

**`--evaluate-final`** is the second, pre-registered test (NOTES.md, “Segunda prueba, fijada el 2026-11-09”).
It loads `modelo_final.joblib` and the candidate without fitting, using each bundle's own frozen pool
and champion role table. It excludes final training and development IDs and requires start time
strictly after both the bundle's last training match and **2026-11-05 18:00 UTC**. The final model wins
only if the paired 95 % bootstrap interval of constant minus final log loss lies entirely above zero;
the constant is the final training blue-win rate. Candidate metrics are only a drift reference.
Accuracy/AUC intervals, calibration, selective accuracy at confidence ≥ 0.55 / 0.60 / 0.65, dates, tier
counts and exclusion counts are reported. Under 1,000 matches it warns about low power. Runs append
only to `data/models/segunda_prueba.json`; zero test matches write nothing. Earlier run count is
printed; repeats on the same or overlapping data add no independent evidence. This command has been
implemented and tested synthetically; **the real second test has not been run**.

## Collecting more data

The discover step asks for the newest `--matches-per-player` (default 5) matches of each player in the
last `--days-back` (default 14) days. Taking only the newest five piles matches into the last days.
The first independent test spans 2026-10-22 through 2026-11-05. Asking for more matches per player spreads them over the period. `--limit` caps `fetch`, `history` and `snapshot`.

    cp data/db/lol.sqlite data/db/lol_antes_de_ampliar.sqlite   # 1. backup of the database before growing it
    caffeinate -i .venv/bin/python -m scripts.run_pipeline --steps discover --matches-per-player 15 --days-back 14
        # 2. queue more matches (1 call per player); caffeinate keeps the Mac awake
    caffeinate -i .venv/bin/python -m scripts.run_pipeline --steps fetch,history --limit 300
        # 3. download and validate up to 300 queued matches, then their players' histories (the slow part)
    cp -R data/export data/export_2026-11-09
        # 4a. once only: keep the export the notebook reproduces (2,973 matches) before overwriting it
    .venv/bin/python -m scripts.export_dataset
        # 4b. STOP collection first; database -> CSV files, including participant_spells.csv
    .venv/bin/python -m src.models.predict --evaluate-final
        # 5. only after collection stops and export completes: second test, no refit
        # No further --train step is planned; do not repeat export while collection is running.

Measured cost (an estimate from the first collection): about 64 calls and 1.3 minutes per match, so 300
matches take about 6.5 hours and 1,000 about 22 hours. Run it as the only process using the API key.

`notebooks/modelo_final.ipynb` runs end to end locally or in Google Colab, with charts and the
2026-11-09 post-hoc analysis. It refits from CSVs and asserts reproduction of the first test; it needs
no joblib files and never runs either evaluation command or writes test results. Required inputs:
`matches.csv`, `participants.csv`, `player_history.csv`, `participant_spells.csv` from the 2026-11-09
export (2,973 valid matches), plus `conjunto_desarrollo.csv` (920 ajuste / 286 seleccion).
Place them beside the notebook, or in Drive at `lol-win-predictor-data/completo`; locally it also finds
`../data/export` and `../data/models/conjunto_desarrollo.csv`. `segunda_prueba.json` is optional and
only displayed if present. Open in Jupyter and run all cells; Colab supplies the scientific libraries.
**Colab clones `BRANCH = 'main'`; this branch must be merged into main before delivery.** The notebook
uses 200 descriptive bootstrap resamples to target a runtime below roughly 10 minutes. The original
analysis uses 500; this is post-hoc description, not a new model selection or a second test.

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
| `notebooks/` | Data exploration (`entrega2_eda.ipynb`) and the model (`modelo_final.ipynb`) |

## Roadmap

- [x] Exercise 1: first API calls (`scripts/hello_riot.py`)
- [x] Shared Riot client with rate limiting and caching
- [x] Database + data collection agents
- [x] Full data collection run: 920 matches with the history of all 10 players
- [x] Feature engineering
- [x] Model training (first version: weak signal, see NOTES.md)
- [x] Sensitivity check on the position variables (see NOTES.md)
- [x] Final model + `predict` command for played matches, per-tier check (see NOTES.md)
- [x] Role inference from spells + champion + history; live prediction (`--live`), live log, `--score-log`
- [x] Frozen 10-variable candidate (`--freeze`, 2026-11-05) and the forward test (`--evaluate-new`)
- [x] First forward test on 1,767 new matches and final refit on all 2,973
- [x] Second-test command, preregistration and synthetic tests; final notebook and 2026-11-09 analysis
- [ ] Complete new collection, stop, export and run the second test with `--evaluate-final`
- [ ] Merge this branch into main before delivering the Colab notebook
- [ ] Forward test: run `--live` on real games, then `--score-log`

---
This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone
officially involved in producing or managing Riot Games properties. Riot Games and all associated properties
are trademarks or registered trademarks of Riot Games, Inc.
