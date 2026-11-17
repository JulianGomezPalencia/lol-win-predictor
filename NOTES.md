# Notes / TODO

## Trabajo del 2026-11-16

Añadimos la descarga reanudable de líneas de tiempo, los modelos de estado por
minuto y la interfaz con cuatro pestañas. Preparamos doce ejemplos locales para
recorrer partidas sin credenciales. Conservamos las diez variables y la receta
del modelo previo. Corregimos el informe según la revisión y el cotejo de citas,
separamos su trazabilidad y copiamos las figuras nuevas al directorio de entrega.
Reejecutamos el notebook con el export del 9 de noviembre y añadimos la lectura
de las tablas de estado, remontadas y objetivos.

Los resultados siguientes se conservan en las tablas de `data/analysis/`.
El estado acertó 61,78 % al minuto 5 (IC 95 % de 57,21 a 66,35, n = 416) y
78,16 % al minuto 15 (74,27 a 81,80, n = 412). En las mismas 416 partidas del
minuto 5, el previo acertó 54,57 % (49,52 a 59,62). Comparamos los dos modelos
en las mismas partidas dentro de cada minuto. La población cambió entre minutos.
Añadir el previo al estado no dio una mejora concluyente. Al minuto 10 la reducción
de pérdida fue 0,0052 (−0,0069 a 0,0171).

Contamos 787 remontadas entre 2.780 partidas sin empate al minuto 15,
28,31 % (26,67 a 30,01). El equipo con primera sangre ganó 56,29 %
(56,06 a 56,51, n = 180.130), con primera torre 68,28 %
(68,06 a 68,49, n = 177.993) y con primer dragón 61,57 %
(61,35 a 61,80, n = 179.245). Cuando un mismo equipo obtuvo los tres,
ganó 49.531 de 64.127 partidas, 77,24 % (76,91 a 77,56).
El primer barón se asoció con 79,79 % (79,58 a 80,00, n = 137.665)
y el primer inhibidor con 89,18 % (89,02 a 89,33, n = 159.560).
Todos estos intervalos son del 95 %. Las tasas de objetivos son asociaciones,
con denominadores distintos, y no permiten afirmar efectos causales.

Descargamos 2.820 líneas de tiempo y entrenamos los modelos por minuto con el corte
2026-11-09T18:09:41+00:00. Completamos una hora de vigilancia en vivo. Al corte del
16 de noviembre de 2026 a las 20:22, hora local de Colombia, el registro tenía
33 partidas, 27 terminadas y puntuadas y 6 sin terminar. Acertamos 12 de 27
(44,4 %), con pérdida logarítmica 0,7024 y AUC 0,516. Con confianza de al menos
55 %, hubo 15 partidas y 60,0 % de acierto (Wilson 95 % de 35,7 % a 80,2 %).
La muestra es demasiado pequeña para concluir. La fuente es
`data/predictions/live_log.csv` y la salida de `python -m src.models.predict --score-log`
de ese corte. La batería completa terminó con 325 pruebas aprobadas y la interfaz
cargó sus cuatro pestañas.

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

- [x] **Target matches are unbalanced across tiers** — handled in training: `train.tier_weights` makes every tier count the same in the fit. The dataset itself is still 51 % Master+.  (found after the first discovery run, 954 matches)
  Seeds are balanced (40 players/tier), but high-elo players play much more in 14 days:
  MASTER 160, GRANDMASTER 159, CHALLENGER 149, DIAMOND 129 … BRONZE 42.
  Almost half the dataset comes from Master+, which is ~1% of the real player base.
  Options: cap matches per seed_tier when training (sample equally), and/or seed more
  low-elo players. Check this again before training the model.

## From the code review (2026-10-20)

Reviewer 1 reviewed the whole repo; Reviewer 2 verified each finding against the code.
Confirmed fixes were applied. These were judged real but not urgent:

- [x] **Check history completeness before feature engineering.** Checked 2026-11-04 on the 9,200
  participants of the 920 matches: 9,036 have all 10 rows, 56 have 1 to 9, and 108 have none (they get
  `sin_historial = 1`). The lowest match has 65 of 100 rows.
- [x] **Decide whether remakes count toward the 10 history games.** Decided: collection keeps them in
  their slot (`remake = 1`, 3.5 % of the rows) and `build.player_features` ignores them, so a player
  can have slightly fewer than 10 usable games. `partidas_previas` records how many.
- [ ] **History games only store the player's own side.** By design (see schema.sql), the other 9
  players of a history game aren't in `match_participants`. The full JSON is cached, so this can be
  backfilled if a feature ever needs it. The current 29 variables don't need it.
- [ ] **Champion mastery gap.** Agent 5 snapshots each player once. If a later discovery run adds
  matches where they play a different champion, that champion's mastery is never fetched. The model
  doesn't use mastery: the snapshot reflects the day of the query, not the day of the match.
- [x] **Store why a match is invalid.** New `target_matches.invalid_reason` column; `db.connect` adds it
  to older databases. Only matches rejected from now on get a reason: the 34 existing ones stay NULL
  (the Entrega 2 report counted them: 29 remakes, 5 missing position).
- [x] **More tests:** `save_history` rolling back when an insert fails; a history test with all
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

## Position sensitivity (2026-11-04)

`team_position` is filled in by Riot after the game and the API has no champion-select role, so the
seven position variables might carry information from the match itself. We can't check that directly;
`train.position_sensitivity` repeats the walk-forward (logistic regression) without them:

| | Accuracy | ROC AUC (95 % CI) |
|---|---|---|
| All 29 variables | 54.6 % | 0.571 (0.532–0.611) |
| Without the 7 position variables | 54.6 % | 0.549 (0.507–0.588) |

The paired AUC difference is +0.022 (95 % CI −0.004 to +0.051): not distinguishable from zero, but a
leak can't be ruled out either. Quote 0.549 as the cautious estimate. The remaining lane-against-lane
variables still use `team_position` to pair opponents.

## Per-tier check (2026-11-04)

51 % of the matches come from Master or above, so we checked whether the signal holds lower down.
`train.metrics_by_tier_band` splits the walk-forward predictions (logistic regression, all 29
variables) by `seed_tier`, the tier of the player through whom the match was found, NOT of all ten:

| Band | Matches | Accuracy (95 % CI) | ROC AUC (95 % CI) |
|---|---|---|---|
| Hierro a Platino | 225 | 53.3 % (47.1–60.0) | 0.586 (0.512–0.662) |
| Esmeralda y Diamante | 136 | 54.4 % (45.6–62.5) | 0.550 (0.452–0.651) |
| Maestro o superior | 404 | 55.4 % (50.5–60.4) | 0.570 (0.515–0.625) |

The intervals overlap heavily: the data cannot separate the bands, so we claim no difference.

**These figures are for the 29-variable model, NOT the saved one.** The same check with the 22 variables
of the final model (`train.final_model_tier_check`, no position variables):

| Band | Matches | Accuracy (95 % CI) | ROC AUC (95 % CI) |
|---|---|---|---|
| Hierro a Platino | 225 | 53.3 % (47.1–60.0) | 0.561 (0.487–0.637) |
| Esmeralda y Diamante | 136 | 59.6 % (51.5–67.6) | 0.568 (0.467–0.666) |
| Maestro o superior | 404 | 53.7 % (48.8–58.7) | 0.536 (0.479–0.593) |

Every AUC interval now includes 0.5, so for the final model we cannot say the signal is above chance
in any single band. The conclusion that the bands cannot be separated still holds.

The final model (`python -m src.models.predict --train`, saved in `data/models/`) uses the 22
variables without position (15 of them are still lane-against-lane and use the post-game `team_position` to
pair opponents); temporal ROC AUC 0.549 (0.507–0.588). It scores played matches only:
live prediction was built afterwards, see the next section. The model is fitted on all
920 matches, so predicting one of them is in-sample (AUC 0.608 on its own training rows vs 0.549
walk-forward); the command warns when the match is in the training data.

## Inferred roles and live prediction (2026-11-04)

Riot's `teamPosition` is assigned after the game, and the live endpoint has no position. `src/features/roles.py`
infers roles before the game from summoner spells (Smite = jungle) and the player's earlier positions
(one-to-one assignment, `scipy.optimize.linear_sum_assignment`). The model is retrained on the same inferred
roles (`build.build(inferred_positions=True)`).

**Role accuracy against Riot's `teamPosition`** (920 matches, 9,200 players): overall 81.9 %.
JUNGLE 100.0 %, UTILITY 80.6 %, BOTTOM 77.9 %, TOP 76.8 %, MIDDLE 73.9 %. All five roles of a team are right
in 60.9 % of the team-matches. JUNGLE is exact because Smite is in every jungler's spell set in this data.

**Walk-forward logistic regression, 22 variables, same 765 matches in both runs:**

| Roles | Accuracy (95 % CI) | ROC AUC (95 % CI) |
|---|---|---|
| Riot's (post-game) | 54.6 % (51.1–58.0) | 0.549 (0.507–0.588) |
| Inferred (pre-game) | 54.2 % (50.7–57.6) | 0.546 (0.505–0.585) |

Paired AUC difference (inferred minus Riot): -0.003 (95 % CI -0.018 to +0.012). The data cannot tell the two apart.

**Reading.** The model with inferred roles is the first version free of the post-game position issue, and its
AUC is practically unchanged. The signal is still weak: the interval includes values barely above 0.5. We did not
tune anything to recover the AUC. The saved model (`data/models/modelo_final.joblib`) is now this one; the previous
file (Riot positions) is kept as `data/models/modelo_final_posiciones_riot.joblib`.

`--evaluate-new` on the real data: 0 new matches (as expected: all 920 exported matches are in training). The
independent test needs the collection recipe in the README. Smoke test: one recent ranked match outside the
training data, predicted end to end with the real API (about 110 calls): 39.2 % blue, red won, no player
without history. One match says nothing about accuracy. `--live` was not run against a real player (needs someone
in a game); it is covered by tests with fake endpoints.

## Congelado el 2026-11-05

**Why.** The analysis chose variables using all 1,206 matches, so those matches can no
longer test anything. 920 (2026-09-26 to 10-10) trained the saved model; 286 (10-22 to 11-05) were used by the
analysis. Pending matches are only guaranteed to be after the 920 (the code enforces this by date); the export has no `game_start_ts` for them, so they may overlap in time with the 286, and older ones are excluded by date.

**Frozen.**
- *Development set:* the 1,206 `history_done` ids in `data/export` (listed in `data/models/conjunto_desarrollo.csv`,
  `ajuste` = the 920, `seleccion` = the 286).
- *Base model:* the saved 22-variable logistic regression, copied byte for byte to `modelo_base_22.joblib`.
- *Deviation from the task wording (decision 3):* the task said to train on the fixed list of 1,206; the plan fits the candidate on the 920 only, so base and candidate share training data and the test is strictly forward. The 286 are in the development list but excluded from both fit and test. `--train` later refits on everything.
- *Candidate:* logistic regression (the existing C grid, tier weights) on exactly the same 920 matches with the 10
  variables of `build.FINAL_FEATURES`: `dif_winrate_10`, `dif_partidas_con_campeon`, `dif_racha`,
  `dif_log_dias_desde_ultima`, `dif_kda_medio`, `dif_partidas_previas`, `dif_sin_historial`, `dif_kills_medio`,
  `dif_partidas_24h`, `dif_winrate_campeon_posicion`. Saved as `modelo_candidato.joblib` (and, until the final
  retrain, as `modelo_final.joblib`).
- *Test set:* matches in the export that are not in the 1,206 and were played after the last of the 920.
  The base uses `build.build(inferred_positions=True)` (old roles, no champion); the candidate uses its frozen pool
  and role table.
- *Primary criterion:* the candidate is better only if the 95 % bootstrap interval of the paired log-loss difference
  (base minus candidate) is above 0. AUC and accuracy are reported too.
- *Sensitivity check (not a second chance):* the first 9 variables, fitted the same way, scored on the same matches.
- Comparing with a constant (the training blue win rate) is also printed.

**Champion in role inference.** `roles.CHAMPION_WEIGHT` adds `weight * share` per position, with `share` from the
frozen pool (the 920's history). Chosen ONLY by agreement with Riot's `teamPosition` on the 1,206 matches (12,060
players), never by win results:

| | Overall | TOP | JUNGLE | MIDDLE | BOTTOM | UTILITY |
|---|---|---|---|---|---|---|
| No champion (old) | 81.53 % | 76.4 | 100.0 | 73.8 | 77.4 | 80.1 |
| Weight 0.5 | 93.81 % | 90.5 | 100.0 | 89.1 | 94.1 | 95.4 |
| **Weight 1 (chosen)** | **95.34 %** | 92.7 | 100.0 | 90.8 | 95.8 | 97.4 |
| Weight 2 | 95.27 % | 92.6 | 100.0 | 90.5 | 95.8 | 97.5 |

Cases like the live Pyke-top / Riven-support game: Pyke games agree with Riot in 69.9 % of cases without the
champion and 93.2 % with it (73 games); Riven (49 games) stays at 81.6 % with and without the champion, so it did not change in the aggregate (an earlier draft gave Aatrox's figures, 80.0 % / 96.0 %, 75 games, by mistake). Part of this agreement is optimistic
(the shares come from the same period), and Smite still decides JUNGLE. Base-model comparisons use the old roles
on purpose.

**Champion-by-position is unstable.** In the analysis its univariate AUC was 0.550 before the gap between the two
collections and 0.496 after it (chance); kills kept the same direction in all four time blocks. It stays in the
frozen list because it improved walk-forward results (10 variables AUC 0.565 vs 0.549 for 9), not because it is
proven; the CV gain was not significant (AUC +0.008, p = 0.066). The 9-variable check shows what it costs or adds.

**Deviation from `analysis.py`.** There the overall win rate used as the last prior is one global number; here it is
the pool's rate BEFORE the match start (a pool row at or after `t` must not change anything). The champion and
champion-by-position rates are the same (k = 20).

**Real `--freeze` output (2026-11-05):**

    Ajuste: 920 partidas (10 variables, {'model__C': 0.01}); desarrollo: 1206 partidas; pool: 66344 partidas previas;
    campeones con tabla de posiciones: 173.
    ROC AUC en validación temporal (optimista): 0.564 (IC 95 %: 0.523 a 0.602)

Walk-forward of the candidate with the new roles: **920 matches AUC 0.564 (0.523 to 0.602), accuracy 54.1 %;
1,206 matches AUC 0.551 (0.515 to 0.587), accuracy 53.3 %** (the 1,206 use the 920's pool for the 286). They differ
from the analysis' numbers; the variable set was not changed because of them. These AUCs are optimistic: the
variables were chosen on these matches.

**Checks.** `dif_kills_medio` and `dif_partidas_24h` of the real table match `analysis.py`'s `n_dif_kills` and
`n_dif_games_24h` exactly on all 1,206 matches (max absolute difference 0.0, same NaNs). `--evaluate-new` on the
current export prints 0 test matches and writes nothing (no `prueba_final.json`). The md5 of every file in
`data/export/` is identical to the one taken before the work.

**Stale-pool conservatism.** The test pool is frozen at 2026-10-10, so for the new matches it is older than what the
final model will have. That makes the test conservative for `dif_winrate_campeon_posicion`: a favourable result is
credible, an unfavourable one on that variable partly reflects the stale pool.

**Code clean-up.** Removed `features_for_match` and the old-bundle branch of `in_training_data`; `scripts/export_dataset.py`
now reads every table from one sqlite backup snapshot in a temporary file.

## Ideas

- [ ] **More data.** The matches cover two weeks only. Re-running `--steps discover` later adds
  newer matches (and a longer test period); the history step is the slow part (~110 calls/match).
- [ ] **Rank at match time.** Riot has no historical rank endpoint. A proxy could be the average
  rank of the seed player's league, or LP gained/lost in the history games (not in match-v5).

## Segunda prueba, fijada el 2026-11-09

Prerregistro anterior a cualquier ejecución de `--evaluate-final`. La recolección sigue escribiendo la DB;
**no se ejecutó esta prueba durante la implementación**. Su resultado está pendiente.

- **Modelo:** `data/models/modelo_final.joblib` exactamente como está guardado: 10 variables congeladas,
  ajuste en las 2.973 partidas, C = 0.01. El comando nunca ajusta ni recalibra el predictor. Cada modelo
  usa su propio `pool` y `champion_positions` guardados, mediante `predict._table_for`.
- **Muestra:** partidas `history_done` del export, ajenas tanto a `match_ids` como a `development_ids`
  del bundle final. Inicio estrictamente posterior al `last_match_ts` del bundle y al
  **2026-11-05 18:00 UTC**. El segundo límite queda también en código para no depender de que el máximo
  del export coincida exactamente con esa hora. Se imprimen exclusiones por ajuste, desarrollo (sin
  volver a contar ajuste) y fecha (entre las restantes), ligas y fechas UTC.
- **Criterio primario:** IC 95 % bootstrap emparejado de pérdida logarítmica de la constante menos la
  del final enteramente por encima de cero, con `train.paired_log_loss_difference` (2.000 remuestreos,
  semilla 42). Constante = `bundle['blue_win_rate']` del modelo final. Si el intervalo toca cero,
  no se demuestra mejora. No se cambia el criterio al ver AUC o accuracy.
- **Secundarios, no decisivos:** accuracy y AUC con IC 95 % (2.000 remuestreos), pérdida logarítmica,
  Brier y ECE (10 bins de igual ancho), intercepto y pendiente de recalibración logística conjunta
  con IC bootstrap (500 remuestreos, semilla 42). La recalibración es solo diagnóstico. AUC no
  calculable con una sola clase o predicción constante; intercepto/pendiente no identificables con
  una sola clase o predicción constante se registran como `null`.
- **Selectividad:** confianza = max(p, 1−p) ≥ 0.55 / 0.60 / 0.65; n, cobertura y accuracy con IC Wilson
  95 %. Un subconjunto vacío tiene accuracy e IC indefinidos (`null`). Se informa para final,
  candidato congelado y constante. `modelo_candidato.joblib` es solo referencia de deriva, con su
  propio pool: no compite para sustituir al final.
- **Potencia:** en la primera prueba la ganancia del candidato sobre constante fue
  0.6922 − 0.6847 = 0.0075 en 1.767 partidas. Con menos de 1.000 se imprime aviso de poca potencia;
  ese umbral no garantiza potencia suficiente ni autoriza detener la colección según el resultado.
- **Registro:** cada ejecución no vacía se añade a `data/models/segunda_prueba.json`, formato
  `{"runs": [...]}`; se imprime cuántas ejecuciones anteriores existen. Con cero partidas no se escribe
  nada. Nunca se modifica `prueba_final.json`. Repetir con la misma muestra no añade evidencia;
  sucesivas ejecuciones con muestras superpuestas tampoco son réplicas independientes. No usar
  inspecciones repetidas para decidir cuándo detenerse o cambiar la hipótesis.
- **Después:** no se prevé otro reentrenamiento. El comportamiento de `--train` se conserva por
  compatibilidad; no es el siguiente paso de esta segunda prueba.

Orden autorizado para una ejecución futura: terminar/detener recolección → exportar una vez →
`python -m src.models.predict --evaluate-final`. No ejecutar durante la recolección. El comando lee
solo CSV y bundles, sin DB, API ni clave Riot.

## Análisis del 2026-11-09

Fuente numérica: el análisis del 2026-11-09 y sus comprobaciones.
Es análisis **post-hoc de la primera prueba** (n = 1.767), no una segunda prueba ni selección de un
modelo nuevo. Las métricas reproducen `prueba_final.json`: candidato accuracy 0.564799, AUC 0.577487,
log loss 0.684708; base 0.534239 / 0.549108 / 0.691666; nueve 0.551783 / 0.574274 / 0.684867.

| Confianza mínima del candidato | n | Cobertura | Accuracy (IC 95 % Wilson) |
|---|---:|---:|---:|
| 0.55 | 975 | 55.18 % | 58.36 % (55.24–61.41) |
| 0.60 | 414 | 23.43 % | 60.14 % (55.36–64.75) |
| 0.65 | 140 | 7.92 % | 59.29 % (51.01–67.07) |

Sacrificar cobertura mejora modestamente el acierto. No se eligió un umbral óptimo; los subconjuntos
anidados son pequeños y no permiten comparar modelos como si fueran las mismas partidas.

**Calibración:** candidato Brier 0.245742 (0.241739–0.249314), ECE 0.016218
(0.010299–0.042837), intercepto −0.030477 (−0.133355–0.065950), pendiente 0.766082
(0.507876–1.057078). Base: pendiente 0.513848 (0.248792–0.785763), ECE 0.037954.
La pendiente del candidato incluye 1; no se afirma sobreconfianza concluyente. ECE cambia con los bins:
0.0162 / 0.0128 / 0.0281 con 10 / 15 / 20. No es una propiedad independiente de la discretización;
su sesgo finito impide usar ese IC como prueba de calibración perfecta. Los deciles de frecuencia
igual del gráfico de fiabilidad son distintos de los bins de ancho igual de ECE.

**Factores, asociaciones y no causas:** kills OR por SD 1.25 (1.17–1.36), KDA 1.17 (1.09–1.28),
experiencia con campeón 1.14 (1.07–1.23), tasa campeón/posición 1.09 (1.01–1.18).
Permutar kills/KDA reduce AUC aproximadamente 0.030 / 0.037, con IC positivos en ambos esquemas.
Experiencia con campeón es **limítrofe**: el IC de caída de AUC incluye cero en el esquema original
(−0.000146 a 0.034565), pero lo excluye en el recheck; la pérdida logarítmica sí es positiva en ambos.
El contraste global de quintiles inferior/superior de kills es 39.4 % / 56.5 % de victoria azul y conserva
su dirección en las tres bandas. Los contrastes de KDA/experiencia no son visibles en Esmeralda/Diamante.
El OR ajustado de win rate de 0.86 es condicional a rendimiento, no evidencia de que ganar perjudique.
Racha, pausa, historial y actividad de 24 h tienen IC que incluyen 1; permanecen por congelación.

**Draft:** se ensayó una sola variable, tasa media global del campeón con pool local disponible antes
del inicio. Ganancia CV −0.000065 (p corregido 0.20), ganancia walk-forward −0.000107 en 2.475
predicciones: no retenida. No hay n de potencia para un efecto no positivo. El preregistro se enmendó
antes de calcular resultados para respetar disponibilidad; la versión retrospectiva no decidió la
selección. No se ensayaron balance AD/AP, tanques, melee, matchups ni tasas por parche. El nulo solo
se refiere a esa variable adicional, no al draft en general.

**Límites:** aproximadamente 58 % de la primera prueba es Maestro o superior; la repetición de jugadores
produce clustering no modelado y puede estrechar demasiado los IC por partida. Los bootstrap de
coeficientes son descriptivos, las variables están correlacionadas y el rango actual no es rango
histórico. El notebook reproduce los ajustes desde CSV y usa 200 remuestreos descriptivos para mantener
el tiempo de Colab; sus IC pueden diferir de los 500 del análisis original. No evalúa el final en sus
2.973 partidas como si fueran una prueba nueva.

## Resultado de la segunda prueba (2026-11-11)

Fuente confirmatoria: `data/models/segunda_prueba.json`, una sola ejecución.

```text
Segunda prueba: ejecuciones anteriores 0.
Excluidas por ajuste: 2973; por desarrollo (sin contar ajuste): 0; por fecha: 0.
n = 1264; fechas UTC: 2026-11-05T18:28:21.281000+00:00 a 2026-11-11T05:23:25.606000+00:00
Por liga: BRONZE 38, CHALLENGER 259, DIAMOND 132, EMERALD 47, GOLD 87, GRANDMASTER 205, IRON 130, MASTER 228, PLATINUM 76, SILVER 62
Modelo final (10 variables): accuracy 57.0% (54.4% a 59.8%); ROC AUC 0.589 (0.559 a 0.621); log loss 0.6792
  Calibración: Brier 0.243 (0.239 a 0.247); ECE (10 intervalos) 0.016 (0.010 a 0.049); intercepto -0.059 (-0.176 a 0.056); pendiente 0.997 (0.673 a 1.343)
  Confianza >= 55%: 699 partidas (55.3%); accuracy 59.5% (55.8% a 63.1%)
  Confianza >= 60%: 275 partidas (21.8%); accuracy 65.8% (60.0% a 71.2%)
  Confianza >= 65%: 84 partidas (6.6%); accuracy 67.9% (57.3% a 76.9%)
Candidato congelado (solo referencia de deriva): accuracy 55.9% (53.3% a 58.7%); ROC AUC 0.575 (0.545 a 0.607); log loss 0.6838
  Calibración: Brier 0.245 (0.241 a 0.250); ECE (10 intervalos) 0.010 (0.011 a 0.046); intercepto -0.029 (-0.148 a 0.092); pendiente 0.808 (0.500 a 1.154)
  Confianza >= 55%: 701 partidas (55.5%); accuracy 57.5% (53.8% a 61.1%)
  Confianza >= 60%: 305 partidas (24.1%); accuracy 60.3% (54.7% a 65.7%)
  Confianza >= 65%: 83 partidas (6.6%); accuracy 68.7% (58.1% a 77.6%)
Constante (tasa del entrenamiento): accuracy 52.1% (49.4% a 54.7%); ROC AUC no calculable; log loss 0.6922
  Calibración: Brier 0.250 (0.249 a 0.251); ECE (10 intervalos) 0.002 (0.001 a 0.033); intercepto no calculable; pendiente no calculable
  Confianza >= 55%: 0 partidas (0.0%); sin partidas
  Confianza >= 60%: 0 partidas (0.0%); sin partidas
  Confianza >= 65%: 0 partidas (0.0%); sin partidas
Constante menos modelo final, log loss: +0.0130 (IC 95 %: +0.0042 a +0.0224); positivo = mejor el modelo final.
Candidato: referencia de deriva solamente. El modelo final es mejor que la constante: el IC 95 % de la reducción de pérdida logarítmica queda por encima de 0.
Guardado en data/models/segunda_prueba.json; repetir con los mismos datos no añade evidencia.
```

Registro numérico íntegro (sin redondeo adicional):

```json
{
  "fecha": "2026-11-11T12:17:17+00:00",
  "n": 1264,
  "desde": "2026-11-05T18:28:21.281000+00:00",
  "hasta": "2026-11-11T05:23:25.606000+00:00",
  "por_tier": {
    "BRONZE": 38,
    "CHALLENGER": 259,
    "DIAMOND": 132,
    "EMERALD": 47,
    "GOLD": 87,
    "GRANDMASTER": 205,
    "IRON": 130,
    "MASTER": 228,
    "PLATINUM": 76,
    "SILVER": 62
  },
  "excluidas_desarrollo": 0,
  "excluidas_por_fecha": 0,
  "modelos": {
    "final": {
      "accuracy": 0.5704113924050633,
      "accuracy_ic95": [
        0.5442840189873418,
        0.5981012658227848
      ],
      "auc": 0.5891470923889188,
      "auc_ic95": [
        0.559174678105709,
        0.6214972788481584
      ],
      "log_loss": 0.6792493782542327
    },
    "candidato": {
      "accuracy": 0.5585443037974683,
      "accuracy_ic95": [
        0.5332278481012658,
        0.5870253164556962
      ],
      "auc": 0.5745469594552226,
      "auc_ic95": [
        0.5445300816210316,
        0.6065617847281625
      ],
      "log_loss": 0.6837882692334527
    },
    "constante": {
      "accuracy": 0.5213607594936709,
      "accuracy_ic95": [
        0.49365110759493674,
        0.5474683544303798
      ],
      "auc": null,
      "auc_ic95": null,
      "log_loss": 0.6922454612291382
    }
  },
  "diferencias": {
    "log_loss": {
      "diferencia": 0.012996082974905598,
      "ic95": [
        0.004169021157469503,
        0.022447486534272126
      ]
    }
  },
  "veredicto": "El modelo final es mejor que la constante: el IC 95 % de la reducción de pérdida logarítmica queda por encima de 0.",
  "excluidas_ajuste": 2973,
  "calibracion": {
    "final": {
      "brier": {
        "estimate": 0.24318509382308545,
        "ic95": [
          0.23916480886410382,
          0.24734447085388286
        ]
      },
      "ece": {
        "estimate": 0.01634047594363292,
        "ic95": [
          0.009618699592370914,
          0.0493917183025937
        ]
      },
      "intercept": {
        "estimate": -0.05915697298345244,
        "ic95": [
          -0.1760353617926854,
          0.055807180298598706
        ]
      },
      "slope": {
        "estimate": 0.9971650861052314,
        "ic95": [
          0.6730635370321123,
          1.3433279981680413
        ]
      }
    },
    "candidato": {
      "brier": {
        "estimate": 0.2454000497412234,
        "ic95": [
          0.24097314694445138,
          0.2495769276094647
        ]
      },
      "ece": {
        "estimate": 0.010425258168541047,
        "ic95": [
          0.01108639704653115,
          0.04559537014835847
        ]
      },
      "intercept": {
        "estimate": -0.02851575903090957,
        "ic95": [
          -0.14817121067371636,
          0.09214349603675667
        ]
      },
      "slope": {
        "estimate": 0.8077150964942349,
        "ic95": [
          0.49976458762667775,
          1.1542944024008406
        ]
      }
    },
    "constante": {
      "brier": {
        "estimate": 0.24954927051263381,
        "ic95": [
          0.24852537986388368,
          0.2506317980707838
        ]
      },
      "ece": {
        "estimate": 0.002356386806149857,
        "ic95": [
          0.0007741083251371905,
          0.03285120541625682
        ]
      },
      "intercept": {
        "estimate": null,
        "ic95": null
      },
      "slope": {
        "estimate": null,
        "ic95": null
      }
    }
  },
  "selectiva": {
    "final": [
      {
        "threshold": 0.55,
        "n": 699,
        "coverage": 0.553006329113924,
        "accuracy": 0.5951359084406295,
        "lo": 0.558322613148216,
        "hi": 0.6309092517778547
      },
      {
        "threshold": 0.6,
        "n": 275,
        "coverage": 0.2175632911392405,
        "accuracy": 0.6581818181818182,
        "lo": 0.6002875935206928,
        "hi": 0.7117176598155566
      },
      {
        "threshold": 0.65,
        "n": 84,
        "coverage": 0.06645569620253164,
        "accuracy": 0.6785714285714286,
        "lo": 0.5727856143388975,
        "hi": 0.7687387683701526
      }
    ],
    "candidato": [
      {
        "threshold": 0.55,
        "n": 701,
        "coverage": 0.5545886075949367,
        "accuracy": 0.5748930099857347,
        "lo": 0.537986574764984,
        "hi": 0.6109830959119945
      },
      {
        "threshold": 0.6,
        "n": 305,
        "coverage": 0.2412974683544304,
        "accuracy": 0.6032786885245902,
        "lo": 0.5474179918868822,
        "hi": 0.6565701651454168
      },
      {
        "threshold": 0.65,
        "n": 83,
        "coverage": 0.06566455696202532,
        "accuracy": 0.6867469879518072,
        "lo": 0.5805862952746208,
        "hi": 0.7763860589554854
      }
    ],
    "constante": [
      {
        "threshold": 0.55,
        "n": 0,
        "coverage": 0.0,
        "accuracy": null,
        "lo": null,
        "hi": null
      },
      {
        "threshold": 0.6,
        "n": 0,
        "coverage": 0.0,
        "accuracy": null,
        "lo": null,
        "hi": null
      },
      {
        "threshold": 0.65,
        "n": 0,
        "coverage": 0.0,
        "accuracy": null,
        "lo": null,
        "hi": null
      }
    ]
  },
  "mejor": true,
  "poca_potencia": false,
  "modelo_trained_at": "2026-11-09T18:09:41+00:00"
}
```

## Familias nuevas evaluadas el 2026-11-11

Después de la segunda prueba se evaluaron cuatro familias de cambios, solo con las 2.973 partidas de ajuste (D). Las
1.264 partidas de la segunda prueba (N) no se usaron. La regla se fijó antes de calcular
(plan del 2026-11-11, fase 4a): una familia entra si su ganancia media de pérdida logarítmica en validación
cruzada (3 × 5 folds estratificados, semilla 42) es mayor que 0 con p corregido de Nadeau-Bengio menor que 0,10, y
además su ganancia en la validación hacia adelante (5 bloques) es mayor que 0. Se evalúa la familia entera, sin elegir
variables dentro. Ganancia = pérdida de la referencia (logística de 10 variables) − pérdida de la familia.

| Familia | Qué cambia | Ganancia CV | p corregido | Ganancia hacia adelante | Entra |
| --- | --- | --- | --- | --- | --- |
| F1 Premade | + `dif_parejas_premade` (parejas del equipo con ≥ 2 partidas previas compartidas) | −0,000249 | 0,204 | −0,000138 | no |
| F2 Composición | + `dif_tanques`, `dif_a_distancia`, `dif_desbalance_dano` | −0,000748 | 0,313 | −0,000010 | no |
| F3 Por rol | kills y KDA medios por rol (10 variables) en lugar de las dos medias de equipo | −0,001254 | 0,554 | −0,004355 | no |
| F4 No lineal | HistGradientBoosting (profundidad 2) con las mismas 10 variables | −0,004489 | 0,022 (peor) | −0,004572 | no |

**No hay retador; no hay tercera prueba.** Ninguna familia entra, así que no se congeló ningún modelo nuevo y N no se
volvió a tocar. El programa sigue con el modelo final de 10 variables.

Notas:
- F2 usa `champion.json` versión 16.22.1 tomado de un espejo público en jsdelivr
  (`InFinity54/LoL_DDragon`), porque esta máquina no pudo conectar con el CDN de Riot. Copia y sha256 en
  `data/static/`. Las etiquetas son las del parche 16.22.1 para todas las partidas.
- Las cuatro familias se calcularon en el mismo script, con los mismos folds. F2 se añadió después de
  una enmienda al plan escrita antes de ver resultados de F2.
- No se probaron enfrentamientos de línea (0 pares con ≥ 30 partidas en cuatro roles, 1 en BOTTOM), tasas de campeón
  por parche ni rango en el momento de la partida (el rango consultado después incluye el resultado).
- Los resultados por fold se resumen arriba.


## Programa: confianza, abstención y vigilancia (2026-11-11)

La salida usa la probabilidad original del modelo. No se recalibra: la ganancia de pérdida
logarítmica fue 0,000293, IC 95 % −0,000766 a 0,001344, insuficiente para justificarlo.
La confianza es `max(p, 1-p)`; se abstiene por debajo de 0,55. Desde 0,55 / 0,60 / 0,65
los niveles son baja / media / alta. En la segunda prueba, esos umbrales tuvieron 699 / 275 / 84
partidas, coberturas 55,30 / 21,76 / 6,65 % y aciertos 59,51 / 65,82 / 67,86 %.
Las cifras históricas describen grupos seleccionados, no una garantía individual.

El reajuste conserva una copia fechada del bundle anterior, verifica SHA-256 y nunca pisa una
copia distinta. Se mantienen la comprobación de la primera prueba y, para el modelo actual,
la segunda prueba por `trained_at`, salvo `--force`. `modelo_probado` conserva fecha de ajuste,
número de partidas, fecha y tamaño de la segunda prueba, accuracy, AUC, pérdida logarítmica y
selectividad final. El nuevo reajuste no tiene prueba independiente propia: la salida atribuye
esas cifras a la versión probada, sin presentarlas como validación del reajuste. Se conserva
la versión 2 del bundle porque el campo es opcional y los bundles anteriores siguen funcionando.

`--watch` lee semillas en modo solo lectura, comparte con `--live` la ruta desde el objeto de
partida y evita partidas ya registradas. Omite cortafuegos, ausencia de partida, otra cola y
partidas incompletas; cuenta fallos de red y de predicción por separado y continúa. Repite el
recorrido con pausa de hasta 30 segundos y termina por tiempo, máximo de partidas o Ctrl+C.
El plazo se propaga al cliente HTTP y limita también reintentos, esperas y timeout de petición.
No debe convivir con el recolector.

El contexto de rango y maestría se consulta mientras transcurre la partida, sin su resultado.
Se registra una vez por partida en `live_context.csv`, con campos vacíos si no están disponibles,
sin nombres ni identificadores de jugadores. No entra en el modelo; las diez variables y la
receta siguen intactas. `live_log.csv` mantiene sus columnas. `--score-log` añade pérdida
logarítmica, abstenciones y selectividad con Wilson 95 % para los tres umbrales; advierte con
menos de veinte resultados. Las evaluaciones congeladas no se modificaron.

### Implementación y comprobación

- `src/models/predict.py`: confianza, copia segura y procedencia del reajuste, ruta de predicción
  compartida, vigilancia, contexto anónimo y métricas del registro.
- `tests/test_predict.py`: bordes simétricos de confianza, procedencia histórica, copia idéntica
  o conflictiva, exigencia de segunda prueba, contexto y privacidad, vigilancia y puntuación.
  Los endpoints adicionales de rango y maestría están simulados también en las pruebas previas.
  Se actualizaron las expectativas antiguas de entrenamiento para exigir copia y segunda prueba.
- `README.md`: comandos, interpretación y estado actualizado; esta sección explica las decisiones.
- No se ejecutaron comandos reales de entrenamiento, evaluación o vigilancia; no se llamó a Riot
  ni se modificó `data/`, notebooks o entregables. La batería completa se reserva para la
  comprobación automática solicitada. Se preservó el contenido previo de estas notas.
- Desviación del proceso: antes de leer la regla «sin git» se ejecutó por error únicamente
  `git status --short` (lectura). No hubo otras operaciones de git, commits ni pushes.

Validación de esta implementación:

- `.venv/bin/python -m pytest -q tests/test_predict.py tests/test_live_and_pipeline.py`:
  **98 aprobadas**, 64 avisos de scikit-learn debidos a folds sintéticos con una sola clase
  (scoring no calculable y puntuaciones no finitas). Sin errores de prueba.
- Tras precisar que un umbral sin resultados conserva `n_historico = 0` y
  `acierto_historico = None`, sin heredar cifras de otro umbral:
  `.venv/bin/python -m pytest -q tests/test_predict.py -k 'confidence_levels_and_exact_edges or confidence_highest_threshold_without_outcomes'`:
  **17 aprobadas**, 75 deseleccionadas. Estas 17 incluyen 16 casos anteriores y un caso nuevo;
  no se suman a las 98 como si fueran pruebas distintas (99 casos distintos cubiertos).
- El primer intento tuvo un fallo en el reloj simulado de la prueba de vigilancia; se corrigió
  y quedó incluido en la ejecución aprobada. La comprobación adicional de sintaxis y ausencia
  de definiciones duplicadas también pasó. No se ejecutó la batería completa, por instrucción
  expresa del solicitante; queda a cargo de la comprobación automática.
- No queda ningún apartado de implementación pendiente del plan. El entrenamiento y la
  vigilancia reales posteriores quedaron para una sesión posterior.

### Correcciones de revisión de código (2026-11-11)

Se verificaron y corrigieron los tres hallazgos de la revisión de código:
contexto perdido tras registrar la predicción, plazo que no alcanzaba los reintentos y errores
de predicción que terminaban la vigilancia. El contexto se publica de forma atómica antes del
registro y ambos modos reparan contextos ausentes o parciales mientras siga en curso la partida.
El plazo alcanza las consultas y las esperas del cliente HTTP; al agotarlo se conserva el resumen.
Los errores de predicción se cuentan en `errores_prediccion`, sin imprimir sus mensajes, y
permiten continuar. `KeyboardInterrupt` mantiene su parada propia.

La documentación de vigilancia anterior se actualizó para
reflejar estas correcciones. No se ejecuta la batería completa en este paso.

Validación final de las correcciones:

- `.venv/bin/python -m pytest -q tests/test_client.py tests/test_predict.py -k 'client or live or context or watch or seed_database' --tb=short`:
  **54 aprobadas, 73 deseleccionadas**, en 24,30 s. Incluye 20 casos nuevos de regresión.
- Los 64 avisos de scikit-learn proceden de folds sintéticos con una sola clase y puntuaciones
  no finitas durante la preparación del modelo de pruebas; no hubo fallos en el pase final.
- Solo pruebas pertinentes a los cambios. La batería completa queda para la comprobación
  automática, tal como se solicitó. Sin llamadas reales a Riot ni escrituras en `data/`.
