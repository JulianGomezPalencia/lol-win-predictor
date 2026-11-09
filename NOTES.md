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
