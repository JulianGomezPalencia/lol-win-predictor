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
cp .env.example .env             # optional, only for collection and live queries
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

Run one agent at a time: the rate limiter lives inside each process, so two running together
would go over Riot's limit (100 requests / 2 minutes).

| # | Agent | What it does | Writes to |
|---|---|---|---|
| 1 | `seeder` | Samples 40 players from every tier, Iron → Challenger | `players` |
| 2 | `match_discovery` | Their ranked matches from the last 14 days → the job queue | `target_matches` (`pending`) |
| 3 | `match_fetcher` | Downloads each match, drops remakes / wrong queue / missing lanes | `match_participants`, status `fetched` or `invalid` |
| 4 | `history_builder` | Each player's last 10 ranked games before the match (no future leakage) | `player_history`, status `history_done` |
| 5 | `player_snapshot` | Today's rank + champion mastery of each participant (NULL rank if unranked) | `player_snapshots`, `champion_mastery` |

## Training the model

    python -m scripts.export_dataset   # database -> data/export/*.csv (anonymized player IDs)
    python -m src.features.build       # -> data/features/partidas.csv, one row per match
    python -m src.models.train         # baselines, logistic regression, random forest + metrics, also per rank band
    python -m src.models.predict --freeze         # historical, completed ONCE: keep the 22-variable model as the base, fit the 10-variable candidate
    python -m src.models.predict --evaluate-new   # historical, completed: frozen candidate vs base on matches collected after the freeze
    python -m src.models.predict --train          # refit after both tests. keep a dated copy of the current bundle
    python -m src.models.predict --evaluate-final # second test, completed. never refits
    python -m src.models.predict LA1_123456       # probability that blue won that (already played) match
    python -m src.models.predict --live "Name#TAG"   # a game in progress, before it ends
    python -m src.models.predict --watch --watch-minutes 60 --max-games 20  # watch local seeds, collection stopped
    python -m src.models.predict --score-log      # results, log loss, coverage and Wilson intervals

Estado (2026-11-11): la segunda prueba del modelo final pasó: 1.264 partidas,
accuracy 57,04 % y AUC 0,5891. Se conserva la probabilidad sin recalibrar y las diez variables.
Las cuatro familias ensayadas (premade, composición, por rol y no lineal) quedaron fuera. Tabla y resultados en [NOTES.md](NOTES.md#familias-nuevas-evaluadas-el-2026-11-11).

Reajuste seguro. `--train` exige `prueba_final.json` y una ejecución en `segunda_prueba.json`
con el `trained_at` del modelo actual. `--force` permite omitir las comprobaciones con aviso.
Antes de sustituirlo, conserva `modelo_final_AAAA-MM-DD.joblib` (fecha de su ajuste) y verifica
su SHA-256. Si esa copia existe con contenido distinto, se detiene sin sobrescribirla.
El bundle nuevo conserva `modelo_probado` con las métricas y selectividad de la versión probada.
El reajuste con más partidas no tiene prueba independiente propia. Los bundles anteriores
sin ese campo siguen cargando. No repetir `--freeze` ni `--evaluate-new` para evaluar un reajuste.

Confianza y abstención. Tanto las partidas jugadas como las predicciones en vivo muestran
el favorito y confianza `max(p, 1-p)`: alta desde 65 %, media desde 60 % y baja desde 55 %.
Por debajo de 55 % se muestra «Partida pareja: sin predicción». Ejemplo:
`Predicción: gana el equipo azul. Confianza media (62,3 %).`
Cuando está disponible, se añade el acierto histórico del umbral alcanzado en la segunda prueba,
para un reajuste se identifica explícitamente como el de la versión probada del modelo.
La confianza no equivale a una garantía de acierto de esa partida.

Vigilancia. `--watch` recorre las semillas de SQLite en modo solo lectura, vuelve a recorrerlas
cada 30 segundos y predice partidas de cola 420 con diez jugadores que no estén ya en el registro.
Se detiene al alcanzar el tiempo (60 minutos por defecto), el máximo de partidas nuevas (20),
o Ctrl+C y muestra revisados, predichas y omisiones por motivo. Los fallos de red y de predicción
se cuentan por separado y permiten continuar. El plazo limita también las consultas, sus
reintentos y esperas. El timeout de cada petición se reduce al tiempo restante.
No ejecutar junto con la recolección: cada proceso tiene su propio limitador.
La salida contiene campeones y posiciones inferidas, nunca nombres ni identificadores de jugador.

En `--live` y `--watch`, `live_context.csv` guarda una fila por jugador con rango, LP, resultados
previos, maestría y segundos transcurridos. No incluye identificadores personales ni el resultado
de la partida actual y no modifica las variables del modelo. Los campos fallidos u ocultos quedan
vacíos. Añade unas veinte consultas por partida. Las diez filas se publican juntas antes de
registrar la predicción. Ambos modos reparan contextos ausentes o parciales de partidas todavía
en curso, sin duplicar la predicción. El registro principal mantiene sus columnas.
`--score-log` informa pérdida logarítmica, abstenciones y, para 55/60/65 %, n, cobertura y aciertos
con IC de Wilson 95 %. Con menos de veinte partidas avisa de que la muestra es insuficiente.
`--score-log` descarga las partidas terminadas que faltan en caché, por lo que cuenta como un proceso con red.

Roles. Riot assigns `teamPosition` after the game and the live endpoint (spectator-v5) gives no
position. So `src/features/roles.py` infers each team's roles from what is known before the match: the
summoner spells (Smite = jungle) and the positions of the player's earlier games, solved as a
one-to-one assignment. The model is trained on the same inferred roles, so training and prediction use
one method (`build.build(inferred_positions=True)`, which needs `data/export/participant_spells.csv`,
`python -m scripts.export_dataset` writes it from the cached match JSON, with no API calls). The default
`build.build()` still uses Riot's positions. The final notebook explicitly requests inferred positions.

The final model is a logistic regression on 10 frozen team-difference variables (`build.FINAL_FEATURES`). The
three newest are mean recent kills, games in the last 24 hours and the champion-in-position pool win rate.
The champion played also counts in role inference (a table made from the frozen pool). `--freeze`,
`--evaluate-new`, `--evaluate-final` and `--train` need no Riot API key. The prediction commands do. A model saved before this
change (22 variables, no pool), kept as `modelo_base_22.joblib`, is refused with a
message telling the user to run `--freeze` first and `--train` after the test. If the match was part of the training data the command prints a warning,
because that probability is in-sample. The signal is weak (see NOTES.md).

`--live` limits. The player must be in a ranked solo/duo game (queue 420) that has started loading,
otherwise the command says so and stops. A hidden player or one blocked by Riot's firewall (`--` in the
puuid) has no history but is still predicted. Reading the 10 histories takes about 1 to 3 minutes. Each
game adds one row to `data/predictions/live_log.csv` (a game predicted twice is logged once),
`--score-log` later downloads the results of finished games and prints n, accuracy and, with at least 10
games of both outcomes, the ROC AUC. Do not run collection (`run_pipeline`) and `--live` at the same
time: each process has its own rate limiter, so together they go over Riot's limit.

`--evaluate-new` loads `modelo_base_22.joblib` and `modelo_candidato.joblib` WITHOUT refitting (it never reads
`modelo_final.joblib`) and scores only matches that are not in the 1,206 development matches and were played after
the last fitted one. It prints base, candidate, a 9-variable sensitivity model and a constant, the paired
differences (log loss, AUC, accuracy) and a plain verdict: the candidate is better only if the 95 % interval of
(base minus candidate) log loss is above 0. Each run is appended to `data/models/prueba_final.json`. With no
new matches it says so and writes nothing. Repeating it on the same data adds no evidence.

`--evaluate-final` is the second, pre-registered test (NOTES.md, “Segunda prueba, fijada el 2026-11-09”).
It loads `modelo_final.joblib` and the candidate without fitting, using each bundle's own frozen pool
and champion role table. It excludes final training and development IDs and requires start time
strictly after both the bundle's last training match and 2026-11-05 18:00 UTC. The final model wins
only if the paired 95 % bootstrap interval of constant minus final log loss lies entirely above zero,
the constant is the final training blue-win rate. Candidate metrics are only a drift reference.
Accuracy/AUC intervals, calibration, selective accuracy at confidence ≥ 0.55 / 0.60 / 0.65, dates, tier
counts and exclusion counts are reported. Under 1,000 matches it warns about low power. Runs append
only to `data/models/segunda_prueba.json`. Zero test matches write nothing. Earlier run count is
printed. Repeats on the same or overlapping data add no independent evidence. Completamos este comando sobre 1.264 partidas. El resultado está en `data/models/segunda_prueba.json`. El reajuste posterior con 4.237 partidas sigue sin una prueba independiente propia.

## Collecting more data

The discover step asks for the newest `--matches-per-player` (default 5) matches of each player in the
last `--days-back` (default 14) days. Taking only the newest five piles matches into the last days.
The first independent test spans 2026-10-22 through 2026-11-05. Asking for more matches per player spreads them over the period. `--limit` caps `fetch`, `history` and `snapshot`.

    cp data/db/lol.sqlite data/db/lol_antes_de_ampliar.sqlite   # 1. backup of the database before growing it
    caffeinate -i .venv/bin/python -m scripts.run_pipeline --steps discover --matches-per-player 15 --days-back 14
        # 2. queue more matches (1 call per player). caffeinate keeps the Mac awake
    caffeinate -i .venv/bin/python -m scripts.run_pipeline --steps fetch,history --limit 300
        # 3. download and validate up to 300 queued matches, then their players' histories (the slow part)
    cp -R data/export data/export_2026-11-09
        # 4a. once only: keep the export the notebook reproduces (2,973 matches) before overwriting it
    .venv/bin/python -m scripts.export_dataset
        # 4b. STOP collection first. database -> CSV files, including participant_spells.csv
    .venv/bin/python -m src.models.predict --evaluate-final
        # 5. only after collection stops and export completes: second test, no refit
        # Historical collection sequence. do not repeat export while collection is running.

Measured cost (an estimate from the first collection): about 64 calls and 1.3 minutes per match, so 300
matches take about 6.5 hours and 1,000 about 22 hours. Run it as the only process using the API key.

## Notebook del informe

Reejecutamos `notebooks/modelo_final.ipynb` con Python 3.14.3 y los CSV de
`data/export_2026-11-09/`. Necesita `matches.csv`, `participants.csv`,
`player_history.csv`, `participant_spells.csv` y
`data/models/conjunto_desarrollo.csv` (920 de ajuste y 286 de selección).
Reconstruye la primera prueba y muestra la segunda desde
`data/models/segunda_prueba.json`. No guarda modelos ni modifica los registros
de prueba. Usa 200 remuestras para el análisis descriptivo, frente a las 500 del
análisis original.

La sección de estado por minuto lee `predictibilidad.csv`, `remontadas.csv` y `objetivos.csv`
de `data/analysis/`. La curva compara previo y estado en las mismas partidas
de cada minuto y separa el punto previo de la segunda prueba completa.
En Colab hay que proporcionar también estas tablas y el JSON de la segunda prueba.
El notebook clona la rama `main` cuando el repositorio no está disponible.
Para ver los cambios de esta entrega, esa rama debe contenerlos.

## Project layout

| Folder | What goes there |
|---|---|
| `src/riot_client/` | The only code that talks to the Riot API: rate limiting, retries (up to 30 min of outage), match cache |
| `src/storage/` | SQLite database: `schema.sql` (tables) and `db.py` (queries) |
| `src/agents/` | The five data-collection agents |
| `src/features/` | Pre-game variables per player and per match (blue minus red, lane against lane), `roles.py` infers positions before the game |
| `src/models/` | Chronological split, baselines, models and metrics (`train.py`). Final model and prediction (`predict.py`) |
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
- [x] Role inference from spells + champion + history. Live prediction (`--live`), live log, `--score-log`
- [x] Frozen 10-variable candidate (`--freeze`, 2026-11-05) and the forward test (`--evaluate-new`)
- [x] First forward test on 1,767 new matches and final refit on all 2,973
- [x] Second-test command, preregistration and synthetic tests. Final notebook and 2026-11-09 analysis
- [x] Complete new collection, stop, export and run the second test with `--evaluate-final`
- [x] Confidence and abstention, safe refit, watch, anonymous live context and selective log scoring
- [ ] Merge this branch into main before delivering the Colab notebook
- [x] Completamos una hora de vigilancia en vivo y puntuamos el registro al corte del 16 de noviembre de 2026 a las 20:22, hora local de Colombia.

---
This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone
officially involved in producing or managing Riot Games properties. Riot Games and all associated properties
are trademarks or registered trademarks of Riot Games, Inc.

## Descarga de líneas de tiempo

Guardamos las respuestas completas en `data/raw/timelines/`. El descargador lee
las partidas de la base local y omite las líneas de tiempo que ya están en caché.
Necesita las credenciales de Riot en `.env`.

```bash
.venv/bin/python scripts/download_timelines.py --max 2800 --minutes 75
```

`--max` limita las descargas nuevas exitosas. `--minutes` limita el tiempo de la
corrida, incluidas consultas y esperas. Una operación local o respuesta ya iniciada
puede terminar después. Podemos repetir el comando para continuar con lo pendiente.
Lo ejecutamos como único proceso que consulta Riot. Antes hay que detener la
recolección y la vigilancia. No se debe usar la consulta en vivo mientras descarga.
Descargamos 2.820 líneas de tiempo.

## Modelos por minuto

Entrenamos el estado con las líneas de tiempo locales, separando por fecha de inicio
con zona horaria y el corte del 2026-11-09T18:09:41+00:00. La ayuda se puede consultar sin descargar ni entrenar.

```bash
.venv/bin/python -m src.models.ingame --help
.venv/bin/python -m src.models.ingame --train --before 2026-11-09T18:09:41+00:00
```

Para una evaluación separada existe el siguiente comando. La evaluación confirmatoria
ya está completada y no debe repetirse sobre las mismas partidas.

```bash
.venv/bin/python -m src.models.ingame --evaluate --after 2026-11-09T18:09:41+00:00
```

Estos comandos leen datos locales y no descargan partidas. Los modelos de estado
no cambian las diez variables ni la receta del modelo previo.

## Interfaz local

Añadimos cuatro pestañas para consultar una partida en vivo, repetir una partida,
probar diferencias de juego y leer los resultados. La abrimos desde la raíz.

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.app
```

La página se abre en `http://127.0.0.1:8501`. En la comprobación del 16 de noviembre la interfaz cargó sus cuatro pestañas y
la batería completa terminó con 325 pruebas aprobadas. La consulta
en vivo tarda varios minutos y usa credenciales de Riot. El dato escrito para buscar
al jugador no se guarda. Esperamos a que termine cualquier descarga antes de consultar.

## Modo de demostración

Abrimos la misma interfaz y elegimos Repetición para recorrer los ejemplos de
`data/app/ejemplos.json` o, si falta, `demo/ejemplos.json`. No hay un comando
ni un interruptor de demo aparte.
Los ejemplos guardados funcionan sin credenciales ni partida activa. En Qué aprendimos
podemos leer las tablas y figuras locales. La calculadora necesita los modelos guardados,
deja en cero las variables sin control visible y usa el modelo del minuto 30 para los
minutos posteriores. Si falta un archivo, la pantalla informa de su ausencia.

Para generar ejemplos con `scripts/build_demo_examples.py` hay que proporcionar
un CSV de candidatas con `match_id`, `categoria` y `prob_previa`. El script elige con semilla 42 cuatro aciertos con confianza, dos
fallos con confianza, tres remontadas y tres partidas al azar, sin repetir partidas.
Las categorías se calculan en la corrida de análisis. `scripts/build_demo_examples.py`
necesita que se le proporcione ese CSV mediante `--candidates RUTA_AL_CSV`, además de
las partidas y líneas de tiempo en caché y `data/models/modelos_en_partida.joblib`.
No genera el CSV de candidatas.

El clon incluye `demo/` con doce ejemplos, las cinco tablas, las figuras 3 a 5,
el catálogo de campeones y los modelos para ambas calculadoras. La interfaz busca
primero cada producto en `data/` y, si falta, usa la copia pública de `demo/`.
No hace falta crear `.env` para abrir las repeticiones y los resultados.
La calculadora previa usa diez diferencias numéricas y muestra los factores que
más pesan. Solo la consulta en vivo requiere credenciales y conexión.

Regeneramos la copia pública después de producir los datos locales con:

```bash
python scripts/build_demo.py
```

El comando no consulta Riot. Exporta el pipeline previo sin tablas por jugador,
omite identificadores de entrenamiento y de ejemplos, quita la columna `fuente`
y comprueba que el total no supere 15 MB. Ver [demo/LEEME.md](demo/LEEME.md).

Los análisis locales se encuentran en `scripts/analisis/`:

```bash
python -m scripts.analisis.minuto --debug
python -m scripts.analisis.minuto --final
python -m scripts.analisis.previo_misma_poblacion
python -m scripts.analisis.objetivos
python scripts/check_timeline_consistency.py
```

Necesitan la base, exportaciones, caché de partidas y líneas de tiempo, el modelo
congelado y los resultados de prueba originales bajo `data/`. El protocolo está en
[scripts/analisis/protocolo.md](scripts/analisis/protocolo.md). `--debug` revisa solo
entrenamiento. `--final` realiza una evaluación confirmatoria y deja una guardia
contra repeticiones. La corrección del previo añade filas una sola vez a la tabla
original. Estos comandos pueden tardar y no se necesitan para usar la demo.
Escriben informes, auditorías, tablas y figuras en `data/analysis/`, con temporales
en `data/analysis/tmp/`. El análisis por minuto también genera el modelo en
`data/models/` y ejemplos en `data/app/`. La comprobación de eventos guarda
`data/analysis/consistencia.md`.
