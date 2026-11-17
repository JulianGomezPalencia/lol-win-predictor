# Uso avanzado

## Entorno y archivos locales

Trabajamos con Python 3.14.3. Para usar los comandos de esta guía desde la raíz, activa el entorno creado al instalar y configura las credenciales solo si vas a consultar Riot.

```bash
source .venv/bin/activate
cp .env.example .env
```

Edita `.env` en tu equipo. No publiques su contenido ni los archivos privados de `data/`. La demo no requiere este paso. El catálogo de campeones, las tablas, las figuras y los modelos públicos vienen en `demo/`.

| Carpeta | Contenido |
|---|---|
| `src/riot_client/` | Peticiones, reintentos, limitación de frecuencia y caché |
| `src/storage/` | Esquema y consultas de SQLite |
| `src/agents/` | Recolección de semillas, partidas, historiales y contexto |
| `src/features/` | Variables previas, posiciones inferidas y estado por minuto |
| `src/models/` | Ajuste, evaluación, predicción y aportes por variable |
| `src/app/` | Servidor Flask y página local |
| `scripts/analisis/` | Análisis por minuto, comparación del previo y objetivos |
| `notebooks/` | Análisis del informe |
| `demo/` | Productos públicos para abrir la interfaz sin credenciales |
| `data/` | Base, caché, exportaciones, modelos y resultados privados |

## Recolección reanudable

La primera orden elige semillas y descubre sus partidas. La segunda continúa lo pendiente. Repetir descubrimiento amplía el conjunto y no equivale a reanudar una descarga.

```bash
python -m scripts.run_pipeline --steps seed,discover
python -m scripts.run_pipeline
python -m scripts.run_pipeline --steps history --limit 50
```

| Paso | Operación |
|---|---|
| `seed` | Selecciona jugadores de distintos rangos |
| `discover` | Añade sus partidas recientes a la cola |
| `fetch` | Descarga partidas y descarta las que no cumplen los filtros |
| `history` | Recupera hasta diez partidas anteriores de cada participante |
| `snapshot` | Consulta rango y maestría actuales como contexto |

Cada proceso limita sus propias peticiones. No ejecutes recolección, vigilancia o puntuación del registro simultáneamente. Los fallos de red se reintentan y la caché evita volver a descargar lo guardado.

Para ampliar el conjunto, detén primero los otros procesos y respalda la base. Estos comandos muestran la secuencia usada. Los límites se pueden reducir para una recolección propia.

```bash
cp data/db/lol.sqlite data/db/lol_respaldo.sqlite
python -m scripts.run_pipeline --steps discover --matches-per-player 15 --days-back 14
python -m scripts.run_pipeline --steps fetch,history --limit 300
python -m scripts.export_dataset
```

En macOS se puede anteponer `caffeinate -i` al proceso largo para impedir que el equipo duerma. La descarga de historiales tarda mucho más que abrir la demo. No exportes mientras la recolección esté escribiendo. Antes de sobrescribir un export que necesites reproducir, conserva una copia con fecha.

## Variables y posiciones

La receta previa infiere posiciones con hechizos, campeón e historial anterior. Asigna una posición distinta a cada integrante. No lee las posiciones asignadas al terminar la partida. El rango consultado después de jugar queda como contexto y no entra como predictor.

Los comandos generales `python -m src.features.build` y `python -m src.models.train` son exploratorios. El primero conserva por defecto posiciones posteriores y el segundo selecciona las variables presentes en su tabla. No reproducen el modelo previo de diez variables.

Tras exportar con `python -m scripts.export_dataset`, fija `CORTE` antes de mirar los resultados. Sustituye la fecha del ejemplo por el corte de tu recogida, con suficientes partidas anteriores para la validación temporal y nuevas partidas posteriores para una sola evaluación. No uses las cifras históricas como resultados de este ajuste.

```bash
CORTE=2026-11-18T00:00:00+00:00 python - <<'PY'
import os
from pathlib import Path
import pandas as pd
from src.features import build
from src.models import predict, train

export = Path("data/export")
cut = int(pd.Timestamp(os.environ["CORTE"]).timestamp() * 1000)
matches, history = predict._read_export(export)
fit_ids = matches.loc[matches.game_start_ts < cut, "match_id"]
pool, shares = predict._pool_and_shares(history, fit_ids)
# _table_for activa inferred_positions=True y final_features=True.
table = predict._table_for(export, pool, shares)
fit = table[table.game_start_ts < cut].copy()
test = table[table.game_start_ts > cut].copy()
if fit.empty or test.empty:
    raise ValueError("Se necesitan partidas a ambos lados del corte")
output = Path("data/models/modelo_propio.joblib")
if output.exists():
    raise FileExistsError("Conserva el modelo congelado y sus resultados")
bundle = predict.fit_final_model(fit, pool, shares)
assert bundle["features"] == build.FINAL_FEATURES
predict.save_model(bundle, output)
prob = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
print("Partidas nuevas", len(test))
print(train.evaluate(test.blue_win.astype(int), prob))
PY
```

El pool y las frecuencias de posición usan solo historiales del ajuste. La construcción filtra los antecedentes por la fecha de cada partida. Los faltantes pasan por el imputador del modelo. Conserva el corte, el bundle, las exportaciones y las métricas para identificar esta evaluación. Repetirla sobre las mismas partidas no aporta evidencia nueva. Esta receta usa funciones del proyecto y guarda un modelo propio separado del histórico.

## Modelo previo y pruebas históricas

Conservamos los siguientes comandos para documentar la secuencia original. Ya completamos las dos pruebas. No deben repetirse sobre las mismas partidas para producir evidencia nueva.

```bash
python -m src.models.predict --freeze
python -m src.models.predict --evaluate-new
python -m src.models.predict --evaluate-final
python -m src.models.predict --train
```

`--freeze` conservó la base y ajustó el candidato de diez variables. `--evaluate-new` cargó ambos sin reajustarlos y excluyó las partidas de desarrollo. La mejora exigió un intervalo positivo de reducción de pérdida logarítmica frente a la base.

`--evaluate-final` evaluó el modelo congelado frente a una constante con la tasa azul del ajuste. Excluyó ajuste y desarrollo y exigió fechas posteriores a las del entrenamiento y al corte registrado. Añadió resultados a `data/models/segunda_prueba.json`. No volvió a entrenar. Repetir con partidas solapadas no constituye otra prueba independiente.

`--train` se ejecutó después de ambas pruebas. Exige los registros correspondientes al modelo actual y conserva una copia fechada con comprobación SHA-256 antes de sustituirlo. Si esa copia ya existe con contenido distinto, se detiene. `--force` omite las comprobaciones con aviso y no debe usarse para atribuir resultados históricos a un reajuste. Los bundles conservan `modelo_probado` para distinguir las métricas previas. El reajuste con 4.237 partidas sigue sin prueba independiente propia.

## Confianza y predicción previa en vivo

La confianza es el mayor valor entre la probabilidad azul y su complemento. En los comandos de predicción, por debajo de 55 % no damos favorito. Desde 55 % el nivel es bajo, desde 60 % medio y desde 65 % alto. Estos niveles describen el cálculo y los grupos históricos, no garantizan una partida. La pantalla de votación permite ver la probabilidad aunque esté cerca de la mitad.

```bash
python -m src.models.predict --help
python -m src.models.predict --watch --watch-minutes 60 --max-games 20
python -m src.models.predict --score-log
```

Para consultar una persona usa `--live` con el nombre y etiqueta solo en tu terminal. La partida debe estar en Solo/Dúo y haber empezado a cargar. Recuperar los historiales puede tardar varios minutos. Una consulta bloqueada o sin historial deja entradas faltantes. No cambia la receta del modelo.

La vigilancia recorre las semillas locales en lectura, omite partidas repetidas y se detiene al alcanzar tiempo o número de partidas. Ctrl+C también la detiene. El plazo limita consultas, reintentos y esperas. Los errores de red y predicción se cuentan por separado.

El registro `data/predictions/live_log.csv` permite puntuar después las partidas terminadas. `--score-log` puede consultar Riot para recuperar resultados aún no guardados, por eso también debe ejecutarse solo. Informa acierto, pérdida, abstenciones y resultados por umbral con intervalos Wilson. Una muestra pequeña sigue siendo insuficiente aunque el programa produzca métricas.

`live_context.csv` conserva contexto por participante, como rango, maestría y antecedentes, sin identificadores personales. Las filas se publican juntas antes de registrar la predicción y los datos ausentes quedan vacíos. Las consultas de contexto aumentan el tiempo y no modifican las diez variables.

## Líneas de tiempo y modelos por minuto

```bash
python scripts/download_timelines.py --max 2800 --minutes 75
python scripts/check_timeline_consistency.py
python -m src.models.ingame --help
CORTE=2026-11-18T00:00:00+00:00
python -m src.models.ingame --train --before "$CORTE" --model data/models/estado_propio.joblib
```

La descarga se reanuda desde la caché y se detiene ante credenciales rechazadas. La comprobación contrasta eventos y totales de objetivos y escribe `data/analysis/consistencia.md`. El entrenamiento lee datos locales, sin llamadas a Riot.

La receta de estado usa diferencias azul menos rojo hasta el cuadro observado. Un modelo corresponde a cada minuto. Sustituye la fecha del ejemplo por el corte reservado para tus datos, antes de entrenar. Evalúa una sola vez con el mismo corte y bundle, sin reajustar.

```bash
python -m src.models.ingame --evaluate --after "$CORTE" --model data/models/estado_propio.joblib
```

El estudio original usó el corte 2026-11-09T18:09:41+00:00. No repitas una evaluación sobre partidas ya usadas. Estos comandos propios no producen las tablas confirmatorias del estudio original. Los minutos finales pueden incluir cuadros de partidas ya terminadas. Esa limitación debe acompañar la interpretación de las tasas.

## Análisis y regeneración de la demo

```bash
python -m scripts.analisis.minuto --debug
python -m scripts.analisis.minuto --final
python -m scripts.analisis.previo_misma_poblacion
python -m scripts.analisis.objetivos
python scripts/build_demo.py
```

Estas órdenes documentan el estudio original y no son una receta general para datos propios.

| Orden | Requisitos y alcance |
|---|---|
| `minuto --debug` | Necesita SQLite, sal de exportación, partidas completas y timelines locales, además de `data/models/modelo_final_2026-11-09.joblib`. Usa únicamente filas anteriores al corte fijo 2026-11-09T18:09:41+00:00. Entrena los modelos de estado con todas esas filas y limita a cien partidas el cálculo previo fuera de muestra. Escribe `debug_entrenamiento.json`, sin evaluar prueba ni generar las tablas finales. |
| `minuto --final` | Necesita lo anterior, `data/export/` original y `data/models/segunda_prueba.json`. Reconstruye la pertenencia y exige las 1.264 partidas de la segunda prueba. Usa el mismo corte fijo, espera una descarga local activa y conserva `evaluacion_iniciada.json` contra repeticiones. |
| `previo_misma_poblacion` | Necesita el modelo histórico fechado, `modelos_en_partida.joblib`, `segunda_prueba.json`, la guardia, exportación y caché originales, y `predictibilidad.csv` aún sin las filas de esta corrección. No entrena ni ejecuta `--final`. Añade el previo en las mismas partidas de cada minuto y se detiene si ya se aplicó. |
| `objetivos` | Lee partidas completas en `data/raw/matches/*.gz`, sin modelos ni timelines. Filtra Solo/Dúo y duración mínima de cinco minutos, agrega objetivos y genera su tabla y figura. Puede describir otra colección local, con sus propios tamaños y límites. |
| `build_demo` | Requiere productos locales ya calculados, modelos, tablas, figuras y ejemplos. No calcula una evaluación ni reconstruye los datos privados desde `demo/`. |

El modelo histórico fechado contiene el pipeline, las diez variables, el pool, las frecuencias por posición y los identificadores de ajuste y desarrollo. El modelo reducido de `demo/` no lo sustituye. Los archivos privados originales no están en el repositorio. Cambiar solo la ruta del modelo o el corte no adapta las comprobaciones históricas a otra muestra. El [protocolo](../scripts/analisis/protocolo.md) conserva las decisiones previas. No se reconstruyen los resultados originales solo con los ejemplos públicos.

Los análisis escriben tablas, auditorías y figuras en `data/analysis/`, temporales en `data/analysis/tmp/`, modelos en `data/models/` y ejemplos en `data/app/`. No hay que ejecutarlos para presentar.

`scripts/build_demo_examples.py` necesita un CSV de candidatas con las columnas `match_id`, `categoria` y `prob_previa`, pasado mediante `--candidates`. No genera ese CSV. La selección original usó semilla 42 y cuotas de cuatro aciertos con confianza, dos fallos con confianza, tres remontadas y tres al azar, sin repeticiones.

`build_demo.py` no consulta Riot. Exporta el pipeline previo sin tablas por jugador, quita identificadores de ejemplos y entrenamiento, elimina la columna de procedencia de las tablas y comprueba el límite de 15 MB. Lee los productos locales ya calculados. Ver [archivos de demostración](../demo/LEEME.md).

## Notebook y comprobaciones

`notebooks/modelo_final.ipynb` necesita `data/export_2026-11-09/`, los CSV de partidas, participantes, historiales y hechizos, `data/models/conjunto_desarrollo.csv` y los resultados originales de segunda prueba. Reconstruye el análisis previo en memoria y no guarda modelos. Para el análisis por minuto lee las tablas de `data/analysis/`.

En otro entorno hay que aportar esos archivos privados. La celda que clona el proyecto usa `main`, de modo que esa rama debe contener los cambios que se quieran ejecutar. Los intervalos descriptivos del notebook no sustituyen los de la evaluación original.

```bash
python -m pytest -q
python -m src.app
```

La interfaz Flask usa recursos locales y siete vistas. Las operaciones de datos viven en `src/app/logic.py`. La calculadora usa las diez diferencias previas, con cero como igualdad, y explica aportes al cálculo sin interpretarlos como causas de victoria. La consulta en vivo necesita el modelo local completo. El reloj de las repeticiones llega hasta el último cuadro guardado, como máximo 30 en los ejemplos distribuidos, y el ganador se revela por separado.
