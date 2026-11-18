# Verificación de la demostración

Esta verificación documenta la preparación de los archivos de demostración.

## Archivos preparados

Distribuimos en `demo/` el pipeline previo con sus diez variables, los modelos
por minuto, doce ejemplos numerados, el catálogo público de campeones, cinco
tablas y las figuras 3 a 5. El archivo `demo/LEEME.md` explica su procedencia
y cómo regenerarlos. Omitimos las tablas por jugador, los identificadores de
entrenamiento y la columna de procedencia de las tablas de análisis.

Adaptamos la carga de productos para usar los datos locales cuando existen y
recurrir a la demostración cuando faltan. Conservamos los parámetros entrenados.
Las pruebas compararon las probabilidades antes y después de exportarlos.

## Correcciones comprobadas

Confirmamos los tres hallazgos de la revisión. El escaneo anterior operaba sobre
bytes y omitía nombres Unicode. La regeneración conservaba archivos obsoletos.
Los dos notebooks contenían puntuación que incumplía las reglas de redacción.

Cambiamos el escaneo para trabajar con texto Unicode normalizado. Decodificamos
las cadenas escapadas de JSON, los metadatos de imágenes y las cadenas de los
modelos, incluidos los comprimidos. Excluimos los bloques numéricos de matrices.
Conservamos las comprobaciones de tokens largos, identificadores de partida,
identificadores de cuenta, campos privados y rutas locales.

Añadimos un cotejo de campos de texto completos con las identidades de la base
local y de la caché de partidas.
Abrimos SQLite en modo de solo lectura. Comparamos nombres sin etiqueta, identificadores personales,
identificadores de cuenta e identificadores de partida en memoria. No guardamos ni mostramos los
valores leídos. El escaneo informa solo los archivos afectados. Distinguimos los
números de las tablas y los nombres oficiales de campeones. Verificamos el
catálogo público mediante SHA-256 antes de excluirlo del cotejo de identidades.
La excepción de nombres en los ejemplos se limita a sus campos de campeones.
También distinguimos las etiquetas numeradas y las categorías previstas de
equipo, evento y objetivos. Las excepciones usan valores y campos definidos en
el código, sin excluir textos desconocidos ni columnas adicionales.
Un nombre igual en una tabla o un modelo sigue siendo un hallazgo.
Sin datos locales,
las reglas por forma siguen activas, pero no permiten reconocer cualquier nombre
sin etiqueta. El cotejo local tampoco reconoce nombres incrustados dentro de
frases más largas. Las reglas por formato sí buscan coincidencias dentro del
texto. No inspeccionamos el texto dibujado dentro de una imagen.

Vaciamos el directorio de salida antes de regenerarlo. Rechazamos una salida
que coincida con los datos de entrada, esté dentro de ellos o los contenga.
Así evitamos borrar las entradas durante la limpieza. Las pruebas añadieron
archivos obsoletos y comprobaron que desaparecieran de la nueva distribución.

Corregimos las rayas largas y los puntos y coma de las celdas Markdown de
`notebooks/entrega2_eda.ipynb` y `notebooks/modelo_final.ipynb`. Conservamos sus
celdas de código y resultados. La comprobación posterior del cuaderno final se documenta al final de esta página.

## Regeneración local

Ejecutamos `.venv/bin/python scripts/build_demo.py` después de las correcciones.
La nueva distribución ocupó 815.303 bytes. El escaneo Unicode y el cotejo
completo con la base y la caché locales terminaron sin hallazgos.

## Pruebas y alcance

Ejecutamos `.venv/bin/python -m pytest -q tests/test_demo.py`. Pasaron las 16
pruebas. Cubrieron la distribución sin
conexión, las calculadoras, la prioridad de productos locales, la exportación,
la limpieza de archivos antiguos y el rechazo de salidas que contienen entradas.
También cubrieron identificadores de cuenta Unicode y nombres sintéticos sin etiqueta en JSON,
CSV, modelos comprimidos y metadatos PNG. Comprobaron que la lectura de SQLite
y de la caché no alterara sus archivos ni creara otros junto a ellos.

No repetimos
análisis pesados, entrenamientos ni evaluaciones confirmatorias. Esta verificación de la distribución no necesitó consultas en red. La instalación desde un clon nuevo con un entorno
virtual nuevo queda pendiente. Las pruebas sin datos privados usaron una raíz
temporal y las dependencias ya instaladas.

## Comprobación posterior del cuaderno y del registro

Ejecutamos una copia de `notebooks/modelo_final.ipynb` con nbclient desde la raíz del proyecto. Sus nueve celdas de código terminaron sin errores. El archivo original conservó su contenido y no se guardaron modelos ni nuevas evaluaciones en los datos.

La puntuación del registro del modelo evaluado dio 33 partidas terminadas, 15 aciertos (45,5 %), AUC 0,515 y pérdida 0,7081. Una predicción de otra versión quedó excluida. Con confianza de al menos 55 % hubo 12 aciertos de 20 (60,0 %, IC Wilson 38,7 a 78,1). No hubo pendientes ni descartes del modelo evaluado. La muestra sigue siendo insuficiente. La consulta se realizó sin escribir en `data/`.

Las cinco pruebas de `tests/test_predict.py` seleccionadas con `-k score_log` pasaron. La batería completa pasó: 368 pruebas aprobadas.
