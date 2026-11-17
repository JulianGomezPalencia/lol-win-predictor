# Verificación de la demostración

Trabajamos el 17 de noviembre de 2026 en la rama `feature/presentacion`.

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
Riot IDs, campos privados y rutas locales.

Añadimos un cotejo de campos de texto completos con las identidades de la base
local y de la caché de partidas.
Abrimos SQLite en modo de solo lectura. Comparamos nombres sin etiqueta, puuids,
Riot IDs e identificadores de partida en memoria. No guardamos ni mostramos los
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
celdas de código y resultados. No reejecutamos los notebooks.

## Regeneración local

Ejecutamos `.venv/bin/python scripts/build_demo.py` después de las correcciones.
La nueva distribución ocupó 815.303 bytes. El escaneo Unicode y el cotejo
completo con la base y la caché locales terminaron sin hallazgos.

## Pruebas y alcance

Ejecutamos `.venv/bin/python -m pytest -q tests/test_demo.py`. Pasaron las 16
pruebas. Cubrieron la distribución sin
conexión, las calculadoras, la prioridad de productos locales, la exportación,
la limpieza de archivos antiguos y el rechazo de salidas que contienen entradas.
También cubrieron Riot IDs Unicode y nombres sintéticos sin etiqueta en JSON,
CSV, modelos comprimidos y metadatos PNG. Comprobaron que la lectura de SQLite
y de la caché no alterara sus archivos ni creara otros junto a ellos.

Reservamos la batería completa para la comprobación automática. No repetimos
análisis pesados, entrenamientos ni evaluaciones confirmatorias. No hicimos
llamadas a Riot ni commits. La instalación desde un clon nuevo con un entorno
virtual nuevo queda pendiente. Las pruebas sin datos privados usaron una raíz
temporal y las dependencias ya instaladas.
