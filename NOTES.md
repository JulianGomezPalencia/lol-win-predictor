# Diario de desarrollo

## 2026-10-20

Corregimos los reintentos del cliente HTTP para evitar esperas después del último
intento y reanudar descargas interrumpidas. Guardamos los datos originales en caché
y separamos recolección, historiales y exportación. Las consultas de historial
usan solo partidas anteriores a la partida objetivo.

## 2026-10-26

Evaluamos 920 partidas con 29 diferencias entre equipos. La división temporal usó
736 para ajuste y 184 para prueba. La logística acertó 56,0 % en esa prueba
(IC 95 % de 49 a 63). En validación hacia adelante tuvo AUC 0,571
(0,532 a 0,611). El bosque aleatorio no mejoró la logística.
La diferencia de acierto frente a la clase mayoritaria no fue concluyente.

## 2026-11-04

Revisamos posiciones inferidas y predicción en vivo. Dejamos de depender de la
posición que Riot asigna al terminar la partida. Unificamos el cálculo para
partidas jugadas y partidas en curso, a partir de campeones, hechizos e historial.
El rango consultado después de jugar se conserva como contexto, no como variable.

## 2026-11-05

Congelamos diez variables y la receta de regresión logística. Las 1.206 partidas
usadas durante el desarrollo ya no podían servir como prueba independiente.
Ajustamos el candidato con las mismas 920 del modelo base y reservamos partidas
posteriores. Incluimos las otras 286 en la lista de desarrollo para excluirlas
de las pruebas. El pipeline usa imputación por mediana y escalado antes de la
logística. La comparación por nueve variables se mantuvo como sensibilidad.

## 2026-11-09

Fijamos la segunda prueba antes de consultar sus resultados. Conservamos el modelo
probado y separamos el reajuste posterior. Evaluamos el aporte de la tasa global
de victoria de campeones con información disponible antes del juego. No mejoró
la pérdida logarítmica y no entró en la receta.

## 2026-11-11

La segunda prueba incluyó 1.264 partidas nuevas. El modelo final acertó 57,04 % y
obtuvo AUC 0,5891. Los resultados completos están en
`data/models/segunda_prueba.json`. Evaluamos premade, composición, variables por
rol y un modelo no lineal solo sobre las 2.973 partidas de ajuste. Ninguna familia
cumplió la regla de mejora y no hicimos una tercera prueba.

Conservamos las probabilidades sin recalibrar. La ganancia estimada de pérdida
por recalibración fue 0,000293, IC 95 % de −0,000766 a 0,001344.
La confianza es el mayor valor entre p y 1-p. Por debajo de 55 % no damos un
favorito. Los umbrales de 55, 60 y 65 % tuvieron 699, 275 y 84 partidas en la
segunda prueba y aciertos de 59,51, 65,82 y 67,86 %. Son resultados de grupos,
no garantías sobre una partida individual.

El reajuste conserva una copia fechada del modelo anterior con verificación de
SHA-256. Las cifras históricas siguen atribuidas a la versión probada.
La vigilancia omite partidas repetidas, limita su tiempo y cuenta errores sin
mostrar datos personales. El modo de consulta de la interfaz no guarda el Riot ID.
Corregimos la recuperación del contexto, la propagación del plazo a reintentos
y el manejo de errores durante la vigilancia.

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

## Trabajo del 2026-11-17

Preparamos una demostración pública para usar la interfaz desde un clon sin datos
privados ni credenciales. El modelo previo distribuido conserva solo el pipeline
y las diez variables. El modelo por minuto conserva sus parámetros y omite la
lista de partidas usadas para ajustar. Numeramos los ejemplos y quitamos las
columnas de procedencia de las tablas. Los nombres de campeones vienen del
catálogo público del parche 16.22.1.

Trasladamos los tres scripts de análisis a `scripts/analisis/` y conservamos sus
cálculos y guardias. Las salidas quedan bajo `data/analysis/`. Añadimos el protocolo
para conservar las decisiones fijadas antes de la evaluación original.
No repetimos esos análisis ni entrenamos modelos. La demo se regenera con
`python scripts/build_demo.py`, que lee los productos locales y escanea su salida.
Las pruebas de `tests/test_demo.py` cubren carga sin datos locales, predicción,
explicaciones y ausencia de identificadores en los archivos públicos.
