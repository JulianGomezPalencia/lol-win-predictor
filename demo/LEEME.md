# Demostración sin conexión

Regeneramos estos archivos desde la raíz con `python scripts/build_demo.py`.
El comando lee los productos locales de `data/` y no consulta Riot.

- `modelo_previo.joblib` contiene el pipeline y las diez variables de `data/models/modelo_final.joblib`. Omitimos las tablas por jugador y los identificadores.
- `modelos_en_partida.joblib` conserva los modelos de `data/models/modelos_en_partida.joblib`, sin la lista de partidas de entrenamiento.
- `ejemplos.json` procede de `data/app/ejemplos.json`. Publicamos solo campos de pantalla y numeramos las partidas.
- `analysis/` contiene las cinco tablas de `data/analysis/`, sin la columna de procedencia.
- `figuras/` copia las figuras 3 a 5 de `data/analysis/figuras/`.
- `champion_16.22.1.json` copia los nombres públicos de campeones de `data/static/`.

Las tablas y figuras se regeneran con los módulos de `scripts/analisis/`, descritos en el README del proyecto. La demostración no necesita esos datos originales, credenciales ni conexión. La calculadora previa usa solo diferencias numéricas, sin historiales individuales.
