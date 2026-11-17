"""Prepara los archivos públicos de demostración desde los productos locales."""
import argparse
import hashlib
import gzip
import json
import pickle
import pickletools
import re
import shutil
import sqlite3
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from src.app.logic import public_example, champion_names
from src.features.build import FINAL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CATALOG_SHA256 = 'ab0bebb32fdc048a35ed6182d127ae43adcc4734b6360a84ee0b6f508773e0cf'
TABLES = ('predictibilidad', 'comparacion_previo', 'calibracion_15', 'remontadas', 'objetivos')
# Patrones aplicados al texto decodificado.
PRIVATE = re.compile(r'(?<![A-Za-z0-9_+/-])[A-Za-z0-9_+/-]{70,}={0,2}(?![A-Za-z0-9_+/-])|\b(?:LA[12]|NA1|EUW1|EUN1|KR|BR1|OC1|JP1|RU|TR1)_\d+\b|[\w.-]+#[\w.-]+|/[U]sers/|[.]flow/|\b(?:puuid|summonerName|riotIdGameName|riotIdTagline|match_id_interno|match_ids)\b')


# Vocabulario de categorías publicado por el análisis de objetivos.
OBJECTIVE_VALUES = {
    'tipo': {'objetivo', 'cobertura_objetivo', 'combinacion_mismo_equipo',
             'combinacion_patron', 'cobertura_combinacion', 'ausencia_combinacion', 'auditoria_archivos'},
    'objetivo': {'champion', 'tower', 'dragon', 'horde', 'riftHerald', 'baron',
                 'inhibitor', 'atakhan', 'voidGator', 'sangre_torre_dragon'},
    'etiqueta': {'Primera sangre', 'Primera torre', 'Primer dragón', 'Primeras larvas',
                 'Primer heraldo', 'Primer barón', 'Primer inhibidor', 'Primer Atakhan',
                 'Objetivo voidGator', 'sangre_torre_dragon'},
    'lado': {'total', 'azul', 'rojo'},
    'patron': {'AAA', 'AAR', 'ARA', 'ARR', 'RAA', 'RAR', 'RRA', 'RRR'},
    'estado': {'valido', 'ausente', 'first_ausente', 'first_invalido', 'sin_primero',
               'contradictorio', 'no_evaluable', 'repartidos', 'ilegible', 'duplicado',
               'sin_id', 'sin_equipos', 'sin_ganador', 'cola', 'remake'},
}


IDENTITY_FIELDS = {
    'puuid', 'seed_puuid', 'summonerName', 'summoner_name', 'riotIdGameName',
    'riot_id', 'match_id', 'matchId', 'target_match_id',
    'history_match_id', 'gameName',
}


def local_identifiers(data):
    """Lee identidades locales sin crear archivos ni modificar la base."""
    data = Path(data)
    for database in sorted((data / 'db').glob('*.sqlite')):
        with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (table,) in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                columns = [row[1] for row in conn.execute(f'PRAGMA table_info({quoted})')
                           if row[1] in IDENTITY_FIELDS]
                for column in columns:
                    for (value,) in conn.execute(f'SELECT DISTINCT "{column}" FROM {quoted}'):
                        if isinstance(value, str) and value.strip():
                            yield value
    for path in sorted((data / 'raw/matches').glob('*.json*')):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            match = json.load(stream)
        yield from match.get('metadata', {}).get('participants', [])
        match_id = match.get('metadata', {}).get('matchId')
        if match_id:
            yield match_id
        for player in match.get('info', {}).get('participants', []):
            for field in IDENTITY_FIELDS:
                value = player.get(field)
                if isinstance(value, str) and value.strip():
                    yield value
            if player.get('riotIdGameName') and player.get('riotIdTagline'):
                yield player['riotIdGameName'] + '#' + player['riotIdTagline']


def text_values(value):
    """Extrae valores de texto completos, sin interpretar números como nombres."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from text_values(child)


def scan_demo(directory, data=None):
    """Inspecciona Unicode y coteja campos completos con identidades locales."""
    candidates = {}
    failures = set()
    catalogue = Path(directory) / 'champion_16.22.1.json'
    public_champions = set()
    verified_catalogue = (catalogue.is_file() and
                          hashlib.sha256(catalogue.read_bytes()).hexdigest() == PUBLIC_CATALOG_SHA256)
    if verified_catalogue:
        public_champions = set(champion_names(catalogue).values())
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file():
            continue
        if path.suffix == '.joblib':
            serialized = pickle.dumps(joblib.load(path), protocol=4)
            # Los bloques binarios de matrices no son cadenas de texto.
            values = [value for opcode, value, _ in pickletools.genops(serialized)
                      if opcode.name in ('UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8')]
            payload = '\n'.join(values)
        elif path.suffix == '.png':
            from PIL import Image
            with Image.open(path) as image:
                values = list(text_values(image.info))
                payload = json.dumps(image.info, ensure_ascii=False)
        elif path.suffix == '.json':
            document = json.loads(path.read_text(encoding='utf-8'))
            payload = json.dumps(document, ensure_ascii=False)
            values = list(text_values(document))
            # Los nombres oficiales de campeones son datos públicos, aunque
            # algún jugador elija el mismo nombre. Solo eximimos esos campos.
            if path == catalogue and verified_catalogue:
                values = []
            elif path.name == 'ejemplos.json' and isinstance(document, list):
                for index, example in enumerate(document, 1):
                    if example.get('etiqueta') == f'Partida {index}':
                        values.remove(example['etiqueta'])
                    for event in example.get('eventos', []):
                        for field, allowed in [('tipo', {'dragón', 'torre', 'barón'}),
                                               ('equipo', {'azul', 'rojo'})]:
                            if event.get(field) in allowed:
                                values.remove(event[field])
                    for team in example.get('campeones', {}).values():
                        for name in team:
                            if name in public_champions:
                                values.remove(name)
        else:
            payload = path.read_text(encoding='utf-8')
            if path.suffix == '.csv':
                table = pd.read_csv(path)
                values = [value for column in table for value in table[column]
                          if isinstance(value, str) and not (
                              path.name == 'objetivos.csv'
                              and value in OBJECTIVE_VALUES.get(column, set()))]
            else:
                values = payload.splitlines() + re.findall(r'`([^`]+)`', payload)
        payload = unicodedata.normalize('NFC', payload)
        for value in [*values, path.stem]:
            value = unicodedata.normalize('NFC', value.strip())
            if value:
                candidates.setdefault(value, set()).add(path.name)
        if PRIVATE.search(payload) or PRIVATE.search(str(path.relative_to(directory))):
            failures.add(path.name)
    if data is not None and Path(data).exists():
        for value in local_identifiers(data):
            value = unicodedata.normalize('NFC', value.strip())
            failures.update(candidates.get(value, ()))
    return sorted(failures)


def build(data, output):
    data, output = Path(data), Path(output)
    if (data.resolve() == output.resolve() or data.resolve() in output.resolve().parents
            or output.resolve() in data.resolve().parents):
        raise ValueError('La salida debe estar fuera de los datos de entrada.')
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    prior = joblib.load(data / 'models/modelo_final.joblib')
    if prior['features'] != FINAL_FEATURES:
        raise ValueError('El modelo previo debe tener las diez variables previstas.')
    # El archivo completo incluye tablas por jugador e identificadores de partidas.
    joblib.dump({'model': prior['model'], 'features': list(prior['features'])}, output / 'modelo_previo.joblib')
    ingame = joblib.load(data / 'models/modelos_en_partida.joblib')
    # Los parámetros se conservan. La lista de partidas de entrenamiento se omite.
    ingame = {key: ingame[key] for key in ('version', 'features', 'models', 'counts', 'unavailable_minutes', 'trained_at', 'recipe')}
    joblib.dump(ingame, output / 'modelos_en_partida.joblib')
    raw = json.loads((data / 'app/ejemplos.json').read_text())
    names = champion_names(data / "static/champion_16.22.1.json").values()
    examples = [public_example(example, index, names) for index, example in enumerate(raw, 1)]
    (output / 'ejemplos.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n')
    (output / 'analysis').mkdir(exist_ok=True)
    for name in TABLES:
        table = pd.read_csv(data / f'analysis/{name}.csv').drop(columns=['fuente'], errors='ignore')
        table.to_csv(output / f'analysis/{name}.csv', index=False)
    (output / 'figuras').mkdir(exist_ok=True)
    for number in (3, 4, 5):
        sources = list((data / 'analysis/figuras').glob(f'figura_{number}_*.png'))
        if len(sources) != 1:
            raise ValueError(f'Falta una figura única para el número {number}.')
        shutil.copyfile(sources[0], output / 'figuras' / sources[0].name)
    shutil.copyfile(data / 'static/champion_16.22.1.json', output / 'champion_16.22.1.json')
    (output / 'LEEME.md').write_text('''# Demostración sin conexión

Regeneramos estos archivos desde la raíz con `python scripts/build_demo.py`.
El comando lee los productos locales de `data/` y no consulta Riot.

- `modelo_previo.joblib` contiene el pipeline y las diez variables de `data/models/modelo_final.joblib`. Omitimos las tablas por jugador y los identificadores.
- `modelos_en_partida.joblib` conserva los modelos de `data/models/modelos_en_partida.joblib`, sin la lista de partidas de entrenamiento.
- `ejemplos.json` procede de `data/app/ejemplos.json`. Publicamos solo campos de pantalla y numeramos las partidas.
- `analysis/` contiene las cinco tablas de `data/analysis/`, sin la columna de procedencia.
- `figuras/` copia las figuras 3 a 5 de `data/analysis/figuras/`.
- `champion_16.22.1.json` copia los nombres públicos de campeones de `data/static/`.

Las tablas y figuras se regeneran con los módulos de `scripts/analisis/`, descritos en el README del proyecto. La demostración no necesita esos datos originales, credenciales ni conexión. La calculadora previa usa solo diferencias numéricas, sin historiales individuales.
''')
    failures = scan_demo(output, data)
    if failures:
        raise ValueError(f'Contenido privado detectado en {failures}')
    size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    if size > 15_000_000:
        raise ValueError('La demostración supera los 15 MB.')
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'demo')
    args = parser.parse_args()
    print(f'Demostración regenerada: {build(args.data, args.output)} bytes. Escaneo sin hallazgos.')


if __name__ == '__main__':
    main()
