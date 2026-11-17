"""Contrato de distribución pública y funcionamiento sin datos privados."""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts import build_demo
from src.app import logic
from src.features.build import FINAL_FEATURES
from src.models import ingame

DEMO = Path(__file__).resolve().parents[1] / 'demo'


@pytest.fixture
def clean_clone(tmp_path, monkeypatch):
    shutil.copytree(DEMO, tmp_path / 'demo')
    monkeypatch.setattr(logic, 'ROOT', tmp_path)
    monkeypatch.setattr(ingame, 'ROOT', tmp_path)
    monkeypatch.delenv('RIOT_API_KEY', raising=False)
    import requests
    monkeypatch.setattr(requests.sessions.Session, 'request',
                        lambda *args, **kwargs: pytest.fail('La demo no debe consultar la red'))
    return tmp_path


def test_clone_loads_examples_tables_figures_and_calculators(clean_clone):
    assert not (clean_clone / 'data').exists()
    examples = logic.load_examples()
    assert len(examples) == 12
    assert [row['etiqueta'] for row in examples] == [f'Partida {i}' for i in range(1, 13)]
    assert all(len(row['campeones'][team]) == 5 for row in examples for team in ('azul', 'rojo'))
    assert all(c != 'Campeón' for row in examples for team in row['campeones'].values() for c in team)
    paths = logic.findings_paths()
    assert {p.stem for p in paths['tablas']} == set(build_demo.TABLES)
    assert len(paths['figuras']) == 3
    for path in paths['tablas']:
        table = logic.read_findings_table(path)
        assert not table.empty
        assert 'fuente' not in table
    bundle = logic.load_prior_model()
    assert set(bundle) == {'model', 'features'}
    assert bundle['features'] == FINAL_FEATURES
    assert [type(step).__name__ for _, step in bundle['model'].steps] == [
        'SimpleImputer', 'StandardScaler', 'LogisticRegression']
    values = dict.fromkeys(FINAL_FEATURES, 0.)
    result = logic.prior_calculator(values)
    assert 0 < result['prob_azul'] < 1
    assert len(result['factores']) == 10
    logit = bundle['model'].decision_function(__import__('pandas').DataFrame([values]))[0]
    assert np.isclose(sum(f['aporte'] for f in result['factores']) + bundle['model'][-1].intercept_[0], logit)
    assert 0 < logic.calculator(logic.load_ingame_model(), 15, 100, 1, 1, 1) < 1
    assert 0 < logic.calculator(ingame.load_bundle(), 15) < 1


def test_local_products_take_precedence_and_partial_data_falls_back(clean_clone):
    local = clean_clone / 'data/models/modelo_final.joblib'
    local.parent.mkdir(parents=True)
    bundle = joblib.load(DEMO / 'modelo_previo.joblib')
    bundle['local'] = True
    joblib.dump(bundle, local)
    assert logic.load_prior_model()['local'] is True
    assert len(logic.load_examples()) == 12
    assert logic.asset_path('models/modelo_final.joblib', 'modelo_previo.joblib') == local
    assert logic.findings_paths()['tablas'][0].parent == clean_clone / 'demo/analysis'


def test_all_distributed_files_are_small_and_scan_clean():
    assert build_demo.scan_demo(DEMO) == []
    assert sum(p.stat().st_size for p in DEMO.rglob('*') if p.is_file()) <= 15_000_000
    assert 'match_ids' not in joblib.load(DEMO / 'modelos_en_partida.joblib')
    for row in json.loads((DEMO / 'ejemplos.json').read_text()):
        assert set(row) == {'etiqueta', 'gana_azul', 'prob_previa', 'campeones', 'historial', 'curva', 'eventos'}
        for team in ('azul', 'rojo'):
            assert set(row['historial'][team]) == {'winrate_10', 'kda_medio', 'partidas_con_campeon'}
            assert all(isinstance(v, (int, float)) and np.isfinite(v) for v in row['historial'][team].values())
            assert 0 <= row['historial'][team]['winrate_10'] <= 1


def test_distributed_examples_include_inhibitor_and_comeback(clean_clone):
    examples = logic.load_examples()
    assert any(e['tipo'] == 'inhibidor' for row in examples for e in row['eventos'])
    assert any((p['prob_azul'] >= .5) != row['gana_azul']
               for row in examples for p in row['curva'] if p['minuto'] == 15)


@pytest.mark.parametrize('secret', ['Z' * 80, 'A+/' * 30, 'LA1_123456789', 'Jugador#Etiqueta', '玩家#LAN', 'Jugador Uno#LAN'])
def test_scan_detects_identifiers_in_compressed_models_and_text(tmp_path, secret):
    joblib.dump({'nested': [secret]}, tmp_path / 'modelo.joblib', compress=3)
    (tmp_path / 'ejemplo.json').write_text(json.dumps({'valor': secret}))
    assert build_demo.scan_demo(tmp_path) == ['ejemplo.json', 'modelo.joblib']


def test_build_sanitizes_exports_and_preserves_predictions(tmp_path):
    data = tmp_path / 'data'
    for folder in ('models', 'app', 'static'):
        (data / folder).mkdir(parents=True)
    shutil.copytree(DEMO / 'analysis', data / 'analysis')
    shutil.copytree(DEMO / 'figuras', data / 'analysis/figuras')
    shutil.copyfile(DEMO / 'champion_16.22.1.json', data / 'static/champion_16.22.1.json')
    prior = joblib.load(DEMO / 'modelo_previo.joblib')
    prior['pool'] = {'puuid': 'Z' * 80}
    prior['match_ids'] = ['LA1_123456789']
    joblib.dump(prior, data / 'models/modelo_final.joblib')
    state = joblib.load(DEMO / 'modelos_en_partida.joblib')
    state['match_ids'] = ['LA1_123456789']
    joblib.dump(state, data / 'models/modelos_en_partida.joblib')
    examples = json.loads((DEMO / 'ejemplos.json').read_text())
    examples[0].update(match_id_interno='LA1_123456789', puuid='Z' * 80, nombre='Jugador#Etiqueta')
    examples[0]['campeones']['azul'][0] = 'Jugador#Etiqueta'
    (data / 'app/ejemplos.json').write_text(json.dumps(examples))
    table = data / 'analysis/predictibilidad.csv'
    import pandas as pd
    frame = pd.read_csv(table)
    frame['fuente'] = 'ruta privada'
    frame.to_csv(table, index=False)
    output = tmp_path / 'publico'
    output.mkdir()
    (output / 'obsoleto.csv').write_text('Jugador antiguo')
    (output / 'subcarpeta').mkdir()
    (output / 'subcarpeta/figura_vieja.png').write_bytes(b'obsoleto')
    assert build_demo.build(data, output) < 15_000_000
    assert not (output / 'obsoleto.csv').exists()
    assert not (output / 'subcarpeta').exists()
    assert build_demo.scan_demo(output) == []
    assert json.loads((output / 'ejemplos.json').read_text())[0]['campeones']['azul'][0] == 'Campeón'
    assert 'fuente' not in pd.read_csv(output / 'analysis/predictibilidad.csv')
    values = dict.fromkeys(FINAL_FEATURES, 0.)
    exported = joblib.load(output / 'modelo_previo.joblib')
    assert logic.prior_calculator(values, prior) == logic.prior_calculator(values, exported)
    for minute in ingame.CURVE_MINUTES:
        assert logic.calculator(state, minute) == logic.calculator(joblib.load(output / 'modelos_en_partida.joblib'), minute)

    cache = data / 'raw/matches'
    cache.mkdir(parents=True)
    (cache / 'sintetica.json').write_text(json.dumps({
        'info': {'participants': [{'summonerName': 'Nombre Sintético Único'}]}}))
    frame['etiqueta'] = 'Nombre Sintético Único'
    frame.to_csv(table, index=False)
    with pytest.raises(ValueError, match='predictibilidad.csv'):
        build_demo.build(data, output)


def test_web_works_without_data_or_credentials(clean_clone):
    from src.app.server import create_app
    client = create_app().test_client()
    assert client.get('/').status_code == 200
    assert len(client.get('/api/examples').json) == 12
    assert len(client.get('/api/calculator').json['variables']) == 10


def test_consistency_default_output_is_in_analysis(tmp_path, monkeypatch):
    from scripts import check_timeline_consistency as consistency
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['check_timeline_consistency.py'])
    monkeypatch.setattr(consistency, 'check', lambda: ({}, 0))
    consistency.main()
    assert (tmp_path / 'data/analysis/consistencia.md').is_file()


@pytest.mark.parametrize('source', ['database', 'cache'])
def test_scan_checks_local_names_without_tags_read_only(tmp_path, source):
    import gzip
    import sqlite3

    data = tmp_path / 'data'
    data.mkdir()
    secret = 'Nombre Sintético Único'
    if source == 'database':
        folder = data / 'db'
        folder.mkdir()
        path = folder / 'lol.sqlite'
        with sqlite3.connect(path) as conn:
            conn.execute('CREATE TABLE players (summoner_name TEXT, puuid TEXT, match_id TEXT)')
            conn.execute('INSERT INTO players VALUES (?, ?, ?)', (secret, 'Z' * 80, 'LA1_123456789'))
    else:
        folder = data / 'raw/matches'
        folder.mkdir(parents=True)
        path = folder / 'partida.json.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump({'info': {'participants': [{'summonerName': secret}]}}, stream)
    original = path.read_bytes()
    output = tmp_path / 'demo'
    output.mkdir()
    (output / 'tabla.csv').write_text('nombre\n' + secret + '\n', encoding='utf-8')
    (output / 'ejemplo.json').write_text(json.dumps({'valor': secret}), encoding='utf-8')
    joblib.dump({'nested': [secret]}, output / 'modelo.joblib', compress=3)
    from PIL import Image, PngImagePlugin
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('Description', secret)
    Image.new('RGB', (1, 1)).save(output / 'figura.png', pnginfo=metadata)
    assert build_demo.scan_demo(output) == []
    assert build_demo.scan_demo(output, data) == ['ejemplo.json', 'figura.png', 'modelo.joblib', 'tabla.csv']
    assert path.read_bytes() == original
    assert list(folder.iterdir()) == [path]


def test_build_rejects_output_ancestor_before_removing_files(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    sentinel = data / 'conservar.txt'
    sentinel.write_text('conservar')
    with pytest.raises(ValueError, match='fuera de los datos'):
        build_demo.build(data, tmp_path)
    assert sentinel.read_text() == 'conservar'


def test_scan_exempts_only_verified_public_champion_fields(tmp_path):
    import sqlite3

    data = tmp_path / 'data/db'
    data.mkdir(parents=True)
    examples = json.loads((DEMO / 'ejemplos.json').read_text())
    name = examples[0]['campeones']['azul'][0]
    with sqlite3.connect(data / 'lol.sqlite') as conn:
        conn.execute('CREATE TABLE players (summoner_name TEXT)')
        conn.executemany('INSERT INTO players VALUES (?)', [(name,), ('azul',), ('rojo',), ('baron',), ('objetivo',)])
    output = tmp_path / 'demo'
    output.mkdir()
    shutil.copyfile(DEMO / 'champion_16.22.1.json', output / 'champion_16.22.1.json')
    (output / 'ejemplos.json').write_text(json.dumps(examples))
    shutil.copyfile(DEMO / 'analysis/objetivos.csv', output / 'objetivos.csv')
    assert build_demo.scan_demo(output, data.parent) == []
    (output / 'tabla.csv').write_text('nombre\n' + name + '\n')
    assert build_demo.scan_demo(output, data.parent) == ['tabla.csv']
    examples[0]['extra'] = 'azul'
    (output / 'ejemplos.json').write_text(json.dumps(examples))
    assert 'ejemplos.json' in build_demo.scan_demo(output, data.parent)
    catalogue = output / 'champion_16.22.1.json'
    document = json.loads(catalogue.read_text())
    document['extra'] = name
    catalogue.write_text(json.dumps(document))
    assert 'champion_16.22.1.json' in build_demo.scan_demo(output, data.parent)
