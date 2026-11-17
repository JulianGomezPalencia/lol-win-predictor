"""Contratos web con demo aislada y red bloqueada."""
import json
import shutil
from pathlib import Path

import pytest
from src.app import logic
from src.app.server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    shutil.copytree(Path(__file__).resolve().parents[1] / 'demo', tmp_path / 'demo')
    monkeypatch.setattr(logic, 'ROOT', tmp_path)
    monkeypatch.delenv('RIOT_API_KEY', raising=False)
    monkeypatch.setattr('src.app.server.load_dotenv', lambda: None)
    import requests
    monkeypatch.setattr(requests.sessions.Session, 'request',
                        lambda *a, **kw: pytest.fail('La demo no consulta la red'))
    app = create_app()
    app.config['TESTING'] = True
    return app.test_client()


@pytest.mark.parametrize('path', ['/', '/api/examples', '/api/examples/0/prediction',
    '/api/examples/0/winner', '/api/examples/0/replay', '/api/findings',
    '/api/calculator', '/api/live', '/api/comeback'])
def test_demo_routes_and_privacy(client, path):
    response = client.get(path)
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    for secret in ('puuid', 'summonerName', 'riot_id', 'match_id', 'LA1_', 'RGAPI-'):
        assert secret not in text
    assert response.headers['Cache-Control'] == 'no-store'


def test_reveals_are_separate(client):
    data = client.get('/api/examples').json
    assert len(data) == 12
    assert all(set(row) == {'etiqueta', 'campeones', 'historial'} for row in data)
    assert set(client.get('/api/examples/0/prediction').json) == {'prob_azul'}
    assert set(client.get('/api/examples/0/winner').json) == {'gana_azul'}
    assert set(client.get('/api/examples/0/replay').json) == {'curva', 'eventos'}
    assert client.get('/api/examples/999/replay').status_code == 404
    assert client.get('/api/examples/0/private').status_code == 404


def test_calculator_explanation_and_invalid_values(client):
    names = [v['nombre'] for v in client.get('/api/calculator').json['variables']]
    values = dict.fromkeys(names, 0.)
    result = client.post('/api/calculator', json=values)
    assert result.status_code == 200
    assert 0 < result.json['prob_azul'] < 1
    assert len(result.json['factores']) == 10
    for value in ('private#tag', float('inf'), float('nan'), None, [], True, 10001):
        assert client.post('/api/calculator', json={**values, names[0]: value}).status_code == 400
    for payload in ([], {}, {'unexpected': 'private#tag'}):
        assert client.post('/api/calculator', json=payload).status_code == 400


def test_no_credentials_live_and_sanitized_errors(client, monkeypatch):
    assert client.post('/api/live', json={'jugador': 'Privado#Etiqueta'}).json['disponible'] is False
    monkeypatch.setenv('RIOT_API_KEY', 'synthetic')
    assert client.get('/api/live').json['disponible'] is False
    full = logic.load_prior_model()
    full.update(pool={}, champion_positions={})
    monkeypatch.setattr(logic, 'load_prior_model', lambda: full)
    # Use a fresh app so its cached model reflects the complete local bundle.
    client = create_app().test_client()
    assert client.get('/api/live').json['disponible'] is True
    def failure(*a):
        raise ValueError('Privado#Etiqueta private-token')
    monkeypatch.setattr(logic, 'live', failure)
    response = client.post('/api/live', json={'jugador': 'Privado#Etiqueta'})
    assert response.status_code == 503
    assert 'Privado' not in response.text and 'private-token' not in response.text
    assert client.post('/api/live', json=[]).status_code == 400


def test_comeback_is_real_and_findings_have_samples(client):
    index = client.get('/api/comeback').json['indice']
    replay = client.get(f'/api/examples/{index}/replay').json
    winner = client.get(f'/api/examples/{index}/winner').json['gana_azul']
    p15 = next(p['prob_azul'] for p in replay['curva'] if p['minuto'] == 15)
    assert (p15 >= .5) != winner
    findings = client.get('/api/findings').json
    assert {r['objetivo'] for r in findings['objetivos']} == {'champion', 'dragon', 'tower', 'baron', 'inhibitor', 'sangre_torre_dragon'}
    assert all(r['n'] > 0 for r in findings['objetivos'])
    assert len([r for r in findings['predictibilidad'] if r['modelo'] == 'previo_misma_poblacion']) == 5
    assert any(e['tipo'] == 'inhibidor'
               for i in range(12) for e in client.get(f'/api/examples/{i}/replay').json['eventos'])


def test_private_values_never_leave_routes(client, monkeypatch):
    secrets = ['Persona Sintética Única', 'PersonaSintética#Prueba', 'X' * 80, 'LA1_987654321']
    path = logic.ROOT / 'demo/ejemplos.json'
    rows = json.loads(path.read_text())
    for row in rows:
        row.update(dict(zip(['nombre', 'riot_id', 'puuid', 'match_id_interno'], secrets)))
        row['etiqueta'] = secrets[0]
        row['campeones']['azul'][0] = secrets[1]
        row['historial']['azul'].update(extra=secrets[0], kda_medio=secrets[2])
        row['curva'][0]['extra'] = secrets[3]
        row['eventos'].append({'minuto': 1, 'tipo': secrets[0], 'equipo': secrets[1]})
    path.write_text(json.dumps(rows))
    responses = [client.get('/') , client.get('/api/examples'), client.get('/api/comeback'),
                 client.get('/api/findings'), client.get('/api/calculator'), client.get('/api/live')]
    for i in range(len(rows)):
        responses.extend(client.get(f'/api/examples/{i}/{action}')
                         for action in ('prediction', 'winner', 'replay'))
    values = dict.fromkeys(logic.load_prior_model()['features'], 0.)
    responses.append(client.post('/api/calculator', json=values))
    monkeypatch.setenv('RIOT_API_KEY', 'synthetic')
    full = logic.load_prior_model()
    full.update(pool={}, champion_positions={})
    monkeypatch.setattr(logic, 'load_prior_model', lambda: full)
    from src.models import predict
    monkeypatch.setattr(predict, 'predict_live', lambda *a, **kw: {
        'prob_blue': .6, 'teams': {100: [(secrets[0], secrets[1])], 200: [('TOP', 'Annie')]},
        'feature_values': values, **dict(zip(['nombre', 'riot_id', 'puuid', 'match_id'], secrets))})
    live_client = create_app().test_client()
    responses.append(live_client.get('/api/live'))
    responses.append(live_client.post('/api/live', json={'jugador': secrets[1]}))
    assert responses[-1].json['prob_azul'] == .6
    for response in responses:
        assert response.status_code == 200
        decoded = json.dumps(response.json, ensure_ascii=False) if response.is_json else response.text
        assert all(secret not in decoded for secret in secrets)


def test_missing_products_and_inhibitor(client, monkeypatch):
    monkeypatch.setattr(logic, 'load_examples', lambda: [])
    assert client.get('/api/examples').json == []
    assert client.get('/api/examples/0/winner').status_code == 404
    event = {'type': 'BUILDING_KILL', 'buildingType': 'INHIBITOR_BUILDING', 'teamId': 200, 'timestamp': 900000}
    assert logic.replay_events({'info': {'frames': [{'events': [event]}]}}) == [
        {'minuto': 15, 'tipo': 'inhibidor', 'equipo': 'azul'}]
    safe = logic.public_example({'gana_azul': True, 'curva': [], 'eventos': [
        {'minuto': 15, 'tipo': 'inhibidor', 'equipo': 'azul'}]})
    assert safe['eventos'][0]['tipo'] == 'inhibidor'
