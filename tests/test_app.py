import json
from types import SimpleNamespace

import pytest
from src.app import logic
from src.features.timeline import STATE_FEATURES
from src.models import ingame, predict
from scripts import build_demo_examples as demo


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests
    def denied(*args, **kwargs):
        raise AssertionError("Las pruebas de la app no pueden acceder a la red")
    monkeypatch.setattr(requests.sessions.Session, "request", denied)


@pytest.fixture
def small_bundle():
    return ingame.train([{"minute": minute, "match_id": f"SYN_{i}", "blue_win": i % 2,
                          **dict.fromkeys(STATE_FEATURES, 1 if i % 2 else -1)}
                         for minute in ingame.CURVE_MINUTES for i in range(4)])


def test_examples_missing_corrupt_and_private(tmp_path):
    path = tmp_path / "examples.json"
    assert logic.load_examples(path) == []
    path.write_text("incompleto")
    assert logic.load_examples(path) == []
    secret = "nombre_privado#Etiqueta"
    path.write_text(json.dumps([{"etiqueta": secret, "match_id_interno": "SYN_99", "puuid": "private-player-token",
                                "gana_azul": True, "prob_previa": None,
                                "campeones": {"azul": [secret], "rojo": []},
                                "curva": [{"minuto": 1, "prob_azul": .6, "puuid": secret}],
                                "eventos": [{"minuto": 1, "tipo": secret, "equipo": "azul"}]}]))
    examples = logic.load_examples(path)
    assert examples[0]["etiqueta"] == "Partida 1"
    public = json.dumps(examples)
    assert all(s not in public for s in [secret, "private-player-token", "SYN_99", "puuid", "match_id_interno"])


def test_calculator_real_contract(small_bundle):
    assert logic.calculator(small_bundle, 15, 100, 2, 1, 1) > .5
    assert logic.calculator(small_bundle, 15, -100, -2, -1, -1) < .5
    assert logic.calculator(small_bundle, 40) == logic.calculator(small_bundle, 30)


def test_replay_exact_two_calls_no_prior(small_bundle):
    raw = {"metadata": {"matchId": "SYN_1"}, "info": {"teams": [{"teamId": 100, "win": True}],
            "participants": [{"participantId": i, "teamId": 100 if i <= 5 else 200, "championId": 1, "puuid": "private"} for i in range(1, 11)]}}
    timeline = {"info": {"participants": [{"participantId": i} for i in range(1, 11)], "frames": [
        {"timestamp": m * 60000, "participantFrames": {str(i): {"totalGold": i, "xp": i, "level": 1, "minionsKilled": 1, "jungleMinionsKilled": 0} for i in range(1, 11)},
         "events": [{"type": "ELITE_MONSTER_KILL", "monsterType": "DRAGON", "killerTeamId": 100, "timestamp": 60000}] if m == 1 else []} for m in range(3)]}}
    calls = []
    endpoints = SimpleNamespace(get_match=lambda key: (calls.append("match") or raw), get_match_timeline=lambda key: (calls.append("timeline") or timeline))
    result = logic.replay("SYN_1", small_bundle, endpoints)
    assert calls == ["match", "timeline"]
    assert result["curva"][0]["minuto"] == 1
    assert result["eventos"] == [{"minuto": 1, "tipo": "dragón", "equipo": "azul"}]
    assert "private" not in json.dumps(result)
    assert "SYN_1" not in json.dumps(result)
    with pytest.raises(ValueError):
        logic.replay("../not-a-match", small_bundle, endpoints)


def test_live_uses_no_persistence_and_safe_output(monkeypatch):
    captured = {}
    def fake(riot_id, bundle, **kwargs):
        captured.update(kwargs)
        kwargs["progress"]("historial")
        return {"prob_blue": .6, "teams": {100: [("TOP", "Annie")]}, "puuid": "private-token", "feature_values": {}}
    monkeypatch.setattr(predict, "predict_live", fake)
    monkeypatch.setattr(predict, "_confidence_line", lambda *a: "Confianza media")
    monkeypatch.setattr(logic, "contributions", lambda *a: [{"variable": "private", "aporte": .2, "frase": "La racha favorece al azul."}])
    messages = []
    result = logic.live("Nombre#Etiqueta", {}, messages.append)
    assert captured["persist"] is False
    assert messages == ["historial"]
    assert "private" not in json.dumps(result)
    assert "Nombre" not in json.dumps(result)


def test_findings_missing(tmp_path):
    assert logic.findings_paths(tmp_path, tmp_path) == {"tablas": [], "figuras": []}
    (tmp_path / "resultado.csv").write_text("n,acierto\n10,0.6")
    assert len(logic.findings_paths(tmp_path, tmp_path)["tablas"]) == 1


def test_demo_selection_reproducible_and_unique():
    candidates = [{"match_id": f"SYN_{category}_{i}", "categoria": category} for category in demo.QUOTAS for i in range(8)]
    selected = demo.select_examples(candidates)
    assert selected == demo.select_examples(list(reversed(candidates)))
    assert len(selected) == len({r["match_id"] for r in selected}) == 12
    for category, count in demo.QUOTAS.items():
        assert sum(r["categoria"] == category for r in selected) == count
    with pytest.raises(ValueError):
        demo.select_examples([])


def test_live_game_no_log_mode_returns_factors_without_writes(monkeypatch, tmp_path):
    from src.agents import history_builder
    from src.riot_client import endpoints
    from src.features.build import FINAL_FEATURES, POSITIONS
    monkeypatch.setattr(history_builder, "player_history", lambda *a: [])
    monkeypatch.setattr(endpoints, "is_firewall_blocked", lambda _: False)
    def denied(*a, **kw):
        raise AssertionError("El modo web no puede escribir registros")
    monkeypatch.setattr(predict, "_log_context", denied)
    monkeypatch.setattr(predict, "log_prediction", denied)
    participants = [{"puuid": f"private-{i}", "teamId": 100 if i < 5 else 200, "championId": 1} for i in range(10)]
    values = dict.fromkeys(FINAL_FEATURES, 0.)
    monkeypatch.setattr(predict, "score_game", lambda *a: {"prob_blue": .6, "prob_red": .4,
                       "roles": {f"private-{i}": POSITIONS[i % 5] for i in range(10)},
                       "feature_values": values, "jugadores_sin_historial": 10})
    result = predict.predict_live_game({"participants": participants, "gameId": 123, "gameQueueConfigId": 420},
             {"version": 2, "roles": "inferred"}, names={1: "Annie"}, persist=False, progress=lambda _: None, log_path=tmp_path / "log.csv")
    assert result["feature_values"] == values
    assert result["logged"] is False
    assert list(tmp_path.iterdir()) == []


def test_findings_table_hides_identifiers(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("puuid,riot_id,match_id,n,categoria\nprivate,Name#TAG,SYN_1,10,dragón\n")
    result = logic.read_findings_table(path)
    assert list(result) == ["n", "categoria"]
    assert "private" not in result.to_json()


def test_demo_build_only_reads_cache(monkeypatch):
    candidates = [{"match_id": f"SYN_{c}_{i}", "categoria": c, "prob_previa": .6} for c in demo.QUOTAS for i in range(5)]
    monkeypatch.setattr(demo.cache, "load_match", lambda key: {"id": key})
    monkeypatch.setattr(demo.cache, "load_timeline", lambda key: {})
    monkeypatch.setattr(demo, "example_from_match", lambda match, timeline, bundle, prior, index: {"etiqueta": f"Partida {index}", "prob_previa": prior})
    output = demo.build(candidates, {})
    assert len(output) == 12
    assert output[0] == {"etiqueta": "Partida 1", "prob_previa": .6}


def test_demo_team_history_uses_only_earlier_non_remake_games(tmp_path):
    import gzip
    import pandas as pd
    export = tmp_path / 'export'
    export.mkdir()
    pd.DataFrame([{'match_id': 'SYN_1', 'status': 'history_done',
                   'game_start_ts': 1000000, 'patch': '16.22'}]).to_csv(export / 'matches.csv', index=False)
    participants, histories = [], []
    for player in range(10):
        participants.append({'match_id': 'SYN_1', 'player_id': player, 'team_id': 100 if player < 5 else 200,
                             'team_position': 'TOP', 'champion_id': 1})
        for rank, timestamp, remake in [(1, 900000, 0), (2, 1100000, 0), (3, 800000, 1)]:
            histories.append({'target_match_id': 'SYN_1', 'player_id': player, 'history_rank': rank,
                              'game_start_ts': timestamp, 'remake': remake, 'win': int(player < 5),
                              'team_position': 'TOP', 'champion_id': 1, 'kills': 4, 'assists': 2, 'deaths': 2})
    pd.DataFrame(participants).to_csv(export / 'participants.csv', index=False)
    pd.DataFrame(histories).to_csv(export / 'player_history.csv', index=False)
    folder = tmp_path / 'raw/timelines'
    folder.mkdir(parents=True)
    with gzip.open(folder / 'SYN_1.json.gz', 'wt') as stream:
        json.dump({'info': {'frames': [{'events': [{'type': 'BUILDING_KILL',
            'buildingType': 'INHIBITOR_BUILDING', 'teamId': 200, 'timestamp': 1200000}]}]}}, stream)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = demo.enrich_examples([{'match_id_interno': 'SYN_1'}], tmp_path)[0]
    assert result['historial'] == {
        'azul': {'winrate_10': 1., 'kda_medio': 3., 'partidas_con_campeon': 1.},
        'rojo': {'winrate_10': 0., 'kda_medio': 3., 'partidas_con_campeon': 1.}}
    assert result['eventos'] == [{'minuto': 20., 'tipo': 'inhibidor', 'equipo': 'azul'}]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
