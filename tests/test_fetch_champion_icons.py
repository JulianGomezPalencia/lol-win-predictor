"""Casos límite del generador de retratos, sin acceso a la red."""
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/fetch_champion_icons.py'
PNG = b'\x89PNG\r\n\x1a\n' + b'contenido'


@pytest.fixture
def generator(tmp_path, monkeypatch):
    namespace = runpy.run_path(str(SCRIPT))
    main = namespace['main']
    monkeypatch.setitem(main.__globals__, 'ROOT', tmp_path)
    demo = tmp_path / 'demo'
    demo.mkdir()
    (demo / 'champion_16.22.1.json').write_text(json.dumps({
        'data': {'MonkeyKing': {'name': 'Wukong', 'id': 'MonkeyKing'}}
    }))
    (demo / 'ejemplos.json').write_text(json.dumps([
        {'campeones': {'azul': ['Wukong'], 'rojo': ['Wukong']}}
    ]))
    return main, tmp_path


def test_cached_icon_is_reused_and_duplicate_names_are_mapped_once(generator, monkeypatch):
    main, root = generator
    destination = root / 'src/app/static/champions'
    destination.mkdir(parents=True)
    (destination / 'MonkeyKing.png').write_bytes(PNG)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('No debe descargar un icono existente'))

    main()

    assert (destination / 'MonkeyKing.png').read_bytes() == PNG
    assert json.loads((destination / 'manifest.json').read_text()) == {'Wukong': 'MonkeyKing'}


def test_failed_download_stops_without_using_another_host(generator, monkeypatch):
    main, root = generator
    urls = []

    def download(command, **kwargs):
        urls.append(command[-1])
        raise subprocess.CalledProcessError(22, command)

    monkeypatch.setattr(subprocess, 'run', download)
    with pytest.raises(subprocess.CalledProcessError):
        main()

    assert urls == ['https://ddragon.leagueoflegends.com/cdn/16.22.1/img/champion/MonkeyKing.png']
    assert not (root / 'src/app/static/champions/MonkeyKing.png').exists()
    assert not (root / 'src/app/static/champions/manifest.json').exists()


def test_invalid_download_does_not_publish_icon_or_manifest(generator, monkeypatch):
    main, root = generator
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=b'<html>error</html>'))

    with pytest.raises(ValueError, match='MonkeyKing'):
        main()

    destination = root / 'src/app/static/champions'
    assert not (destination / 'MonkeyKing.png').exists()
    assert not (destination / 'manifest.json').exists()


def test_unknown_champion_fails_before_downloading(generator, monkeypatch):
    main, root = generator
    (root / 'demo/ejemplos.json').write_text(json.dumps([
        {'campeones': {'azul': ['Desconocido'], 'rojo': ['Wukong']}}
    ]))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('No debe descargar un catálogo incompleto'))

    with pytest.raises(KeyError, match='Desconocido'):
        main()
    assert not (root / 'src/app/static/champions/manifest.json').exists()


def test_successful_download_publishes_icon_and_mapping(generator, monkeypatch):
    main, root = generator
    urls = []

    def download(command, **kwargs):
        urls.append(command[-1])
        return SimpleNamespace(stdout=PNG)

    monkeypatch.setattr(subprocess, 'run', download)
    main()

    destination = root / 'src/app/static/champions'
    assert urls == ['https://ddragon.leagueoflegends.com/cdn/16.22.1/img/champion/MonkeyKing.png']
    assert (destination / 'MonkeyKing.png').read_bytes() == PNG
    assert json.loads((destination / 'manifest.json').read_text()) == {'Wukong': 'MonkeyKing'}
