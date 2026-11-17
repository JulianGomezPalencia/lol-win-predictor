"""Guarda retratos de los ejemplos para usar la interfaz sin conexión."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VERSION = '16.22.1'


def download(address):
    return subprocess.run(
        ['curl', '--fail', '--silent', '--show-error', '--location',
         '--retry', '2', '--max-time', '30', address],
        check=True, capture_output=True,
    ).stdout


def main():
    catalog = json.loads((ROOT / f'demo/champion_{VERSION}.json').read_text())['data']
    names = {c['name']: c['id'] for c in catalog.values()}
    examples = json.loads((ROOT / 'demo/ejemplos.json').read_text())
    champions = sorted({name for row in examples for team in row['campeones'].values() for name in team})
    destination = ROOT / 'src/app/static/champions'
    destination.mkdir(parents=True, exist_ok=True)
    mapping = {name: names[name] for name in champions}
    for champion_id in sorted(set(mapping.values())):
        path = destination / f'{champion_id}.png'
        if not path.exists():
            url = f'https://ddragon.leagueoflegends.com/cdn/{VERSION}/img/champion/{champion_id}.png'
            image = download(url)
            if not image.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError(f'Retrato no válido: {champion_id}')
            path.write_bytes(image)
    (destination / 'manifest.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(mapping)} retratos locales')


if __name__ == '__main__':
    main()
