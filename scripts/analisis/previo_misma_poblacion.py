#!/usr/bin/env python3
"""Corrección local del previo congelado. No ajusta modelos ni ejecuta --final."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2]/'data/analysis/tmp/mplconfig'))
import csv
import hashlib
import io
import json
import socket
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

def no_network(*args, **kwargs):
    raise RuntimeError('Esta corrección solo permite archivos locales')

socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.models import predict
from src.riot_client import cache
from src.features.timeline import STATE_FEATURES, participant_teams, state_at
from scripts.analisis.minuto import CUT, KEY, metrics, losses

RUN = ROOT / 'data'
CSV = ROOT / 'data/analysis/predictibilidad.csv'
GUARD = RUN / 'analysis/evaluacion_iniciada.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protected = [GUARD, ROOT / 'data/models/modelo_final_2026-11-09.joblib',
                 ROOT / 'data/models/modelos_en_partida.joblib',
                 ROOT / 'data/models/segunda_prueba.json']
    before = [digest(p) for p in protected]
    frozen = joblib.load(protected[1])
    statebundle = joblib.load(protected[2])
    original = predict._table_for(ROOT / 'data/export', frozen['pool'], frozen['champion_positions'])
    excluded = set(map(str, frozen['match_ids'])) | set(map(str, frozen['development_ids']))
    test = original[(~original.match_id.astype(str).isin(excluded)) &
                    (original.game_start_ts > max(frozen['last_match_ts'], predict.SECOND_CUTOFF_TS))]
    assert len(test) == 1264
    candidates = test[test.game_start_ts > CUT].set_index('match_id').sort_index()
    # La segunda prueba está congelada en el export. No enumerar nuevas descargas.
    assert len(candidates) == 416
    with sqlite3.connect((ROOT / 'data/db/lol.sqlite').as_uri() + '?mode=ro', uri=True) as source:
        db = sqlite3.connect(':memory:')
        source.backup(db)
    matches = pd.read_sql_query('select match_id, status, blue_win, game_start_ts from target_matches', db).set_index('match_id')
    parts = pd.read_sql_query('select match_id, team_id, win from match_participants', db)
    db.close()
    rows = []
    # Mismos controles que extract en minuto.py, para todos los cuadros 1 a 30.
    for mid, exported in candidates.iterrows():
        r = matches.loc[mid]
        assert r.status == 'history_done' and r.game_start_ts > CUT
        assert int(r.blue_win) == int(exported.blue_win)
        assert r.game_start_ts == exported.game_start_ts
        tl, match = cache.load_timeline(mid), cache.load_match(mid)
        assert tl is not None and match is not None
        assert tl['metadata']['matchId'] == match['metadata']['matchId'] == mid
        info = match['info']
        assert info['gameStartTimestamp'] == r.game_start_ts
        teams = info['teams']
        assert len(teams) == 2 and {t['teamId'] for t in teams} == {100, 200}
        assert sum(bool(t['win']) for t in teams) == 1
        assert int(next(t['win'] for t in teams if t['teamId'] == 100)) == int(r.blue_win)
        p = parts[parts.match_id == mid]
        assert len(p) == 10 and set(p.team_id) == {100, 200}
        assert all(int(x.win) == int(r.blue_win if x.team_id == 100 else 1-r.blue_win) for x in p.itertuples())
        assert participant_teams(tl) == {p['participantId']: p['teamId'] for p in info['participants']}
        timestamps = [f['timestamp'] for f in tl['info']['frames']]
        assert timestamps == sorted(timestamps) and len(timestamps) == len(set(timestamps))
        for minute in range(1, 31):
            s = state_at(tl, minute)
            if s is None:
                break
            assert len(tl['info']['frames'][minute]['participantFrames']) == 10
            assert np.isfinite(list(s.values())).all()
            if minute in KEY:
                rows.append(dict(match_id=mid, minuto=minute, blue_win=int(r.blue_win), **s))
    states = pd.DataFrame(rows)
    raw_csv = CSV.read_text()
    existing = pd.read_csv(io.StringIO(raw_csv))
    assert not (existing.modelo == 'previo_misma_poblacion').any(), 'Corrección ya aplicada'
    pre = pd.Series(frozen['model'].predict_proba(candidates[frozen['features']])[:, 1], index=candidates.index)
    added = []
    for minute, expected_n in zip(KEY, [416, 414, 412, 382, 336]):
        g = states[states.minuto == minute].sort_values('match_id')
        assert len(g) == expected_n and not g.match_id.duplicated().any()
        y = g.blue_win.to_numpy(int)
        # Control contra los tres resultados originales, sin reajuste ni nueva selección.
        p = statebundle['models'][minute].predict_proba(g[STATE_FEATURES])[:, 1]
        old = existing[(existing.minuto == minute) & (existing.modelo == 'estado')].iloc[0]
        np.testing.assert_allclose([np.mean((p >= .5) == y), roc_auc_score(y, p), losses(y, p).mean()],
                                   [old.acierto, old.auc, old.perdida_logaritmica], rtol=0, atol=1e-12)
        added.append(dict(minuto=minute, modelo='previo_misma_poblacion',
                          **metrics(y, pre.loc[g.match_id].to_numpy()),
                          fuente='scripts/analisis/previo_misma_poblacion.py'))
    # Conservar literalmente todas las filas originales.
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(existing.columns), lineterminator='\n')
    writer.writerows(added)
    CSV.write_text(raw_csv + ('' if raw_csv.endswith('\n') else '\n') + buffer.getvalue())
    frame = pd.read_csv(CSV)
    plot(frame)
    correction = report(added)
    (RUN / 'analysis/previo_misma_poblacion.md').write_text(correction)
    results = RUN / 'analysis/results_minuto.md'
    text = results.read_text() if results.exists() else '# Detalle completo\n'
    assert '# Corrección según el recheck' not in text
    text = text.replace('# Detalle completo', correction + '\n# Detalle completo', 1)
    text = text.replace('El punto cero publicado usa la segunda prueba completa. Los puntos siguientes usan su intersección con timelines posteriores al corte y supervivencia a cada cuadro. No son la misma población.',
                        'El punto cero publicado usa la segunda prueba completa. Los puntos de estado usan su intersección con timelines posteriores al corte y supervivencia a cada cuadro. La corrección anterior añade el previo en la misma población de cada minuto y la figura 3 actualizada separa el punto cero.')
    results.write_text(text)
    assert [digest(p) for p in protected] == before
    audit = dict(bootstrap=2000, semilla=42, ic='percentiles 2,5 y 97,5',
                 recorte_probabilidades=[1e-6, 1-1e-6],
                 n_por_minuto={str(r['minuto']): r['n'] for r in added},
                 control_estado='Acierto, AUC y pérdida reproducidos con tolerancia 1e-12',
                 controles_extract='Cuadros 1 a 30, equipos, ganador, fecha y participantes locales',
                 filas_originales_conservadas=CSV.read_text().startswith(raw_csv),
                 guardia_y_modelos_sin_cambios=True, red_bloqueada=True,
                 salidas_solo_agregadas=True)
    (RUN / 'analysis/previo_misma_poblacion_auditoria.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(correction)

def plot(frame):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    state = frame[frame.modelo == 'estado']
    prior = frame[frame.modelo == 'previo_misma_poblacion']
    published = frame[frame.modelo == 'previo_publicado'].iloc[0]
    for axis, field, label in zip(axes, ['acierto', 'auc'], ['Proporción de aciertos', 'ROC AUC']):
        for d, color, title, style in [(state, '#146B8C', 'Estado con cuadro disponible', '-'),
                                        (prior, '#B35B32', 'Previo en las mismas partidas de cada minuto', 'o--')]:
            axis.plot(d.minuto, d[field], style, color=color, label=title)
            axis.fill_between(d.minuto, d[field+'_ic_bajo'], d[field+'_ic_alto'], alpha=.13, color=color)
        axis.errorbar([0], [published[field]],
                      yerr=[[published[field]-published[field+'_ic_bajo']], [published[field+'_ic_alto']-published[field]]],
                      fmt='D', color='#643E78', capsize=3, label='Previo publicado, n=1.264 (población distinta)')
        axis.annotate('n=1.264', (0, published[field]), xytext=(5, -25), textcoords='offset points', fontsize=8)
        axis.set(xlabel='Minuto del cuadro disponible', ylabel=label, ylim=(.45, 1.01), xticks=[0, 5, 10, 15, 20, 25])
        axis.grid(alpha=.2)
    gold = frame[frame.modelo == 'regla_oro']
    axes[0].plot(gold.minuto, gold.acierto, 's:', color='#757575', label='Regla del oro')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=2, fontsize=9)
    fig.suptitle('Predictibilidad según el minuto: comparación en las mismas partidas')
    fig.text(.5, .045, 'n a los 5, 10, 15, 20 y 25 minutos: 416, 414, 412, 382 y 336. La población cambia entre minutos.', ha='center', fontsize=9)
    fig.text(.5, .015, 'Fuente: data/analysis/predictibilidad.csv. IC del 95 %, bootstrap de partidas. El punto cero se presenta por separado.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .23, 1, .94))
    fig.savefig(ROOT / 'data/analysis/figuras/figura_3_predictibilidad.png', dpi=180)
    plt.close(fig)

def report(rows):
    def f(v):
        return f'{v:.4f}'.replace('.', ',')
    lines = ['# Corrección según el recheck', '',
             'El previo congelado se calculó sobre las mismas partidas que usa el estado en cada minuto. No se reajustó ningún modelo ni se ejecutó la evaluación final o su guardia. No se hicieron llamadas a Riot. Solo se escribieron resultados agregados.', '',
             'Fuente de cada celda: `data/analysis/predictibilidad.csv`, filas `previo_misma_poblacion`. Código: `scripts/analisis/previo_misma_poblacion.py`. Insumos locales: `data/models/modelo_final_2026-11-09.joblib`, `data/export/`, `data/raw/timelines/`, `data/raw/matches/` y copia en memoria de `data/db/lol.sqlite`. Controles: `analysis/previo_misma_poblacion_auditoria.json`.', '',
             '| Minuto de la submuestra | n | Acierto (IC 95 %) | AUC (IC 95 %) | Pérdida logarítmica (IC 95 %) |',
             '|---|---:|---|---|---|']
    for r in rows:
        cells = [f(r[k]) + ' (' + f(r[k+'_ic_bajo']) + ' a ' + f(r[k+'_ic_alto']) + ')' for k in ['acierto', 'auc', 'perdida_logaritmica']]
        lines.append(f"| {r['minuto']} | {r['n']} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Los IC son percentiles del 95 % con 2.000 remuestras de partidas y semilla 42, reiniciada por minuto. Se mantiene el orden de partidas del análisis original. Se usa umbral 0,5 para acierto y recorte de probabilidades a [0,000001, 0,999999] para pérdida. En cada minuto hubo 2.000 remuestras válidas para AUC. Estos intervalos no corrigen la dependencia entre partidas con jugadores compartidos.', '',
              'El punto publicado de 1.264 partidas se conserva aparte: acierto 0,5704, AUC 0,5891 y pérdida 0,6792, tomados de `data/models/segunda_prueba.json`. El previo en las 416 partidas del minuto 5 acierta 0,5457, con AUC 0,5714. La diferencia frente a la prueba completa es de -2,47 puntos porcentuales de acierto y -0,0177 de AUC. Es una diferencia descriptiva entre poblaciones, no una prueba de significación.', '',
              'Las cinco submuestras se reconstruyeron desde la segunda prueba congelada, el corte y los cuadros disponibles, con los controles originales de fechas, ganador y equipos. Los tamaños coinciden y los tres resultados del estado se reprodujeron con el bundle guardado a tolerancia 1e-12, sin reajuste. La comparación entre previo y estado usa las mismas partidas dentro de cada minuto. Entre minutos sigue cambiando la población por disponibilidad de cuadro y duración. Las variaciones entre minutos no son cambios pareados de una cohorte fija ni identifican un minuto universal.', '',
              'La figura actualizada es `data/analysis/figuras/figura_3_predictibilidad.png`. La línea del previo conecta sus resultados en las cinco submuestras y el punto publicado de 1.264 se muestra aislado y etiquetado. La tabla original que sigue se conserva como registro y se complementa con las cinco filas anteriores.', '',
              'Reproducción desde la raíz: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/analisis/previo_misma_poblacion.py`. El script rechaza una segunda inserción y bloquea conexiones de red.']
    return '\n'.join(lines)+'\n'

if __name__ == '__main__':
    main()
