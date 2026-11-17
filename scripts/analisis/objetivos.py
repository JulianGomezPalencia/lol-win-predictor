#!/usr/bin/env python3
"""Protocolo C. Solo lectura local, una pasada, salidas agregadas autorizadas."""
import os
os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__file__).resolve().parents[2]/'data/analysis/tmp/mplconfig'))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import csv
import gzip
import hashlib
import json
import math
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data/analysis/objetivos.csv'
FIG = ROOT / 'data/analysis/figuras/figura_5_objetivos.png'
REPORT = ROOT / 'data/analysis/objetivos.md'
LABELS = {'champion':'Primera sangre', 'tower':'Primera torre', 'dragon':'Primer dragón',
          'horde':'Primeras larvas', 'riftHerald':'Primer heraldo', 'baron':'Primer barón',
          'inhibitor':'Primer inhibidor', 'atakhan':'Primer Atakhan', 'voidGator':'Objetivo voidGator'}
STATES = ['valido','ausente','first_ausente','first_invalido','sin_primero','contradictorio']
COMBO = ('champion','tower','dragon')

def wilson(k, n):
    if not n:
        return None, None
    z = NormalDist().inv_cdf(.975)
    p = k / n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return max(0.,center-half)*100, min(1.,center+half)*100

def parse_file(path):
    # Se devuelven solo identificadores de partida cifrados en memoria y agregados.
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        d = json.loads(gzip.decompress(raw))
    except Exception:
        return ('ilegible', None, None)
    if not isinstance(d, dict) or not isinstance(d.get('info'), dict):
        return ('estructura_invalida', digest, None)
    info = d['info']
    meta = d.get('metadata') or {}
    mid = meta.get('matchId') if isinstance(meta, dict) else None
    if not mid and info.get('gameId') is not None and info.get('platformId'):
        mid = f"{info['platformId']}_{info['gameId']}"
    if not isinstance(mid, str) or not mid:
        return ('sin_id_partida', digest, None)
    key = hashlib.sha256(mid.encode()).digest()
    if info.get('queueId') != 420:
        return ('otra_cola', digest, (key,))
    duration = info.get('gameDuration')
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        return ('duracion_invalida', digest, (key,))
    if duration < 300:
        return ('menos_5_minutos', digest, (key,))
    teams = info.get('teams')
    if (not isinstance(teams, list) or len(teams) != 2 or
        any(not isinstance(t, dict) for t in teams) or
        {t.get('teamId') for t in teams} != {100,200}):
        return ('equipos_invalidos', digest, (key,))
    teams = sorted(teams, key=lambda t:t['teamId'])
    if any(type(t.get('win')) is not bool for t in teams) or sum(t['win'] for t in teams) != 1:
        return ('ganador_invalido', digest, (key,))
    objectives = [t.get('objectives') for t in teams]
    if any(not isinstance(x, dict) for x in objectives):
        return ('estructura_objetivos_invalida', digest, (key,))
    result = {}
    for name in set(objectives[0]) | set(objectives[1]):
        vals = [x.get(name) for x in objectives]
        if any(name not in x for x in objectives):
            result[name] = ('ausente',None)
        elif any(not isinstance(x,dict) or 'first' not in x for x in vals):
            result[name] = ('first_ausente',None)
        elif any(type(x['first']) is not bool for x in vals):
            result[name] = ('first_invalido',None)
        else:
            flags = [x['first'] for x in vals]
            if sum(flags) == 0:
                result[name] = ('sin_primero',None)
            elif sum(flags) == 2:
                result[name] = ('contradictorio',None)
            else:
                result[name] = ('valido', 'azul' if flags[0] else 'rojo')
    winner = 'azul' if teams[0]['win'] else 'rojo'
    return ('incluida',digest,(key,winner,result,str(info.get('gameVersion','ausente')),info.get('gameStartTimestamp')))

def parse_batch(paths):
    return [parse_file(p) for p in paths]

def fmt(x, digits=2):
    return f'{x:,.{digits}f}'.replace(',','X').replace('.',',').replace('X','.')

def main():
    start = time.perf_counter()
    capture = datetime.now(timezone.utc).isoformat()
    paths = sorted((ROOT/'data/raw/matches').glob('*.gz'))
    workers = min(4, max(1, (os.cpu_count() or 3)-2))
    batches = [paths[i:i+256] for i in range(0,len(paths),256)]
    counts = Counter({s: 0 for s in ('ilegible','estructura_invalida','sin_id_partida','otra_cola','duracion_invalida','menos_5_minutos','equipos_invalidos','ganador_invalido','estructura_objetivos_invalida','duplicada','incluida')})
    seen = set()
    records = []
    manifest = hashlib.sha256()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(parse_batch,batches):
            for status,digest,record in batch:
                manifest.update((digest or 'ilegible').encode())
                if record is not None:
                    key = record[0]
                    if key in seen:
                        counts['duplicada'] += 1
                        continue
                    seen.add(key)
                counts[status] += 1
                if status == 'incluida':
                    records.append(record[1:])
    elapsed = time.perf_counter()-start
    n = len(records)
    assert sum(counts.values()) == len(paths)
    objectives = sorted({name for _,obj,_,_ in records for name in obj})
    audit = {name:Counter() for name in objectives}
    tally = Counter()
    patterns = Counter()
    pattern_wins = Counter()
    combo_missing = Counter()
    versions = Counter()
    stamps = []
    for winner,obj,version,stamp in records:
        versions[version] += 1
        if isinstance(stamp,(int,float)) and math.isfinite(stamp):
            stamps.append(stamp)
        for name in objectives:
            status,side = obj.get(name,('ausente',None))
            audit[name][status] += 1
            if status == 'valido':
                tally[name,side,'n'] += 1
                tally[name,side,'win'] += int(side==winner)
        statuses = [obj.get(name,('ausente',None)) for name in COMBO]
        if any(s != 'valido' for s,_ in statuses):
            combo_missing['no_evaluable'] += 1
            for name,(s,_) in zip(COMBO,statuses):
                if s != 'valido':
                    combo_missing[name,s] += 1
        else:
            pattern = ''.join('A' if side=='azul' else 'R' for _,side in statuses)
            patterns[pattern] += 1
            pattern_wins[pattern] += int(winner=='azul')
    rows = []
    def add(kind,name,side,pattern,status,size,wins=None,denom=n):
        low,high = wilson(wins,size) if wins is not None else (None,None)
        rows.append(dict(tipo=kind,objetivo=name,etiqueta=LABELS.get(name,name),lado=side,patron=pattern,
                         estado=status,n=size,victorias=wins,porcentaje_victoria=100*wins/size if wins is not None and size else None,
                         ic95_inferior=low,ic95_superior=high,n_referencia=denom))
    for name in objectives:
        for side in ('total','azul','rojo'):
            sides = ('azul','rojo') if side=='total' else (side,)
            size = sum(tally[name,s,'n'] for s in sides)
            wins = sum(tally[name,s,'win'] for s in sides)
            add('objetivo',name,side,'','valido',size,wins)
        for status in STATES[1:]:
            add('cobertura_objetivo',name,'','',status,audit[name][status])
        assert sum(audit[name].values())==n
    same_n = patterns['AAA']+patterns['RRR']
    same_w = pattern_wins['AAA']+patterns['RRR']-pattern_wins['RRR']
    for side,pat in [('total',''),('azul','AAA'),('rojo','RRR')]:
        size = same_n if side=='total' else patterns[pat]
        wins = same_w if side=='total' else (pattern_wins[pat] if side=='azul' else patterns[pat]-pattern_wins[pat])
        add('combinacion_mismo_equipo','sangre_torre_dragon',side,pat,'valido',size,wins)
    for pattern in ('AAA','AAR','ARA','ARR','RAA','RAR','RRA','RRR'):
        add('combinacion_patron','sangre_torre_dragon','azul',pattern,'valido',patterns[pattern],pattern_wins[pattern],sum(patterns.values()))
    add('cobertura_combinacion','sangre_torre_dragon','','','no_evaluable',combo_missing['no_evaluable'])
    add('cobertura_combinacion','sangre_torre_dragon','','','repartidos',sum(patterns.values())-same_n)
    for (name,status),size in [(k,v) for k,v in combo_missing.items() if isinstance(k,tuple)]:
        add('ausencia_combinacion',name,'','',status,size)
    for status,size in sorted(counts.items()):
        add('auditoria_archivos','','','',status,size,denom=len(paths))
    assert sum(patterns.values())+combo_missing['no_evaluable']==n
    assert all(r['victorias'] is None or 0<=r['victorias']<=r['n'] for r in rows)
    CSV.parent.mkdir(parents=True,exist_ok=True)
    with CSV.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    order = [x for x in LABELS if x in objectives]+[x for x in objectives if x not in LABELS]
    fig,axes=plt.subplots(1,3,figsize=(19,7.7),gridspec_kw={'width_ratios':[1.2,1.2,1]})
    def plot_points(ax,rs,ys,color,label=None):
        valid=[(r,y) for r,y in zip(rs,ys) if r['n']]
        for j,(r,y) in enumerate(valid):
            p=r['porcentaje_victoria']
            ax.errorbar(p,y,xerr=[[p-r['ic95_inferior']],[r['ic95_superior']-p]],fmt='o',color=color,capsize=3,label=label if j==0 else None)
    totals=[next(r for r in rows if r['tipo']=='objetivo' and r['objetivo']==o and r['lado']=='total') for o in order]
    y=np.arange(len(order))
    plot_points(axes[0],totals,y,'#285d8f')
    axes[0].set_yticks(y,[f"{LABELS.get(o,o)}\nn = {r['n']:,}".replace(',','.') for o,r in zip(order,totals)])
    axes[0].set_title('Equipo que logró el primero')
    for side,offset,color in [('azul',-.13,'#2874b8'),('rojo',.13,'#bf4747')]:
        rs=[next(r for r in rows if r['tipo']=='objetivo' and r['objetivo']==o and r['lado']==side) for o in order]
        plot_points(axes[1],rs,y+offset,color,side.capitalize())
    axes[1].set_yticks(y,[LABELS.get(o,o) for o in order])
    axes[1].set_title('Según el lado del primer equipo')
    axes[1].legend(loc='lower right')
    prs=[r for r in rows if r['tipo']=='combinacion_patron']
    plot_points(axes[2],prs,np.arange(8),'#548052')
    axes[2].set_yticks(range(8),[f"{r['patron']}   n = {r['n']:,}".replace(',','.') for r in prs])
    axes[2].set_title('Victoria azul por combinación\nSangre, torre y dragón')
    for ax in axes:
        ax.axvline(50,color='0.6',linestyle=':',linewidth=1)
        ax.set_xlim(0,100)
        ax.set_xlabel('Victorias (%) e IC Wilson del 95 %')
        ax.grid(axis='x',alpha=.2)
        ax.invert_yaxis()
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Figura 5. Primeros objetivos y victoria',fontsize=17)
    fig.text(.5,.035, f"Cola 420, duración de al menos 5 minutos. {n:,} partidas. A = azul, R = rojo.\nAsociación descriptiva en historiales de 400 jugadores semilla. Los intervalos no corrigen jugadores compartidos.".replace(',','.'),ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.10,1,.94],w_pad=2)
    FIG.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG,dpi=180)
    plt.close(fig)
    text=['# Primeros objetivos y victoria','',
          'Aplicamos el protocolo C fijado en explore.md y en la sección 7 del plan. Leímos una sola vez cada gzip local, sin llamadas de red ni acceso a timelines. Los resultados son descriptivos y no entrenamos modelos para este apartado.','',
          '## Muestra y ejecución','',
          f'La captura comenzó el {capture}. Encontramos {fmt(len(paths),0)} archivos y conservamos {fmt(n,0)} partidas únicas de cola 420, con duración de al menos 300 segundos, exactamente los equipos 100 y 200 y un único ganador. Usamos {workers} procesos en una máquina de {os.cpu_count()} núcleos lógicos, dejando al menos dos fuera de este cálculo. La lectura y agregación inicial tomó {fmt(elapsed)} segundos.', '',
          '| Estado del archivo | n |','|---|---:|']
    text += [f'| {s} | {v:,} |'.replace(',','.') for s,v in sorted(counts.items())]
    text += ['', 'La deduplicación se hizo por identificador de partida, resumido en memoria con SHA-256. Conservamos la primera aparición en el orden de rutas. Los filtros de exclusión son secuenciales y sus recuentos son excluyentes. Si falta metadata.matchId usamos platformId y gameId. Si ambos faltan, excluimos el archivo. No guardamos identificadores de jugadores ni de partidas en las salidas.', '',
             f'Huella SHA-256 de la concatenación ordenada de las huellas del contenido comprimido: `{manifest.hexdigest()}`. Los archivos ilegibles, de haberlos, aportan la cadena literal ilegible. Esta huella permite comprobar la misma captura sin publicar nombres de archivos.', '']
    if stamps:
        dates=[datetime.fromtimestamp(x/1000,timezone.utc).isoformat() for x in (min(stamps),max(stamps))]
        text += [f'Las fechas de inicio van de {dates[0]} a {dates[1]}. No aplicamos filtros adicionales por parche o fecha.','']
    text += ['## Victoria del equipo que logró primero cada objetivo','',
             'Para cada campo exigimos first booleano en los dos equipos y exactamente un valor verdadero. El denominador de cada tasa son solo esas partidas. Calculamos Wilson bilateral del 95 %, con z = 1,959963984540054. Las tasas por lado se condicionan a que ese lado haya logrado el primer objetivo, no a todas las partidas de ese lado.','',
             '| Objetivo | n total | Victoria total e IC 95 % | n azul | Victoria azul e IC 95 % | n rojo | Victoria roja e IC 95 % |','|---|---:|---|---:|---|---:|---|']
    def rate(r):
        return 'No estimable' if not r['n'] else f"{fmt(r['porcentaje_victoria'])} % ({fmt(r['ic95_inferior'])} a {fmt(r['ic95_superior'])})"
    for o in order:
        rs=[next(r for r in rows if r['tipo']=='objetivo' and r['objetivo']==o and r['lado']==s) for s in ('total','azul','rojo')]
        text += [f"| {LABELS.get(o,o)} | {rs[0]['n']} | {rate(rs[0])} | {rs[1]['n']} | {rate(rs[1])} | {rs[2]['n']} | {rate(rs[2])} |"]
    blue_n=sum(w=='azul' for w,*_ in records)
    lo,hi=wilson(blue_n,n)
    text += ['',f'Como referencia, el lado azul ganó {blue_n} de {n} partidas, {fmt(100*blue_n/n)} % (IC 95 % {fmt(lo)} a {fmt(hi)}). Esta tasa no ajusta las asociaciones de objetivos por diferencias entre equipos.','',
             'El campo champion se interpretó como primera sangre y horde como larvas. Mantuvimos voidGator como nombre del campo, sin equipararlo a otro objetivo. No asumimos que su ausencia indique cero. Atakhan y voidGator se informan por separado. Para voidGator no hubo una sola partida con primer equipo marcado, por lo que no estimamos una tasa de victoria.','',
             '## Cobertura y valores excluidos por objetivo','',
             'Ausente significa que el objetivo falta en al menos un equipo. first ausente significa que está el objetivo pero falta la marca o su objeto no es válido. first inválido significa que la marca no es booleana. Sin primero corresponde a dos marcas falsas y contradictorio a dos verdaderas. Cada fila suma la muestra incluida.','',
             '| Campo | Válidos | Ausente | first ausente | first inválido | Sin primero | Contradictorio |','|---|---:|---:|---:|---:|---:|---:|']
    text += ['| '+o+' | '+' | '.join(str(audit[o][s]) for s in STATES)+' |' for o in order]
    text += ['', '## Combinación de primera sangre, torre y dragón','',
             f'Pudimos evaluar los tres objetivos en {sum(patterns.values())} partidas. En {combo_missing["no_evaluable"]} faltó una marca válida para al menos uno. En {same_n} un mismo equipo obtuvo los tres y ganó {same_w} veces, {rate(next(r for r in rows if r["tipo"]=="combinacion_mismo_equipo" and r["lado"]=="total"))}. En {sum(patterns.values())-same_n} los tres quedaron repartidos.','',
             '| Equipo con los tres | n | Victorias | Tasa e IC 95 % |','|---|---:|---:|---|']
    for r in rows:
        if r['tipo']=='combinacion_mismo_equipo' and r['lado']!='total':
            text += [f"| {r['lado']} | {r['n']} | {r['victorias']} | {rate(r)} |"]
    text += ['', 'La siguiente tabla conserva el orden sangre, torre y dragón. A significa azul y R rojo. En todas sus filas la victoria es la del lado azul. Así se distinguen las ocho combinaciones sin tratar un objetivo repartido como una ausencia.','',
             '| Patrón | n | Victorias azules | Tasa azul e IC 95 % |','|---|---:|---:|---|']
    text += [f"| {r['patron']} | {r['n']} | {r['victorias']} | {rate(r)} |" for r in prs]
    text += ['', 'Las causas de no evaluación de la combinación por objetivo son las siguientes. Una partida puede aparecer en más de una causa, por lo que estos recuentos no se suman.','', '| Objetivo | Causa | n |','|---|---|---:|']
    text += [f'| {k[0]} | {k[1]} | {v} |' for k,v in combo_missing.items() if isinstance(k,tuple)]
    text += ['', '## Alcance de los resultados','',
             'Estas partidas proceden de historiales de 400 jugadores semilla. No son una muestra representativa del servidor y pueden compartir jugadores. Los intervalos Wilson tratan las partidas como observaciones independientes y no corrigen esa dependencia ni la selección de la muestra. Tampoco ajustamos por habilidad, composición, parche, duración o estado de juego. Los intervalos son marginales por fila, sin ajuste por comparaciones múltiples.','',
             'Las tasas muestran asociación. No estiman cuánto cambiaría la probabilidad de ganar si se forzara a obtener un objetivo. Los equipos que ya van ganando pueden ser quienes lo consiguen primero. Barón e inhibidor requieren además que la partida avance lo suficiente y que se produzca el objetivo. Por eso sus denominadores cambian y una tasa alta no establece que la partida se decida en ese momento. Las combinaciones tampoco miden un efecto adicional causal.','',
             '## Productos, trazabilidad y comprobaciones','',
             'Las tasas, coberturas y exclusiones están en data/analysis/objetivos.csv. tipo=objetivo contiene las tasas por objetivo y lado. tipo=cobertura_objetivo conserva todos los estados excluidos. tipo=combinacion_mismo_equipo usa como referencia al equipo con los tres. tipo=combinacion_patron usa siempre victoria azul. Las filas de cobertura y auditoría no tienen tasa de victoria. n_referencia indica la población de cobertura, no sustituye el denominador n de cada tasa. Los decimales del CSV usan punto para lectura con código.','',
             'La figura está en data/analysis/figuras/figura_5_objetivos.png. Muestra todos los objetivos observados, los lados y las ocho combinaciones, con intervalos y tamaños. No hicimos una copia fuera de las rutas autorizadas.','',
             'El script está en scripts/analisis/objetivos.py. Para reproducir desde la raíz ejecutamos PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/analisis/objetivos.py. No usamos aleatoriedad ni bootstrap en este apartado.','',
             f'Entorno: Python {platform.python_version()}, matplotlib {matplotlib.__version__}, NumPy {np.__version__}. Comprobamos mediante aserciones que los estados de archivo suman todos los gzip, que los estados de cada objetivo suman la muestra, que las combinaciones evaluables y no evaluables suman la muestra y que ninguna victoria excede su denominador. También pasaron controles sintéticos en memoria para el límite de 300 segundos, un único ganador, marcas booleanas, ausencias, contradicciones y Wilson frente al caso conocido de 50 victorias de 100.','',
             '| Versión del juego | Partidas incluidas |','|---|---:|']
    text += [f'| {v} | {c} |' for v,c in sorted(versions.items())]
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'archivos':len(paths),'partidas':n,'objetivos':len(objectives),'procesos':workers,'segundos_lectura':round(elapsed,2),'filas_csv':len(rows),'figura_bytes':FIG.stat().st_size,'informe_bytes':REPORT.stat().st_size},ensure_ascii=False))

if __name__=='__main__':
    main()
