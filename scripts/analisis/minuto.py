#!/usr/bin/env python3
"""Protocolo A/B local. Debug solo entrenamiento. Sin red ni export de jugadores."""
import os
os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__file__).resolve().parents[2]/'data/analysis/tmp/mplconfig'))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import sys, json, sqlite3, argparse, time, subprocess, hashlib
from datetime import datetime, timezone
from collections import Counter
import numpy as np
import pandas as pd
import joblib, sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.features import build
from src.features.timeline import STATE_FEATURES, state_at, participant_teams
from src.models import ingame, predict, train
from src.riot_client import cache
from scripts.export_dataset import QUERIES, anonymize
from scripts import build_demo_examples
RUN=ROOT/'data'
OUT=RUN/'analysis'
CUT=int(datetime.fromisoformat('2026-11-09T18:09:41+00:00').timestamp()*1000)
KEY=[5,10,15,20,25]

def savejson(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def snapshot():
    source=sqlite3.connect((ROOT/'data/db/lol.sqlite').as_uri()+'?mode=ro',uri=True)
    conn=sqlite3.connect(':memory:')
    source.backup(conn); source.close()
    salt=(ROOT/'data/export_salt.txt').read_text().strip()
    tables={}
    for name in ['matches','participants','player_history']:
        t=pd.read_sql_query(QUERIES[name],conn)
        if 'player_id' in t:
            t['player_id']=t.player_id.map(lambda p:anonymize(p,salt))
        tables[name]=t
    conn.close()
    spells=[]
    for mid in sorted(tables['participants'].match_id.unique()):
        match=cache.load_match(mid)
        if match:
            for p in match['info']['participants']:
                spells.append(dict(match_id=mid,player_id=anonymize(p['puuid'],salt),spell1_id=p['summoner1Id'],spell2_id=p['summoner2Id']))
    tables['participant_spells']=pd.DataFrame(spells)
    assert not tables['participants'].duplicated(['match_id','player_id']).any()
    assert not tables['participant_spells'].duplicated(['match_id','player_id']).any()
    return tables

def subset(t,ids):
    return {k:v[v['target_match_id' if k=='player_history' else 'match_id'].isin(ids)].copy() for k,v in t.items()}

def table_for(t,pool,shares):
    m,p,h,s=[t[k] for k in ['matches','participants','player_history','participant_spells']]
    p=build.inferred_participants(p,h,m,s,shares)
    return build.match_features(build.player_features(h,p,m,pool),m,True)

def prior_oof(t,ids,features):
    sub=subset(t,set(ids))
    m=sub['matches'].set_index('match_id').loc[ids].reset_index()
    probs=pd.Series(index=ids,dtype=float)
    audit=[]
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(m,m.blue_win)):
        fit_ids=set(m.iloc[a].match_id); hold_ids=set(m.iloc[b].match_id)
        pool,shares=predict._pool_and_shares(sub['player_history'],fit_ids)
        tab=table_for(sub,pool,shares).set_index('match_id')
        fit=tab.loc[list(fit_ids)].sort_values('game_start_ts',kind='stable')
        model=train.logistic_model()
        model.fit(fit[features],fit.blue_win.astype(int),model__sample_weight=train.tier_weights(fit.seed_tier))
        probs.loc[list(hold_ids)]=model.predict_proba(tab.loc[list(hold_ids),features])[:,1]
        audit.append(dict(pliegue=fold+1,n_ajuste=len(fit_ids),n_retenido=len(hold_ids),C=model.best_params_['model__C']))
    assert probs.notna().all()
    return probs,audit

def extract(t, debug=False):
    m=t['matches'].set_index('match_id'); parts=t['participants']
    rows=[]; counts=Counter(); valid=[]
    paths=sorted((ROOT/'data/raw/timelines').glob('*.json.gz'))
    for path in paths:
        mid=path.name[:-8]
        if debug and (mid not in m.index or not m.loc[mid,'game_start_ts']<CUT):continue
        counts['archivos']+=1
        if mid not in m.index:counts['sin_catalogo']+=1;continue
        r=m.loc[mid]
        if r.status!='history_done' or pd.isna(r.game_start_ts) or r.blue_win not in (0,1):counts['metadatos_invalidos']+=1;continue
        if r.game_start_ts==CUT:counts['empate_corte']+=1;continue
        tl,match=cache.load_timeline(mid),cache.load_match(mid)
        if tl is None or match is None:counts['ilegible_o_sin_partida']+=1;continue
        try:
            info=match['info']; teams=info['teams']; ps=info['participants']
            assert match['metadata']['matchId']==mid==tl['metadata']['matchId']
            assert len(teams)==2 and {x['teamId'] for x in teams}=={100,200} and sum(bool(x['win']) for x in teams)==1
            assert int(next(x['win'] for x in teams if x['teamId']==100))==int(r.blue_win)
            assert info['gameStartTimestamp']==r.game_start_ts
            dbp=parts[parts.match_id==mid]
            assert len(dbp)==10 and set(dbp.team_id)=={100,200}
            assert all(int(x.win)==int(r.blue_win if x.team_id==100 else 1-r.blue_win) for x in dbp.itertuples())
            mapped={p['participantId']:p['teamId'] for p in ps}
            assert participant_teams(tl)==mapped
            ts=[f['timestamp'] for f in tl['info']['frames']]
            assert ts==sorted(ts) and len(ts)==len(set(ts))
            local=[]
            for minute in range(1,31):
                s=state_at(tl,minute)
                if s is None:break
                assert len(tl['info']['frames'][minute]['participantFrames'])==10
                assert np.isfinite(list(s.values())).all()
                local.append(dict(match_id=mid,minute=minute,blue_win=int(r.blue_win),game_start_ts=r.game_start_ts,**s))
            rows.extend(local);valid.append(mid)
        except (ValueError,KeyError,AssertionError,TypeError):counts['inconsistentes']+=1
    counts['validas']=len(valid)
    return pd.DataFrame(rows),dict(counts),valid

def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))

def losses(y,p):
    p=np.clip(p,1e-6,1-1e-6);return -(y*np.log(p)+(1-y)*np.log(1-p))

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);n=len(y)
    idx=np.random.default_rng(42).integers(n,size=(2000,n))
    hits=((p>=.5)==y).astype(float);loss=losses(y,p)
    au=[roc_auc_score(y[i],p[i]) for i in idx if len(np.unique(y[i]))==2]
    d=dict(n=n,acierto=float(hits.mean()),perdida_logaritmica=float(loss.mean()),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,bootstrap_auc_validos=len(au))
    for name,v in [('acierto',hits[idx].mean(1)),('perdida_logaritmica',loss[idx].mean(1)),('auc',au)]:
        ci=np.percentile(v,[2.5,97.5]) if len(v) else [None,None]
        d[name+'_ic_bajo'],d[name+'_ic_alto']=ci
    return d

def wilson(k,n):
    if not n:return None,None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h

def wait_download():
    start=time.monotonic();log=[]
    while True:
        r=subprocess.run(['pgrep','-f','[d]ownload_timelines'],capture_output=True,text=True)
        running=r.returncode==0
        log.append(dict(utc=datetime.now(timezone.utc).isoformat(),activo=running))
        savejson(OUT/'espera_descarga.json',log)
        if not running:return False
        if time.monotonic()-start>=2700:return True
        print('Descarga activa. Comprobación local cada 60 s.',flush=True)
        time.sleep(60)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--debug',action='store_true');parser.add_argument('--final',action='store_true');a=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.final:
        if (OUT/'evaluacion_iniciada.json').exists():raise RuntimeError('Evaluación ya iniciada. No repetir prueba.')
        timeout=wait_download()
    elif not a.debug:raise ValueError('Elegir --debug o --final')
    t=snapshot()
    frozen=joblib.load(ROOT/'data/models/modelo_final_2026-11-09.joblib')
    rows,counts,valid=extract(t,a.debug)
    tr=rows[rows.game_start_ts<CUT].copy()
    bundle=ingame.train(tr.to_dict('records'))
    ids=sorted(tr.match_id.unique())
    if a.debug:ids=ids[:min(100,len(ids))]
    oof,folds=prior_oof(t,ids,frozen['features'])
    if a.debug:
        savejson(OUT/'debug_entrenamiento.json',dict(conteos=counts,n_partidas=len(ids),n_filas=len(tr),modelos=len(bundle['models']),pliegues=folds,prueba_consultada=False))
        print('Debug completado solo con entrenamiento.',flush=True);return
    final_analysis(t,rows,tr,bundle,oof,folds,frozen,counts,valid,timeout)

def final_analysis(t,rows,tr,bundle,oof,folds,frozen,counts,valid,timeout):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=predict._table_for(ROOT/'data/export',frozen['pool'],frozen['champion_positions'])
    def membership(tab):
        ids=tab.match_id.astype(str)
        return tab[~ids.isin(set(map(str,frozen['match_ids']))) & ~ids.isin(set(map(str,frozen['development_ids']))) & (tab.game_start_ts>max(frozen['last_match_ts'],predict.SECOND_CUTOFF_TS))]
    original_test=membership(original)
    published=json.loads((ROOT/'data/models/segunda_prueba.json').read_text())['runs'][-1]
    assert len(original_test)==published['n']==1264
    full=table_for(t,frozen['pool'],frozen['champion_positions'])
    assert not full.match_id.duplicated().any()
    common=sorted(set(original.match_id)&set(full.match_id))
    left=original.set_index('match_id').loc[common,frozen['features']]
    right=full.set_index('match_id').loc[common,frozen['features']]
    np.testing.assert_allclose(left.to_numpy(float),right.to_numpy(float),equal_nan=True,rtol=1e-10,atol=1e-10)
    confirm_ids=set(original_test.match_id)
    test=rows[(rows.game_start_ts>CUT)&rows.match_id.isin(confirm_ids)].copy()
    outside=rows[(rows.game_start_ts>CUT)&~rows.match_id.isin(confirm_ids)]
    post=rows[rows.game_start_ts>CUT]
    ntrain=tr.match_id.nunique()
    audit=dict(captura_utc=datetime.now(timezone.utc).isoformat(),corte_ms=CUT,semilla=42,bootstrap=2000,python=sys.version.split()[0],sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,conteos=counts,timeout_descarga=timeout,entrenamiento=ntrain,segunda_prueba_original=len(confirm_ids),segunda_prueba_con_timeline=len(confirm_ids&set(valid)),segunda_prueba_ausente=len(confirm_ids-set(valid)),segunda_prueba_y_despues_corte=test.match_id.nunique(),posteriores=post.match_id.nunique(),posteriores_fuera_confirmatoria=outside.match_id.nunique(),filas_previas_reproducidas=len(common),pliegues=folds)
    warnings=[]
    if ntrain<800:warnings.append('No se alcanzó el mínimo de 800 partidas anteriores. Resultados parciales, sin cumplimiento de la compuerta.')
    if audit['segunda_prueba_ausente']:warnings.append('No están todas las timelines de la segunda prueba original. La evaluación usa únicamente su intersección con fechas posteriores al corte y cuadros disponibles.')
    if timeout:warnings.append('Se agotaron 45 minutos de espera. Usamos solo los archivos existentes al iniciar la captura.')
    assert set(tr.match_id).isdisjoint(set(test.match_id))
    assert set(bundle['match_ids'])==set(tr.match_id)
    assert full.set_index('match_id').loc[list(test.match_id.unique()),frozen['features']].shape[1]==10
    if test.empty:raise ValueError('No hay prueba confirmatoria con timeline y posterior al corte')
    audit['avisos']=warnings
    savejson(OUT/'auditoria.json',audit)
    savejson(OUT/'evaluacion_iniciada.json',dict(utc=datetime.now(timezone.utc).isoformat(),sha256_script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocolo_sha256=hashlib.sha256((ROOT/'scripts/analisis/protocolo.md').read_bytes()).hexdigest()))
    modelpath=ROOT/'data/models/modelos_en_partida.joblib'
    modelpath.parent.mkdir(parents=True,exist_ok=True)
    bundle['cutoff']=datetime.fromtimestamp(CUT/1000,timezone.utc).isoformat()
    joblib.dump(bundle,modelpath)
    full=full.set_index('match_id')
    testids=sorted(test.match_id.unique())
    pre=pd.Series(frozen['model'].predict_proba(full.loc[testids,frozen['features']])[:,1],index=testids)
    metric_rows=[];comparisons=[];preds={}
    baseline=published['modelos']['final']
    metric_rows.append(dict(minuto=0,modelo='previo_publicado',n=published['n'],acierto=baseline['accuracy'],acierto_ic_bajo=baseline['accuracy_ic95'][0],acierto_ic_alto=baseline['accuracy_ic95'][1],auc=baseline['auc'],auc_ic_bajo=baseline['auc_ic95'][0],auc_ic_alto=baseline['auc_ic95'][1],perdida_logaritmica=baseline['log_loss'],fuente='data/models/segunda_prueba.json'))
    for minute in range(1,31):
        fit=tr[tr.minute==minute];g=test[test.minute==minute].sort_values('match_id');y=g.blue_win.to_numpy(int)
        if g.empty or minute not in bundle['models']:continue
        model=bundle['models'][minute];p=model.predict_proba(g[STATE_FEATURES])[:,1]
        preds[minute]=(g,p)
        metric_rows.append(dict(minuto=minute,modelo='estado',**metrics(y,p),fuente='prueba_interseccion'))
        if minute not in KEY:continue
        gold=g.dif_oro.to_numpy();goldhit=((gold>=0)==y).astype(float)
        idx=np.random.default_rng(42).integers(len(g),size=(2000,len(g)))
        lo,hi=np.percentile(goldhit[idx].mean(1),[2.5,97.5])
        metric_rows.append(dict(minuto=minute,modelo='regla_oro',n=len(g),acierto=float(goldhit.mean()),acierto_ic_bajo=lo,acierto_ic_alto=hi,empates_oro=int((gold==0).sum()),fuente='prueba_interseccion'))
        hist=HistGradientBoostingClassifier(random_state=42).fit(fit[STATE_FEATURES],fit.blue_win.astype(int))
        hp=hist.predict_proba(g[STATE_FEATURES])[:,1]
        metric_rows.append(dict(minuto=minute,modelo='hist_gradient_boosting',**metrics(y,hp),fuente='prueba_interseccion'))
        xfit=fit[STATE_FEATURES].copy();xfit['logit_previo']=logit(oof.loc[fit.match_id].to_numpy())
        xtest=g[STATE_FEATURES].copy();xtest['logit_previo']=logit(pre.loc[g.match_id].to_numpy())
        augmented=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42)).fit(xfit,fit.blue_win.astype(int))
        ap=augmented.predict_proba(xtest)[:,1]
        metric_rows.append(dict(minuto=minute,modelo='estado_mas_previo_oof',**metrics(y,ap),fuente='prueba_interseccion'))
        d=train.paired_log_loss_difference(y,p,ap)
        comparisons.append(dict(minuto=minute,n=len(g),perdida_estado=float(losses(y,p).mean()),perdida_estado_previo=float(losses(y,ap).mean()),diferencia=d['diferencia'],ic95_bajo=d['ic95'][0],ic95_alto=d['ic95'][1]))
        if minute==15 and roc_auc_score(y,p)<.75:
            # extract ya contrastó las diez asignaciones con la partida completa y DB.
            audit['revision_auc_15']='AUC menor que 0,75. Asignaciones de diez participantes, ganador y timestamp comprobados para cada partida en extract, sin discrepancias aceptadas.'
    metrics_df=pd.DataFrame(metric_rows);comp=pd.DataFrame(comparisons)
    analysis=ROOT/'data/analysis';figs=analysis/'figuras';figs.mkdir(parents=True,exist_ok=True)
    metrics_df.to_csv(analysis/'predictibilidad.csv',index=False);comp.to_csv(analysis/'comparacion_previo.csv',index=False)
    # Calibración diagnóstica sin modificar el modelo.
    g,p=preds[15];y=g.blue_win.to_numpy(int);z=logit(p)
    fitcal=minimize(lambda b:np.mean(np.logaddexp(0,b[0]+b[1]*z)-y*(b[0]+b[1]*z)),[0,1],method='BFGS')
    intercept,slope=(map(float,fitcal.x) if np.isfinite(fitcal.x).all() and len(np.unique(y))==2 else (None,None))
    bins=np.minimum((p*10).astype(int),9);cal=[]
    for i in range(10):
        mask=bins==i;n=int(mask.sum())
        cal.append(dict(intervalo_inferior=i/10,intervalo_superior=(i+1)/10,n=n,probabilidad_media=float(p[mask].mean()) if n else None,fraccion_observada=float(y[mask].mean()) if n else None,brier=float(np.mean((y-p)**2)),intercepto=intercept,pendiente=slope,convergencia=bool(fitcal.success)))
    pd.DataFrame(cal).to_csv(analysis/'calibracion_15.csv',index=False)
    # B usa todas las partidas descargadas válidas, incluidas las anteriores.
    comeback=[];bsummary={}
    for minute in [10,15]:
        allg=rows[rows.minute==minute];g=allg[allg.dif_oro!=0];gold=g.dif_oro.to_numpy();y=g.blue_win.to_numpy(int)
        wins=((gold>0)==y);edges=[0,1000,2000,3000,5000,np.inf]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(np.abs(gold)>=lo)&(np.abs(gold)<hi);n=int(mask.sum());k=int(wins[mask].sum());a,b=wilson(k,n)
            comeback.append(dict(minuto=minute,ventaja_desde=lo,ventaja_hasta=None if np.isinf(hi) else hi,n=n,victorias_lider=k,porcentaje=100*k/n if n else None,wilson95_bajo=100*a if a is not None else None,wilson95_alto=100*b if b is not None else None))
        n=len(g);k=int((~wins).sum());a,b=wilson(k,n)
        bsummary[str(minute)]=dict(con_cuadro=len(allg),sin_cuadro=len(valid)-len(allg),empates=int((allg.dif_oro==0).sum()),denominador=n,remontadas=k,porcentaje=100*k/n if n else None,wilson95=[100*a,100*b] if n else None)
    rem=pd.DataFrame(comeback);rem.to_csv(analysis/'remontadas.csv',index=False)
    savejson(OUT/'remontadas_resumen.json',bsummary)
    # Selección por confianza previa, coherente con el modelo publicado de minuto cero.
    candidates=[]
    g,_=preds[15]
    for r in g.itertuples():
        p0=float(pre.loc[r.match_id]);conf=max(p0,1-p0)>=.60;hit=(p0>=.5)==r.blue_win
        cats=['azar']
        if conf:cats.append('acierto_con_confianza' if hit else 'fallo_con_confianza')
        if r.dif_oro!=0 and ((r.dif_oro>0)!=bool(r.blue_win)):cats.append('remontada')
        for cat in cats:candidates.append(dict(match_id=r.match_id,categoria=cat,prob_previa=p0))
    audit['ejemplos_confianza']='probabilidad previa del modelo congelado, umbral 0,60'
    audit['candidatos_ejemplos']={cat:sum(r['categoria']==cat for r in candidates) for cat in build_demo_examples.QUOTAS}
    try:
        examples=build_demo_examples.build(candidates,bundle,42)
        # Lista blanca exacta del contrato y campeón local. No se serializan los candidatos.
        assert len(examples)==12
        assert all(set(e)=={'etiqueta','match_id_interno','gana_azul','prob_previa','campeones','curva','eventos'} for e in examples)
        savejson(ROOT/'data/app/ejemplos.json',examples)
        audit['ejemplos_generados']=len(examples)
    except ValueError as e:
        audit['ejemplos_error']=str(e);warnings.append('No se pudieron completar las cuotas de ejemplos. '+str(e))
    # Figuras en español. El punto cero se distingue porque cambia la población.
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    state=metrics_df[metrics_df.modelo=='estado']
    for axis,field,label in [(ax[0],'acierto','Proporción de aciertos'),(ax[1],'auc','ROC AUC')]:
        axis.plot(state.minuto,state[field],label='Estado de las partidas con cuadro',color='#146B8C')
        axis.fill_between(state.minuto,state[field+'_ic_bajo'],state[field+'_ic_alto'],alpha=.17,color='#146B8C')
        axis.errorbar([0],[metric_rows[0][field]],yerr=[[metric_rows[0][field]-metric_rows[0][field+'_ic_bajo']],[metric_rows[0][field+'_ic_alto']-metric_rows[0][field]]],fmt='o',color='#B35B32',label='Previo en la segunda prueba completa')
        axis.set(xlabel='Minuto',ylabel=label,ylim=(.45,1.01));axis.grid(alpha=.2)
    gold=metrics_df[metrics_df.modelo=='regla_oro'];ax[0].plot(gold.minuto,gold.acierto,'s:',color='#757575',label='Regla del oro')
    ax[0].legend(fontsize=8,loc='lower right');fig.suptitle('Predictibilidad según el minuto')
    fig.text(.5,.01,'Fuente: data/analysis/predictibilidad.csv. IC del 95 %. La población cambia con la duración.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(figs/'figura_3_predictibilidad.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));labels=['0 a 999','1.000 a 1.999','2.000 a 2.999','3.000 a 4.999','5.000 o más']
    for minute,color,offset in [(10,'#146B8C',-.08),(15,'#B35B32',.08)]:
        d=rem[rem.minuto==minute];x=np.arange(5)+offset
        ax.errorbar(x,d.porcentaje,yerr=[d.porcentaje-d.wilson95_bajo,d.wilson95_alto-d.porcentaje],fmt='o-',color=color,label=f'Minuto {minute}',capsize=3)
        for xx,r in zip(x,d.itertuples()):ax.annotate(f'n={r.n}',(xx,r.porcentaje),xytext=(0,9 if minute==15 else -17),textcoords='offset points',ha='center',fontsize=8)
    ax.set(xticks=np.arange(5),xticklabels=labels,xlabel='Ventaja del equipo que lidera en oro',ylabel='Victorias del equipo con ventaja (%)',title='Ventaja en oro y resultado de la partida',ylim=(35,104));ax.legend();ax.grid(alpha=.2)
    fig.text(.5,.01,'Fuente: data/analysis/remontadas.csv. Wilson del 95 %. Empates excluidos. Asociación descriptiva.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(figs/'figura_4_remontadas.png',dpi=180);plt.close(fig)
    audit['avisos']=warnings;savejson(OUT/'auditoria.json',audit)
    write_report(audit,metrics_df,comp,cal,bsummary)
    print('Análisis final guardado. Consultar minuto.md y auditoria.json.',flush=True)

def write_report(audit,df,comp,cal,bsummary):
    def fmt(v,d=4):return f'{v:.{d}f}'.replace('.',',') if pd.notna(v) else 'no calculable'
    lines=['# Predictibilidad por minuto y remontadas','',
        'Ejecutamos el protocolo fijado en `scripts/analisis/protocolo.md` con el script `scripts/analisis/minuto.py`. No hicimos llamadas a Riot. Depuramos solo con entrenamiento antes de iniciar una única evaluación final. El modelo guardado para la aplicación usa exclusivamente partidas anteriores al corte.','',
        '## Cobertura y controles','',
        'Cada cifra de esta sección procede de `data/analysis/auditoria.json`.']
    for k,v in audit.items():
        if k not in ['avisos','pliegues']:lines.append(f'- `{k}`: `{json.dumps(v,ensure_ascii=False)}`.')
    lines+=['','Avisos: '+(' '.join(audit['avisos']) or 'Sin incumplimientos de cobertura detectados.'),'',
        'La pertenencia original se reconstruyó excluyendo ajuste y desarrollo y aplicando el filtro de fecha de `evaluate_final`. El punto cero publicado usa la segunda prueba completa. Los puntos siguientes usan su intersección con timelines posteriores al corte y supervivencia a cada cuadro. No son la misma población. Las partidas posteriores ajenas a la segunda prueba solo entraron en el análisis descriptivo B. Las diez variables compartidas se contrastaron contra `_table_for` con tolerancia numérica, incluidos los faltantes. Las tablas SQLite se leyeron desde una copia consistente en memoria.','',
        '## Protocolo A','',
        'Fuente de cada celda de la tabla: `data/analysis/predictibilidad.csv`. Punto cero tomado sin recalcular de `data/models/segunda_prueba.json`. IC percentiles del 95 %, bootstrap de partidas con semilla fijada. La columna de remuestras válidas de AUC consta en el CSV. La pérdida usa probabilidades recortadas según protocolo.','',
        '| Minuto | Modelo | n | Acierto (IC) | AUC (IC) | Pérdida (IC) |','|---|---|---|---|---|---|']
    for r in df[(df.minuto.isin([0]+KEY))].to_dict('records'):
        def cell(k):
            lo,hi=r.get(k+'_ic_bajo',np.nan),r.get(k+'_ic_alto',np.nan)
            return fmt(r.get(k,np.nan))+' ('+fmt(lo)+' a '+fmt(hi)+')'
        lines.append(f"| {r['minuto']} | {r['modelo']} | {r['n']} | {cell('acierto')} | {cell('auc')} | {cell('perdida_logaritmica')} |")
    lines+=['','La regla de oro resuelve empates a favor de azul y solo tiene acierto, no AUC probabilística ni pérdida logarítmica. Sus empates están en `data/analysis/predictibilidad.csv`. La comparación con HistGradientBoosting es secundaria y descriptiva. No seleccionamos modelos con la prueba.','',
        'Para el previo de entrenamiento generamos predicciones retenidas en pliegues estratificados. Cada pool y tabla de posiciones se construyó solo con el ajuste del pliegue. Cada ajuste ordenó sus partidas por fecha y usó la búsqueda temporal interna, imputación, escala y pesos de liga de `train.logistic_model`. Los parámetros elegidos y tamaños constan en `analysis/auditoria.json`, campo `pliegues`. El previo de prueba es el bundle congelado, sin reajuste.','',
        'Fuente de cada diferencia pareada: `data/analysis/comparacion_previo.csv`. Positiva significa que ayuda añadir el previo.','']
    for r in comp.itertuples():lines.append(f'- Minuto {r.minuto}, n={r.n}: pérdida estado menos estado con previo {fmt(r.diferencia)}, IC 95 % {fmt(r.ic95_bajo)} a {fmt(r.ic95_alto)} (`data/analysis/comparacion_previo.csv`).')
    c=cal[0]
    lines+=['','La calibración es diagnóstica, no aplicada al modelo. Fuente `data/analysis/calibracion_15.csv`: Brier '+fmt(c['brier'])+', intercepto '+fmt(c['intercepto'])+' y pendiente '+fmt(c['pendiente'])+'. Los intervalos de anchura fijada incluyen n, probabilidad media y fracción observada. El último incluye el extremo superior.','',
        '## Protocolo B','',
        'Los tramos, victorias del equipo con ventaja, porcentajes e IC Wilson figuran en `data/analysis/remontadas.csv`. No mezclamos este análisis descriptivo de todas las partidas válidas descargadas con la evaluación confirmatoria.','']
    for minute,v in bsummary.items():lines.append(f"- Minuto {minute}: {json.dumps(v,ensure_ascii=False)}. Fuente `data/analysis/remontadas_resumen.json`.")
    lines+=['','Los ejemplos se seleccionaron con el procedimiento de `scripts/build_demo_examples.py`, orden y cuotas congelados, sin escoger a mano. Interpretamos la confianza como la probabilidad del previo, coherente con los aciertos y fallos de la predicción antes de empezar. Los candidatos cumplen la prueba confirmatoria y disponen del cuadro requerido. Disponibilidad por categoría y resultado constan en `analysis/auditoria.json`. `data/app/ejemplos.json` solo contiene el contrato público y un identificador interno de partida.','',
        '## Figuras y límites','',
        '- Figura 3: `data/analysis/figuras/figura_3_predictibilidad.png`, generada desde `data/analysis/predictibilidad.csv`.','- Figura 4: `data/analysis/figuras/figura_4_remontadas.png`, generada desde `data/analysis/remontadas.csv`.','',
        'Los cuadros disponibles excluyen partidas ya terminadas y cambian la población en cada minuto. El índice de cuadro y su timestamp real definen el estado y los eventos según el código del proyecto. Verificamos los equipos de cada participante, el ganador y la fecha contra la partida completa y la base antes de aceptar una fila. La muestra procede de historiales de jugadores semilla y puede compartir jugadores, no representa al servidor. Los intervalos por partida no corrigen esa dependencia. Oro, objetivos y remontadas son asociaciones, no efectos causales. El análisis no identifica un minuto universal en que se decida la partida.','',
        '## Reproducción','',
        'Desde la raíz, `.venv/bin/python scripts/analisis/minuto.py --debug` comprueba entrenamiento. `--final` espera al proceso existente y ejecuta una sola evaluación. El archivo `evaluacion_iniciada.json` impide repetirla accidentalmente. La espera registrada está en `analysis/espera_descarga.json`. Las versiones, captura, exclusiones y semilla están en `analysis/auditoria.json`. El script escribe únicamente en data/analysis/, data/app/ y data/models/.']
    text='\n'.join(lines)+'\n'
    (OUT/'minuto.md').write_text(text)
    (OUT/'results_minuto.md').write_text(text)

if __name__=='__main__':main()
