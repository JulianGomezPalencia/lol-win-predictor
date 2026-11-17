"""Aportes en logit. Explican la predicción, no efectos causales."""
import numpy as np
import pandas as pd
from src.features.build import FINAL_FEATURES

LABELS = {
    "dif_winrate_10": "los resultados de las últimas diez partidas",
    "dif_partidas_con_campeon": "la experiencia reciente con los campeones elegidos",
    "dif_racha": "la racha reciente",
    "dif_log_dias_desde_ultima": "el tiempo desde la última partida",
    "dif_kda_medio": "la relación reciente entre bajas, muertes y asistencias",
    "dif_partidas_previas": "la cantidad de partidas recientes disponibles",
    "dif_sin_historial": "la falta de historial",
    "dif_kills_medio": "las bajas conseguidas en partidas recientes",
    "dif_partidas_24h": "las partidas jugadas en las últimas veinticuatro horas",
    "dif_winrate_campeon_posicion": "los resultados del campeón en su posición",
}


def contributions(bundle, features):
    names = bundle["features"]
    if set(names) != set(FINAL_FEATURES):
        raise ValueError("La explicación necesita las diez variables del modelo previo.")
    pipeline = bundle["model"]
    row = pd.DataFrame([{name: features.get(name, np.nan) for name in names}])
    transformed = pipeline[:-1].transform(row)
    model = pipeline.steps[-1][1]
    sign = 1 if model.classes_[1] == 1 else -1
    values = sign * model.coef_[0] * transformed[0]
    result = []
    for name, value in zip(names, values):
        value = float(value)
        phrase = (f"Por {LABELS[name]}, el cálculo favorece al equipo {'azul' if value > 0 else 'rojo'}." if value else
                  f"{LABELS[name].capitalize()} no cambia el cálculo en esta partida.")
        result.append({"variable": name, "aporte": value, "frase": phrase})
    return sorted(result, key=lambda item: abs(item["aporte"]), reverse=True)
