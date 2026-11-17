"""Pantalla delgada. Las operaciones se ejecutan solo al pulsar un botón."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import altair as alt
import pandas as pd
import streamlit as st
from src.app import logic
from src.models import ingame, predict


def percent(value):
    return f"{value * 100:.1f} %".replace(".", ",")


def show_replay(example):
    st.write(example["etiqueta"])
    st.write("Ganó el equipo " + ("azul" if example["gana_azul"] else "rojo") + ".")
    for team, champions in example["campeones"].items():
        st.write(f"Equipo {team}", ", ".join(champions))
    if not example["curva"]:
        st.info("Esta partida no tiene una curva guardada.")
        return
    data = pd.DataFrame(example["curva"]).rename(columns={"minuto": "Minuto", "prob_azul": "Probabilidad azul"})
    line = alt.Chart(data).mark_line().encode(x="Minuto:Q", y=alt.Y("Probabilidad azul:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")), tooltip=["Minuto", alt.Tooltip("Probabilidad azul", format=".1%")])
    if example["eventos"]:
        events = pd.DataFrame(example["eventos"]).rename(columns={"minuto": "Minuto", "tipo": "Objetivo", "equipo": "Equipo"})
        marks = alt.Chart(events).mark_rule(opacity=.4).encode(x="Minuto:Q", color=alt.Color("Equipo:N", scale=alt.Scale(domain=["azul", "rojo"], range=["#2474ce", "#d94040"])), strokeDash="Objetivo:N", tooltip=["Minuto", "Objetivo", "Equipo"])
        st.altair_chart(line + marks, width="stretch")
        st.dataframe(events, hide_index=True)
    else:
        st.altair_chart(line, width="stretch")


def main():
    st.set_page_config(page_title="La partida minuto a minuto", layout="wide")
    st.title("La partida minuto a minuto")
    st.caption("Estimamos la victoria del equipo azul. Las probabilidades cambian mientras se juega.")
    live_tab, replay_tab, calc_tab, learned_tab = st.tabs(["Partida en vivo", "Repetición", "Calculadora", "Qué aprendimos"])
    with live_tab:
        st.write("La consulta de historiales tarda varios minutos. El Riot ID se usa durante la consulta y no se guarda.")
        st.info("Mientras se descargan líneas de tiempo, espera a que termine esa descarga antes de consultar Riot.")
        with st.form("live"):
            riot_id = st.text_input("Riot ID", placeholder="Nombre#Etiqueta")
            submit = st.form_submit_button("Consultar partida")
        if submit:
            progress = st.progress(0, text="Buscando partida")
            steps = [0]
            def update(_):
                steps[0] += 1
                progress.progress(min(steps[0] / 11, .95), text=f"Consultando historial {min(steps[0], 10)} de 10")
            try:
                result = logic.live(riot_id, predict.load_model(), update)
                progress.progress(1.0, text="Consulta terminada")
                st.metric("Probabilidad del azul", percent(result["prob_azul"]))
                st.write(result["confianza"])
                for team, players in result["equipos"].items():
                    st.write(f"Equipo {team}")
                    st.dataframe(pd.DataFrame(players, columns=["Posición estimada", "Campeón"]), hide_index=True)
                st.write("Los tres factores que más pesaron")
                for factor in result["factores"]:
                    st.write(factor["frase"])
                st.caption("Estos aportes explican el cálculo del modelo. No demuestran causas.")
            except predict.LiveError:
                st.info("No encontramos una partida clasificatoria en curso que podamos consultar. Puedes usar las otras pestañas.")
            except FileNotFoundError:
                st.info("Falta el modelo previo. Puedes usar las repeticiones guardadas.")
            except Exception:
                st.info("No pudimos consultar la partida. Comprueba las credenciales y la conexión. Puedes seguir con la demostración.")
    with replay_tab:
        examples = logic.load_examples()
        if examples:
            choice = st.selectbox("Ejemplos guardados", range(len(examples)), format_func=lambda i: examples[i]["etiqueta"])
            show_replay(examples[choice])
        else:
            st.info("Todavía no hay ejemplos guardados. Los añadiremos después de entrenar los modelos por minuto.")
        with st.form("replay"):
            match_id = st.text_input("Identificador de partida")
            submitted = st.form_submit_button("Cargar repetición")
        if submitted:
            try:
                show_replay(logic.replay(match_id, ingame.load_bundle()))
            except FileNotFoundError:
                st.info("Faltan los modelos por minuto. Las repeticiones guardadas funcionan sin ellos.")
            except Exception:
                st.info("No pudimos cargar esta repetición. Revisa el identificador, las credenciales y los modelos disponibles.")
    with calc_tab:
        st.write("Diferencias del equipo azul menos el rojo. Dejamos las demás variables en cero. Es una simulación simplificada.")
        minute = st.slider("Minuto", 1, 40, 15)
        gold = st.slider("Diferencia de oro", -20000, 20000, 0, 500)
        kills = st.slider("Diferencia de bajas", -40, 40, 0)
        towers = st.slider("Diferencia de torres", -11, 11, 0)
        dragons = st.slider("Diferencia de dragones", -7, 7, 0)
        try:
            p = logic.calculator(ingame.load_bundle(), minute, gold, kills, towers, dragons)
            st.metric("Probabilidad del azul", percent(p))
            if minute > 30:
                st.caption("Después del minuto 30 usamos el modelo del minuto 30.")
        except FileNotFoundError:
            st.info("Todavía no hay modelos por minuto para usar la calculadora.")
        except Exception:
            st.info("No hay un modelo disponible para este minuto.")
        with st.expander("Calculadora antes de la partida"):
            try:
                from src.models.explain import LABELS
                prior = logic.load_prior_model()
                values = {name: st.number_input(LABELS[name].capitalize(), value=0., key=name)
                          for name in prior["features"]}
                result = logic.prior_calculator(values, prior)
                st.metric("Probabilidad previa del azul", percent(result["prob_azul"]))
                for factor in result["factores"][:3]:
                    st.write(factor["frase"])
            except FileNotFoundError:
                st.info("No está disponible el modelo previo para esta calculadora.")
    with learned_tab:
        paths = logic.findings_paths()
        if not any(paths.values()):
            st.info("Los resultados y las figuras aparecerán aquí cuando terminemos el análisis.")
        for path in paths["tablas"]:
            st.write(path.stem.replace("_", " ").capitalize())
            try:
                st.dataframe(logic.read_findings_table(path), hide_index=True)
            except Exception:
                st.info("No pudimos leer esta tabla.")
        for path in paths["figuras"]:
            st.image(str(path))


if __name__ == "__main__":
    main()
