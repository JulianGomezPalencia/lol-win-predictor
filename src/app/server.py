"""Interfaz local con contratos públicos pequeños y sin persistencia de consultas."""
import math
import os
from functools import lru_cache

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from src.app import logic
from src.models.explain import LABELS


def create_app():
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 8192
    load_dotenv()

    @lru_cache(maxsize=1)
    def prior():
        return logic.load_prior_model()

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/examples")
    def examples():
        # The final outcome and prior prediction are separate reveal operations.
        return jsonify([{k: v for k, v in e.items() if k not in ("gana_azul", "prob_previa", "curva", "eventos")}
                        for e in logic.load_examples()])

    def selected(index):
        rows = logic.load_examples()
        if not 0 <= index < len(rows):
            return None
        return rows[index]

    @app.get("/api/examples/<int:index>/<action>")
    def example(index, action):
        row = selected(index)
        if row is None:
            return jsonify(mensaje="No hay una partida con ese número."), 404
        if action == "prediction":
            return jsonify(prob_azul=row["prob_previa"])
        if action == "winner":
            return jsonify(gana_azul=row["gana_azul"])
        if action == "replay":
            return jsonify(curva=row["curva"], eventos=row["eventos"])
        return jsonify(mensaje="Vista no disponible."), 404

    @app.get("/api/comeback")
    def comeback():
        for i, row in enumerate(logic.load_examples()):
            point = next((p for p in row["curva"] if p["minuto"] == 15), None)
            if point and (point["prob_azul"] >= .5) != row["gana_azul"]:
                return jsonify(indice=i)
        return jsonify(mensaje="No hay una remontada disponible."), 404

    @app.get("/api/findings")
    def findings():
        tables = {}
        for name in ("predictibilidad", "objetivos"):
            path = logic.asset_path(f"analysis/{name}.csv", f"analysis/{name}.csv")
            if not path.is_file():
                tables[name] = []
                continue
            table = logic.read_findings_table(path)
            if name == "predictibilidad":
                table = table[table["modelo"].isin(["estado", "previo_misma_poblacion", "previo_publicado"])]
                cols = ["minuto", "modelo", "n", "acierto"]
            else:
                table = table[(table["lado"] == "total") & table["tipo"].isin(["objetivo", "combinacion_mismo_equipo"])]
                cols = ["tipo", "objetivo", "n", "porcentaje_victoria"]
            # Only known category strings leave the analysis boundary.
            allowed = {"estado", "previo_misma_poblacion", "previo_publicado", "objetivo",
                       "combinacion_mismo_equipo", "champion", "dragon", "tower", "baron",
                       "inhibitor", "sangre_torre_dragon"}
            rows = table[cols].to_dict("records")
            tables[name] = [r for r in rows if all(not isinstance(v, str) or v in allowed for v in r.values())]
        return jsonify(tables)

    @app.get("/api/calculator")
    def calculator_fields():
        return jsonify(variables=[{"nombre": n, "etiqueta": LABELS[n], "valor": 0}
                                  for n in prior()["features"]])

    @app.post("/api/calculator")
    def calculate():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != set(prior()["features"]):
            return jsonify(mensaje="Completa las diez diferencias entre azul y rojo."), 400
        try:
            if any(isinstance(v, bool) for v in payload.values()):
                raise ValueError()
            values = {k: float(v) for k, v in payload.items()}
            if not all(math.isfinite(v) and abs(v) <= 10000 for v in values.values()):
                raise ValueError()
        except (ValueError, TypeError):
            return jsonify(mensaje="Usa números finitos entre menos diez mil y diez mil."), 400
        return jsonify(logic.prior_calculator(values, prior()))

    @app.route("/api/live", methods=["GET", "POST"])
    def live():
        if not os.getenv("RIOT_API_KEY"):
            return jsonify(disponible=False, mensaje="La consulta en vivo necesita una clave de Riot en .env. La demo funciona sin ella.")
        if not all(k in prior() for k in ("pool", "champion_positions")):
            return jsonify(disponible=False, mensaje="La consulta en vivo necesita el modelo local completo. La demo incluye la calculadora y los ejemplos guardados.")
        if request.method == "GET":
            return jsonify(disponible=True, mensaje="Consulta una partida clasificatoria activa. Puede tardar varios minutos.")
        payload = request.get_json(silent=True)
        value = payload.get("jugador") if isinstance(payload, dict) else None
        if not isinstance(value, str) or not 3 <= len(value) <= 80 or "#" not in value:
            return jsonify(mensaje="Escribe el nombre y la etiqueta separados por #."), 400
        try:
            return jsonify(logic.live(value, prior()))
        except Exception:
            # Never echo service errors, which can include private identifiers.
            return jsonify(mensaje="No se pudo consultar la partida. Comprueba la clave y que haya una partida activa."), 503

    @app.errorhandler(FileNotFoundError)
    def missing(_error):
        return jsonify(mensaje="Faltan archivos de la demo. Restaura la carpeta demo y vuelve a intentar."), 503

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        return response

    return app
