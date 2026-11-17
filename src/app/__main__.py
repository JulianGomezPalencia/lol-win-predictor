"""Arranca la página web en la máquina local."""
from src.app.server import create_app

if __name__ == "__main__":
    print("Interfaz disponible en http://127.0.0.1:8000", flush=True)
    create_app().run(host="127.0.0.1", port=8000, debug=False)
