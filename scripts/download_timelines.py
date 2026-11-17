"""Descarga reanudable de líneas de tiempo completas.

Ejecutar solo, sin recolección, --watch, --live ni --score-log simultáneos:
el limitador de Riot se comparte únicamente dentro de un proceso.
Uso: .venv/bin/python scripts/download_timelines.py --max 2800 --minutes 75
--max limita las descargas nuevas exitosas, sin contar caché ni errores.
El plazo incluye consultas, lecturas, llamadas y esperas del cliente. Las operaciones
locales y una respuesta HTTP en curso pueden terminar después del plazo.
"""
import argparse
import math
from pathlib import Path
import sqlite3
import sys
import time

import requests

# Permite ejecutar el archivo directamente desde cualquier directorio.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.riot_client.cache import load_timeline
from src.riot_client.client import (RequestDeadlineExceeded, RiotUnavailableError,
                                    check_deadline, request_deadline)
from src.riot_client.endpoints import get_match_timeline

DB_PATH = PROJECT_ROOT / "data" / "db" / "lol.sqlite"
CUTOFF_TS = 1794247781000  # 2026-11-09T18:09:41+00:00, en milisegundos
MAX_CONSECUTIVE_ERRORS = 20


def download(conn, max_downloads=None, minutes=None, out=print):
    """Lee únicamente target_matches y devuelve (conteos, código de salida)."""
    if max_downloads is not None and max_downloads < 1:
        raise ValueError("El máximo debe ser positivo.")
    if minutes is not None and (not math.isfinite(minutes) or minutes <= 0):
        raise ValueError("Los minutos deben ser positivos y finitos.")
    deadline = None if minutes is None else time.monotonic() + minutes * 60
    counts = dict(descargadas=0, en_cache=0, no_encontradas=0, errores=0)
    consecutive_errors = 0
    processed = 0
    exit_code = 0

    def summary():
        out(" ".join(f"{key}={value}" for key, value in counts.items()))

    try:
        with request_deadline(deadline):
            check_deadline()
            rows = conn.execute("""
                SELECT match_id FROM target_matches WHERE status = 'history_done'
                ORDER BY CASE WHEN game_start_ts > ? THEN 0 ELSE 1 END,
                         CASE WHEN game_start_ts > ? THEN game_start_ts END ASC,
                         CASE WHEN game_start_ts <= ? THEN game_start_ts END DESC,
                         match_id ASC
            """, (CUTOFF_TS,) * 3)
            for (match_id,) in rows:
                check_deadline()
                if max_downloads is not None and counts["descargadas"] >= max_downloads:
                    break
                stop = False
                if load_timeline(match_id) is not None:
                    counts["en_cache"] += 1
                else:
                    check_deadline()
                    try:
                        get_match_timeline(match_id)
                    except (requests.RequestException, RiotUnavailableError) as error:
                        status = getattr(getattr(error, "response", None), "status_code", None)
                        if status == 404:
                            counts["no_encontradas"] += 1
                            consecutive_errors = 0
                        else:
                            counts["errores"] += 1
                            consecutive_errors += 1
                            if status in (401, 403):
                                out("Riot rechazó la clave (HTTP 401 o 403). Puede haber caducado.")
                                stop = True
                            elif consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                stop = True
                            if stop:
                                exit_code = 1
                    else:
                        counts["descargadas"] += 1
                        consecutive_errors = 0
                processed += 1
                if processed % 100 == 0:
                    summary()
                if stop:
                    break
    except (RequestDeadlineExceeded, KeyboardInterrupt):
        pass
    finally:
        summary()
    return counts, exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", dest="max_downloads", type=int)
    parser.add_argument("--minutes", type=float)
    args = parser.parse_args(argv)
    if args.max_downloads is not None and args.max_downloads < 1:
        parser.error("--max debe ser positivo")
    if args.minutes is not None and (not math.isfinite(args.minutes) or args.minutes <= 0):
        parser.error("--minutes debe ser positivo y finito")
    try:
        conn = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            _, code = download(conn, args.max_downloads, args.minutes)
            return code
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        print("No se pudo leer la base de datos o guardar la caché.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
