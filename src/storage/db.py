"""
EXERCISE 4 — the project database (SQLite).

Why a database and not files? The match JSONs are fine as files (we only ever look them up
by ID). But the pipeline needs to ASK QUESTIONS about its data:
  "which players haven't been processed yet?", "how many players per tier do we have?"
SQL answers those in one line. SQLite is a whole database in a single file
(data/db/lol.sqlite) and comes built into Python — nothing to install.

How to check your work:
    pytest tests/test_db.py -v
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "db" / "lol.sqlite"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open the database (creating it and its tables if needed) and return the connection.

    Tests pass path=":memory:" to get a throwaway database that lives only in RAM.
    """
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["tier"] instead of row[1]
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


def upsert_player(conn: sqlite3.Connection, puuid: str, tier: str, division: str, lp: int) -> None:
    """Insert a player, or update their tier/division/lp if they're already in the table."""
    seeded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sql = """
        INSERT INTO players (puuid, tier, division, lp, seeded_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(puuid) DO UPDATE SET
            tier = excluded.tier, 
            division = excluded.division,
            lp = excluded.lp, 
            seeded_at = excluded.seeded_at
    """
    conn.execute(sql, (puuid, tier, division, lp, seeded_at))
    conn.commit()
            


def count_players_by_tier(conn: sqlite3.Connection) -> dict[str, int]:
    """Return how many players we have in each tier, e.g. {"GOLD": 40, "IRON": 40}."""
    rows = conn.execute("SELECT tier, COUNT(*) AS n FROM players GROUP BY tier").fetchall()
    tier_count = {row["tier"]: row["n"] for row in rows}
    return tier_count