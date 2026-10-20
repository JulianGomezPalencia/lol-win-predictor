"""
Export the collected data to CSV files for the EDA notebook (Google Colab).

    python -m scripts.export_dataset            # writes data/export/*.csv

Upload the files in data/export/ to a private Google Drive folder; the notebook reads them there.

What's exported (one CSV per table):
  matches.csv           one row per target match (valid or invalid) — the prediction unit
  participants.csv      the 10 players of each valid match
  player_history.csv    each player's ranked games BEFORE each target match (Agent 4)
  seed_players.csv      the players the sample was built from (Agent 1)
  player_snapshots.csv  today's rank of participants (Agent 5), if any were taken

Privacy: player IDs (PUUIDs) are replaced by a salted hash, so the files can't be linked back to a
Riot account. The salt is generated once and stored in data/export_salt.txt (never committed), so
the same player gets the same anonymous ID across re-exports.

Safe to run while an agent is collecting: it copies a consistent snapshot of the database first.
"""
import csv
import hashlib
import secrets
import sqlite3

from src.storage import db

EXPORT_DIR = db.PROJECT_ROOT / "data" / "export"
SALT_PATH = db.PROJECT_ROOT / "data" / "export_salt.txt"

QUERIES = {
    "matches": """
        SELECT match_id, status, seed_tier, game_start_ts, patch, blue_win, discovered_at
        FROM target_matches ORDER BY match_id""",
    "participants": """
        SELECT mp.match_id, mp.puuid AS player_id, mp.team_id, mp.team_position,
               mp.champion_id, mp.champion_name, mp.win
        FROM match_participants mp ORDER BY mp.match_id, mp.team_id, mp.team_position""",
    "player_history": """
        SELECT target_match_id, puuid AS player_id, history_rank, history_match_id, game_start_ts,
               game_duration, remake, win, champion_id, champion_name, team_position, kills, deaths, assists
        FROM player_history ORDER BY target_match_id, puuid, history_rank""",
    "seed_players": "SELECT puuid AS player_id, tier, division, lp FROM players ORDER BY tier, division",
    "player_snapshots": """
        SELECT puuid AS player_id, tier, division, lp, ranked_wins, ranked_losses, snapshot_at
        FROM player_snapshots ORDER BY puuid""",
}


def load_salt() -> str:
    if not SALT_PATH.exists():
        SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SALT_PATH.write_text(secrets.token_hex(16))
    return SALT_PATH.read_text().strip()


def anonymize(puuid: str, salt: str) -> str:
    return "p_" + hashlib.sha256((salt + puuid).encode()).hexdigest()[:12]


def snapshot_connection() -> sqlite3.Connection:
    """A private in-memory copy of the database, so a running agent can't change it mid-export."""
    source = sqlite3.connect(f"file:{db.DB_PATH}?mode=ro", uri=True, timeout=30)
    copy = sqlite3.connect(":memory:")
    source.backup(copy)
    source.close()
    return copy


def export() -> dict[str, int]:
    salt = load_salt()
    conn = snapshot_connection()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, sql in QUERIES.items():
        cursor = conn.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        id_col = columns.index("player_id") if "player_id" in columns else None
        with open(EXPORT_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in rows:
                row = list(row)
                if id_col is not None:
                    row[id_col] = anonymize(row[id_col], salt)
                writer.writerow(row)
        counts[name] = len(rows)
    conn.close()
    return counts


def main():
    for name, n in export().items():
        print(f"{name}.csv: {n} rows")
    print(f"Written to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
