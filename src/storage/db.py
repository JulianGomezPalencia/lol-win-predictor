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


# --- Agent 2: match discovery ---------------------------------------------


def get_all_players(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return every player row (each works like a dict: row["puuid"], row["tier"], ...)."""
    sql = """
        SELECT puuid, tier 
        FROM players 
        ORDER BY tier
    """
    return conn.execute(sql).fetchall()


def add_target_match(conn: sqlite3.Connection, match_id: str, seed_puuid: str, seed_tier: str) -> bool:
    """Add a match to the queue with status 'pending'. Return True if it was new, False if
    we already had it (two seed players can share a match — we only want it once)."""
    discovered_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sql = """
        INSERT INTO target_matches (match_id, seed_puuid, seed_tier, discovered_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(match_id) DO NOTHING
    """
    cursor = conn.execute(sql, (match_id, seed_puuid, seed_tier, discovered_at))
    conn.commit()

    return cursor.rowcount == 1
   


def count_target_matches_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    """e.g. {"pending": 1500}. Later: {"pending": 200, "fetched": 1100, "invalid": 30, ...}"""
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM target_matches GROUP BY status").fetchall()
    matches_count = {row["status"]: row["n"] for row in rows}
    return matches_count


# --- Agent 3: match fetcher ---------------------------------------


def get_match_ids_by_status(conn: sqlite3.Connection, status: str) -> list[str]:
    """Return the IDs of every target match with this status, e.g. all 'pending' ones."""
    rows = conn.execute("SELECT match_id FROM target_matches WHERE status = ?", (status,)).fetchall()
    return [row["match_id"] for row in rows]


def mark_match_invalid(conn: sqlite3.Connection, match_id: str) -> None:
    """Set this target match's status to 'invalid' (remake, wrong queue, ...)."""
    conn.execute("UPDATE target_matches SET status = 'invalid' WHERE match_id = ?", (match_id,))
    conn.commit()


def save_fetched_match(conn: sqlite3.Connection, match_id: str, summary: dict, participants: list[dict]) -> None:
    """Save the 10 participants AND mark the match as 'fetched' — as ONE all-or-nothing step.

    Why together? Imagine the program crashes after saving the participants but before
    updating the status. The match would still say 'pending', and the next run would
    process it again. Doing both in one TRANSACTION means: either both happen, or neither.

    `with conn:` does exactly that: it commits at the end of the block if everything worked,
    and rolls everything back (undoes it) if an error happens inside. No conn.commit() needed.
    """
    with conn:
        # executemany = run the same INSERT once per item in the list (10 participants).
        # Named placeholders (:puuid) read the values from each dict by key.
        # OR IGNORE keeps it safe if the participants are somehow already there.
        conn.executemany(
            """
            INSERT OR IGNORE INTO match_participants
                (match_id, puuid, team_id, team_position, champion_id, champion_name, win)
            VALUES (:match_id, :puuid, :team_id, :team_position, :champion_id, :champion_name, :win)
            """,
            [{**p, "match_id": match_id} for p in participants],  # {**p, ...} = copy of p plus one key
        )
        conn.execute(
            """
            UPDATE target_matches
            SET status = 'fetched', game_start_ts = :game_start_ts, patch = :patch, blue_win = :blue_win
            WHERE match_id = :match_id
            """,
            {**summary, "match_id": match_id},
        )
    

# --- Agent 4: history builder ---


def get_fetched_matches(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every target match waiting for its players' histories (status 'fetched')."""
    sql = "SELECT match_id, game_start_ts FROM target_matches WHERE status = 'fetched' ORDER BY match_id"
    return conn.execute(sql).fetchall()


def get_participants(conn: sqlite3.Connection, match_id: str) -> list[sqlite3.Row]:
    """The 10 players of one target match."""
    sql = "SELECT puuid, team_id, champion_id FROM match_participants WHERE match_id = ?"
    return conn.execute(sql, (match_id,)).fetchall()


def save_history(conn: sqlite3.Connection, target_match_id: str, rows: list[dict]) -> None:
    """Save all history rows of one target match and mark it 'history_done', in one transaction."""
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO player_history
                (target_match_id, puuid, history_rank, history_match_id, game_start_ts, game_duration,
                 remake, win, champion_id, champion_name, team_position, kills, deaths, assists)
            VALUES
                (:target_match_id, :puuid, :history_rank, :history_match_id, :game_start_ts, :game_duration,
                 :remake, :win, :champion_id, :champion_name, :team_position, :kills, :deaths, :assists)
            """,
            [{**row, "target_match_id": target_match_id} for row in rows],
        )
        conn.execute(
            "UPDATE target_matches SET status = 'history_done' WHERE match_id = ?", (target_match_id,)
        )


# --- Agent 5: player snapshot ---


def get_players_without_snapshot(conn: sqlite3.Connection) -> list[str]:
    """PUUIDs of participants (of valid target matches) we haven't snapshotted yet."""
    sql = """
        SELECT DISTINCT mp.puuid
        FROM match_participants mp
        JOIN target_matches tm ON tm.match_id = mp.match_id
        WHERE tm.status IN ('fetched', 'history_done')
          AND mp.puuid NOT IN (SELECT puuid FROM player_snapshots)
        ORDER BY mp.puuid
    """
    return [row["puuid"] for row in conn.execute(sql).fetchall()]


def get_champions_played(conn: sqlite3.Connection, puuid: str) -> list[int]:
    """Champions this player used in our target matches (the ones whose mastery matters)."""
    sql = "SELECT DISTINCT champion_id FROM match_participants WHERE puuid = ? ORDER BY champion_id"
    return [row["champion_id"] for row in conn.execute(sql, (puuid,)).fetchall()]


def save_snapshot(conn: sqlite3.Connection, puuid: str, solo_entry: dict | None, masteries: list[dict]) -> None:
    """Save one player's rank (NULLs if unranked) and champion masteries, in one transaction."""
    snapshot_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = solo_entry or {}
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO player_snapshots
                (puuid, tier, division, lp, ranked_wins, ranked_losses, snapshot_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (puuid, entry.get("tier"), entry.get("rank"), entry.get("leaguePoints"),
             entry.get("wins"), entry.get("losses"), snapshot_at),
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO champion_mastery
                (puuid, champion_id, mastery_level, mastery_points, last_play_time, snapshot_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(puuid, m["championId"], m["championLevel"], m["championPoints"], m["lastPlayTime"], snapshot_at)
             for m in masteries],
        )
