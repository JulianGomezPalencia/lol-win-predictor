"""
AGENT 1 — Seeder.

Collects players from every ranked tier on LAN and saves them in the `players` table.
Later agents use these players as the starting point to find matches.

Why EVERY tier, with the SAME number of players each? If we only seeded Challenger,
the model would only learn how Challenger games work. Taking an equal-sized sample
from each group is called "stratified sampling".

Run it (from the project root, venv active):
    python -m src.agents.seeder

Tests:
    pytest tests/test_seeder.py -v
"""
from src.riot_client import endpoints
from src.storage import db

TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
DIVISIONS = ["IV", "III", "II", "I"]
APEX_TIERS = ["MASTER", "GRANDMASTER", "CHALLENGER"]  # no divisions — one big league each


def pick_players(entries: list[dict], n: int) -> list[dict]:
    """Return up to n entries, skipping players marked as inactive."""
    picked = []
    for entry in entries:
        if entry["inactive"]:
            continue
        picked.append(entry)
        if len(picked) == n:
            break
    return picked


def seed(conn, players_per_division: int = 10) -> int:
    """Seed players from every tier. Returns how many players were saved.

    Each normal tier gets 4 divisions x players_per_division players, and each apex tier
    (no divisions, always "I") gets the same total, so every tier is equally represented.

    We call `endpoints.get_league_entries(...)` (module.function) instead of importing the
    function directly. That lets the tests swap in a fake version.
    """
    saved = 0
    for tier in TIERS:
        for division in DIVISIONS:
            entries = endpoints.get_league_entries(tier, division)
            for entry in pick_players(entries, players_per_division):
                db.upsert_player(conn, entry["puuid"], tier, division, entry["leaguePoints"])
                saved += 1

    for tier in APEX_TIERS:
        league = endpoints.get_apex_league(tier)
        for entry in pick_players(league["entries"], players_per_division * len(DIVISIONS)):
            db.upsert_player(conn, entry["puuid"], tier, "I", entry["leaguePoints"])
            saved += 1
    return saved


def main():
    conn = db.connect()
    saved = seed(conn)
    print(f"Saved {saved} players.")
    print("Players per tier:", db.count_players_by_tier(conn))
    conn.close()


if __name__ == "__main__":
    main()
