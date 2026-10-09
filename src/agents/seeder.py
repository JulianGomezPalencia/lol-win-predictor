"""
AGENT 1 — Seeder (EXERCISE 5, part 2).

Collects players from every ranked tier on LAN and saves them in the `players` table.
Later agents use these players as the starting point to find matches.

Why EVERY tier, with the SAME number of players each? If we only seeded Challenger,
the model would only learn how Challenger games work. Taking an equal-sized sample
from each group is called "stratified sampling".

Run it (from the project root, venv active):
    python -m src.agents.seeder

Check your work:
    pytest tests/test_seeder.py -v
"""
from src.riot_client import endpoints
from src.storage import db

TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
DIVISIONS = ["IV", "III", "II", "I"]
APEX_TIERS = ["MASTER", "GRANDMASTER", "CHALLENGER"]  # no divisions — one big league each


def pick_players(entries: list[dict], n: int) -> list[dict]:
    """Return up to n entries, skipping players marked as inactive."""
    # TODO: go through `entries`, skip any where entry["inactive"] is True,
    #       and return the first n that are left.
    # Hint: build a new list, and stop once it has n items. Or: a list comprehension + slicing [:n].
    entry_ls = []
    for entry in entries:
        if entry["inactive"]:
            continue
        entry_ls.append(entry)
        if len(entry_ls) == n:
            break
    return entry_ls 
        


def seed(conn, players_per_division: int = 10) -> int:
    """Seed players from every tier. Returns how many players were saved."""
    # TODO, part A — normal tiers:
    #   for each tier in TIERS:
    #       for each division in DIVISIONS:
    #           entries = endpoints.get_league_entries(tier, division)
    #           for each entry in pick_players(entries, players_per_division):
    #               db.upsert_player(conn, entry["puuid"], tier, division, entry["leaguePoints"])
    players = 0
    for tier in TIERS:
        for division in DIVISIONS:
            entries = endpoints.get_league_entries(tier, division)
            for entry in pick_players(entries, players_per_division):
                db.upsert_player(conn, entry["puuid"], tier, division, entry["leaguePoints"])
                players += 1

    #
    # TODO, part B — apex tiers:
    #   Each normal tier gets 4 divisions x players_per_division players.
    #   Give each apex tier the SAME total, so every tier is equally represented.
    #   for each tier in APEX_TIERS:
    #       league = endpoints.get_apex_league(tier)
    #       the players are in league["entries"]. Their division is always "I".
    #
    # Keep a counter of how many players you saved, and return it at the end.
    #
    # Note: we always write `endpoints.get_league_entries(...)` (module.function) instead of
    # importing the function directly. That lets the tests swap in a fake version.
    for tier in APEX_TIERS:
        league = endpoints.get_apex_league(tier)
        for entry in pick_players(league["entries"], players_per_division * len(DIVISIONS)):
            db.upsert_player(conn, entry["puuid"] , tier, "I", entry["leaguePoints"])
            players +=1
    return players


def main():
    conn = db.connect()
    saved = seed(conn)
    print(f"Saved {saved} players.")
    print("Players per tier:", db.count_players_by_tier(conn))
    conn.close()


if __name__ == "__main__":
    main()
