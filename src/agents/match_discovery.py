"""
AGENT 2 — Match discoveryy.

For each seeded player, ask Riot for their recent ranked match IDs and add them to the
`target_matches` queue as 'pending'. Agent 3 will download them later.

Why only RECENT matches (last 14 days)? Some features (rank, champion mastery) can only be
read as they are TODAY. For a match from last week, today's value is close to the truth.
For a match from a year ago, it's not — that's the data leakage problem from the plan.

Cost: 1 API call per player -> 400 players ≈ 400 x 1.2 s ≈ 8 minutes.
Safe to re-run: duplicates are ignored by add_target_match().

Run it:   python -m src.agents.match_discovery
Check it: pytest tests/test_match_discovery.py -v
"""
import time

from src.riot_client import endpoints
from src.storage import db

SECONDS_PER_DAY = 24 * 60 * 60


def discover(conn, matches_per_player: int = 5, days_back: int = 14) -> int:
    """Add recent ranked matches of every seeded player to the queue. Returns how many were NEW."""
    now = int(time.time())
    start_time = now - days_back * SECONDS_PER_DAY
    players = db.get_all_players(conn)
    new = 0
    for i, player in enumerate(players, start = 1):
        match_ids = endpoints.get_match_ids(player["puuid"], count=matches_per_player,
                                            start_time=start_time)
        for match_id in match_ids:
            if db.add_target_match(conn, match_id, player["puuid"], player["tier"]):
                new += 1
        if i % 25 == 0:
            print(f"{i}/{len(players)} players, {new} new matches")
    return new



def main():
    conn = db.connect()
    new = discover(conn)
    print(f"Done. {new} new matches.")
    print("Queue:", db.count_target_matches_by_status(conn))
    conn.close()


if __name__ == "__main__":
    main()
