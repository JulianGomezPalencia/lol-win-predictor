"""
AGENT 5 — Player snapshot.

For every player in our valid target matches, save TODAY's:
  - Ranked Solo/Duo rank (tier, division, LP, wins, losses) — NULL if unranked
  - champion mastery for the champions they played in our target matches

These are snapshots of the present, not of the moment the match was played. They're close to
the truth only for recent matches, so treat them as optional features (see NOTES.md).

Cost: 1 rank call + 1 call per champion they played -> roughly 2 calls per player.
Safe to stop and re-run: players that already have a snapshot are skipped.

Run it:   python -m src.agents.player_snapshot [--limit N]
Tests:    pytest tests/test_player_snapshot.py -v
"""
import argparse

from src.riot_client import endpoints
from src.storage import db


def snapshot_players(conn, limit: int | None = None) -> int:
    """Snapshot players that don't have one yet (at most `limit`). Returns how many were saved."""
    puuids = db.get_players_without_snapshot(conn)
    if limit is not None:
        puuids = puuids[:limit]
    for i, puuid in enumerate(puuids, start=1):
        solo_entry = endpoints.get_solo_queue_entry(puuid)
        masteries = []
        for champion_id in db.get_champions_played(conn, puuid):
            mastery = endpoints.get_champion_mastery(puuid, champion_id)
            if mastery is not None:
                masteries.append(mastery)
        db.save_snapshot(conn, puuid, solo_entry, masteries)
        if i % 50 == 0:
            print(f"{i}/{len(puuids)} players")
    return len(puuids)


def main():
    parser = argparse.ArgumentParser(description="Agent 5: snapshot rank and champion mastery of match participants.")
    parser.add_argument("--limit", type=int, help="snapshot at most this many players")
    args = parser.parse_args()

    conn = db.connect()
    saved = snapshot_players(conn, limit=args.limit)
    print(f"Done. {saved} players snapshotted.")
    conn.close()


if __name__ == "__main__":
    main()
