"""
AGENT 4 — History builder.

For every 'fetched' target match, and for each of its 10 players, download the player's last
HISTORY_DEPTH ranked games played BEFORE the target match started, and store a summary of each
one in `player_history`. Then mark the target match 'history_done'.

The no-leakage rule lives here: we ask Riot only for games that ended before the target match
started (endTime), and then double-check every game's start time ourselves. A history game that
starts at or after the target match would let the model "see the future", so it's dropped.

Cost: per target match 10 id-lists + up to 100 match downloads. Players' games overlap a lot,
so the cache saves many calls, but a full run over ~900 matches can still take a day or more.
Safe to stop with Ctrl+C at any time and re-run: finished matches are 'history_done'.

Run it:   python -m src.agents.history_builder [--limit N]
Tests:    pytest tests/test_history_builder.py -v
"""
import argparse

import requests

from src.agents.match_fetcher import MIN_GAME_DURATION
from src.riot_client import endpoints
from src.storage import db

HISTORY_DEPTH = 10


def summarize_for_player(match: dict, puuid: str) -> dict | None:
    """One history game seen from one player's side, or None if they aren't in it."""
    info = match["info"]
    player = next((p for p in info["participants"] if p["puuid"] == puuid), None)
    if player is None:
        return None
    remake = info["gameDuration"] < MIN_GAME_DURATION or bool(player.get("gameEndedInEarlySurrender"))
    return {
        "history_match_id": match["metadata"]["matchId"],
        "game_start_ts": info["gameStartTimestamp"],
        "game_duration": info["gameDuration"],
        "remake": int(remake),
        "win": int(player["win"]),
        "champion_id": player["championId"],
        "champion_name": player["championName"],
        "team_position": player.get("teamPosition") or "",
        "kills": player["kills"],
        "deaths": player["deaths"],
        "assists": player["assists"],
    }


def player_history(puuid: str, target_start_ts: int, depth: int = HISTORY_DEPTH) -> list[dict]:
    """The player's last `depth` ranked games before `target_start_ts` (milliseconds), newest first."""
    end_time = target_start_ts // 1000 - 1  # Riot wants seconds; -1 so the target match itself is excluded
    try:
        match_ids = endpoints.get_match_ids(puuid, count=depth, end_time=end_time)
    except requests.HTTPError as error:
        if endpoints.is_firewall_block(error):
            return []  # Riot's firewall blocks this player's ID: no history, like a brand-new account
        raise
    rows = []
    for match_id in match_ids:
        try:
            match = endpoints.get_match(match_id)
        except requests.HTTPError as error:
            if endpoints.is_not_found(error):
                continue  # a match Riot can't serve anymore — skip it rather than stop the whole run
            raise  # anything else (e.g. 403 expired key) must stop the run, not save a half history
        summary = summarize_for_player(match, puuid)
        if summary is None or summary["game_start_ts"] >= target_start_ts:
            continue  # not their game, or not strictly before the target match (leakage guard)
        rows.append({"puuid": puuid, "history_rank": len(rows), **summary})
    return rows


def build_histories(conn, limit: int | None = None, depth: int = HISTORY_DEPTH) -> int:
    """Process 'fetched' target matches (at most `limit`). Returns how many were completed."""
    matches = db.get_fetched_matches(conn)
    if limit is not None:
        matches = matches[:limit]
    for i, target in enumerate(matches, start=1):
        rows = []
        for participant in db.get_participants(conn, target["match_id"]):
            rows.extend(player_history(participant["puuid"], target["game_start_ts"], depth))
        db.save_history(conn, target["match_id"], rows)
        print(f"{i}/{len(matches)} {target['match_id']}: {len(rows)} history games")
    return len(matches)


def main():
    parser = argparse.ArgumentParser(description="Agent 4: download each player's games before each target match.")
    parser.add_argument("--limit", type=int, help="process at most this many target matches")
    parser.add_argument("--depth", type=int, default=HISTORY_DEPTH, help="history games per player")
    args = parser.parse_args()

    conn = db.connect()
    done = build_histories(conn, limit=args.limit, depth=args.depth)
    print(f"Done. {done} target matches completed.")
    print("Queue:", db.count_target_matches_by_status(conn))
    conn.close()


if __name__ == "__main__":
    main()
