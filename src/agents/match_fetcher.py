"""
AGENT 3 — Match fetcher.

For every 'pending' target match:
  1. download it (get_match uses your cache, so re-runs cost nothing)
  2. check it's a real, complete ranked game  -> otherwise mark it 'invalid'
  3. save the 10 participants and the match summary -> mark it 'fetched'

Cost: 1 call per match -> ~950 matches x 1.25 s ≈ 20 minutes (the first time only).
Safe to stop with Ctrl+C and re-run: it only picks up matches that are still 'pending'.

Run it:   python -m src.agents.match_fetcher
Tests:    pytest tests/test_match_fetcher.py -v
"""
import requests

from src.riot_client import endpoints
from src.storage import db

RANKED_SOLO_QUEUE = 420
MIN_GAME_DURATION = 300  # seconds. Shorter = remake (someone never connected)
VALID_POSITIONS = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}
BLUE_TEAM = 100


# --- Pure functions: they only look at the match dict (no API, no database) -------------


def invalid_reason(match: dict) -> str | None:
    """Return WHY the match can't be used, or None if it's fine.

    Check, in this order:
      - info["queueId"] isn't RANKED_SOLO_QUEUE                       -> "wrong queue"
      - it doesn't have exactly 10 participants                       -> "not 10 players"
      - info["gameDuration"] < MIN_GAME_DURATION                      -> "remake"
      - any participant has gameEndedInEarlySurrender == True         -> "remake"
      - any participant's teamPosition isn't in VALID_POSITIONS       -> "missing position"
        (Riot leaves it "" when it can't tell what lane someone played)
    """
    info = match["info"]
    participants = info["participants"]
    if info["queueId"] != RANKED_SOLO_QUEUE:
        return "wrong queue"
    if len(participants) != 10:
        return "not 10 players"
    if info["gameDuration"] < MIN_GAME_DURATION:
        return "remake"
    if any(p.get("gameEndedInEarlySurrender") for p in participants):
        return "remake"
    if any(p.get("teamPosition") not in VALID_POSITIONS for p in participants):
        return "missing position"
    return None


def summarize(match: dict) -> dict:
    """Return {"game_start_ts": ..., "patch": ..., "blue_win": ...}.

    - game_start_ts: info["gameStartTimestamp"], as is (milliseconds)
    - patch: only the first two parts of info["gameVersion"]: "16.8.766.8562" -> "16.8"
    - blue_win: 1 if the team with teamId BLUE_TEAM won, else 0. Look in info["teams"].
    """
    info = match["info"]
    blue_team = next(team for team in info["teams"] if team["teamId"] == BLUE_TEAM)
    return {
        "game_start_ts": info["gameStartTimestamp"],
        "patch": ".".join(info["gameVersion"].split(".")[:2]),
        "blue_win": int(blue_team["win"]),
    }


def extract_participants(match: dict) -> list[dict]:
    """Return one dict per player, with the keys the match_participants table expects:
    puuid, team_id, team_position, champion_id, champion_name, win (as 1/0, not True/False).
    """
    return [
        {
            "puuid": p["puuid"],
            "team_id": p["teamId"],
            "team_position": p["teamPosition"],
            "champion_id": p["championId"],
            "champion_name": p["championName"],
            "win": int(p["win"]),
        }
        for p in match["info"]["participants"]
    ]


# --- The agent loop ------------------------------------------------------------


def fetch_pending(conn) -> dict[str, int]:
    """Process every pending match. Return counts, e.g. {"fetched": 900, "remake": 30, ...}."""
    match_ids = db.get_match_ids_by_status(conn, "pending")
    counts: dict[str, int] = {}
    for i, match_id in enumerate(match_ids, start=1):
        try:
            match = endpoints.get_match(match_id)
        except requests.HTTPError as error:
            if not endpoints.is_not_found(error):
                raise  # e.g. 403 expired key: stop, so nothing is marked wrongly
            match = None
        reason = "not found" if match is None else invalid_reason(match)
        if reason:
            db.mark_match_invalid(conn, match_id)
            counts[reason] = counts.get(reason, 0) + 1
        else:
            db.save_fetched_match(conn, match_id, summarize(match), extract_participants(match))
            counts["fetched"] = counts.get("fetched", 0) + 1
        if i % 50 == 0:
            print(f"{i}/{len(match_ids)} matches, {counts}")
    return counts


def main():
    conn = db.connect()
    counts = fetch_pending(conn)
    print("Done:", counts)
    print("Queue:", db.count_target_matches_by_status(conn))
    conn.close()


if __name__ == "__main__":
    main()
