"""
EXERCISE 1 — your first Riot API calls.

Goal: given a Riot ID (e.g. "Faker#KR1"), print:
  1. the player's PUUID
  2. the IDs of their last 5 ranked solo games
  3. for the most recent one: did they win, which champion, which lane (teamPosition)

Run it from the project root with:
    python -m scripts.hello_riot "YourName#LAN"

Fill in every TODO. Hints are below each one. Use the official docs to check the
exact paths: https://developer.riotgames.com/apis
"""
import sys

import requests

from src.riot_client.config import API_KEY, REGION_URL

# Every request must carry your key in this header. Riot rejects calls without it (401/403).
HEADERS = {"X-Riot-Token": API_KEY}


def get_puuid(game_name: str, tag_line: str) -> str:
    url = f"{REGION_URL}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    content = response.json()
    return content["puuid"]
    


def get_match_ids(puuid: str, count: int = 5) -> list[str]:
    url = f"{REGION_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    response = requests.get(url, headers=HEADERS, timeout=10, params={"queue": 420, "count": count})
    response.raise_for_status()
    match_ids = response.json()
    return match_ids


def get_match(match_id: str) -> dict:
    url = f"{REGION_URL}/lol/match/v5/matches/{match_id}"
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    match = response.json()
    return match


def main():
    if len(sys.argv) != 2 or "#" not in sys.argv[1]:
        print('Usage: python -m scripts.hello_riot "Name#TAG"')
        sys.exit(1)

    game_name, tag_line = sys.argv[1].split("#", 1)

    puuid = get_puuid(game_name, tag_line)
    print("PUUID:", puuid)

    match_ids = get_match_ids(puuid)
    print("Last ranked games:", match_ids)

    if not match_ids:
        print("No matches found")
        return
    
    match = get_match(match_ids[0])
    match_info = match["info"]

    for player in match_info["participants"]:
        if player["puuid"] == puuid:
            win = player["win"]
            champion_name = player["championName"]
            team_position = player["teamPosition"]
            print(f"Win: {win}")
            print(f"Champion name: {champion_name}")
            print(f"Team position: {team_position}")
            break

    game_duration = match_info["gameDuration"]
    game_version = match_info["gameVersion"]
    print(f"Game duration: {game_duration // 60}:{game_duration % 60}")
    print(f"Game version: {game_version}")

    




if __name__ == "__main__":
    main()
