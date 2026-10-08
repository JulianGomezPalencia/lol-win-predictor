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
    # TODO: call account-v1 "by-riot-id" and return the "puuid" field.
    # Hint: url = f"{REGION_URL}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    # Hint: response = requests.get(url, headers=HEADERS, timeout=10)
    # Hint: response.raise_for_status() turns 4xx/5xx errors into exceptions — use it!
    # Hint: response.json() gives you a Python dict.
    raise NotImplementedError


def get_match_ids(puuid: str, count: int = 5) -> list[str]:
    # TODO: call match-v5 "matches/by-puuid/{puuid}/ids".
    # Hint: pass query parameters with requests.get(url, headers=..., params={"queue": 420, "count": count})
    #       queue 420 = Ranked Solo/Duo
    raise NotImplementedError


def get_match(match_id: str) -> dict:
    # TODO: call match-v5 "matches/{match_id}" and return the JSON.
    raise NotImplementedError


def main():
    if len(sys.argv) != 2 or "#" not in sys.argv[1]:
        print('Usage: python -m scripts.hello_riot "Name#TAG"')
        sys.exit(1)

    game_name, tag_line = sys.argv[1].split("#", 1)

    puuid = get_puuid(game_name, tag_line)
    print("PUUID:", puuid)

    match_ids = get_match_ids(puuid)
    print("Last ranked games:", match_ids)

    # TODO: fetch the first match, then find YOUR participant inside it.
    # Hint: match["info"]["participants"] is a list of 10 dicts.
    #       Find the one whose "puuid" equals yours.
    #       Then print its "win", "championName" and "teamPosition".
    # Bonus: print match["info"]["gameDuration"] and match["info"]["gameVersion"] (the patch).


if __name__ == "__main__":
    main()
