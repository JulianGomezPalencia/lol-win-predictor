from src.riot_client.config import REGION_URL
from src.riot_client.client import get


def get_puuid(game_name: str, tag_line: str) -> str:
    account = get(f"{REGION_URL}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}")
    return account["puuid"]
    


def get_match_ids(puuid: str, count: int = 5) -> list[str]:
    match_ids = get(f"{REGION_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids", params={"queue": 420, "count": count})
    return match_ids


def get_match(match_id: str) -> dict:
    match = get(f"{REGION_URL}/lol/match/v5/matches/{match_id}")
    return match



