from src.riot_client.config import REGION_URL, PLATFORM_URL
from src.riot_client.client import get
from src.riot_client.cache import load_match, save_match


def get_puuid(game_name: str, tag_line: str) -> str:
    account = get(f"{REGION_URL}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}")
    return account["puuid"]
    


def get_match_ids(puuid: str, count: int = 5) -> list[str]:
    match_ids = get(f"{REGION_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids", params={"queue": 420, "count": count})
    return match_ids


def get_match(match_id: str) -> dict:
    cached = load_match(match_id)

    if cached is not None:
        return cached
    
    match = get(f"{REGION_URL}/lol/match/v5/matches/{match_id}")
    save_match(match_id, match)
    return match

def get_league_entries(tier: str, division: str) -> list[str]:
    league_entries = get(f"{PLATFORM_URL}/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}")
    return league_entries

def get_apex_league(tier: str) -> dict:
    apex_league = get(f"{PLATFORM_URL}/lol/league/v4/{tier.lower()}leagues/by-queue/RANKED_SOLO_5x5")
    return apex_league


    



