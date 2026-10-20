import requests

from src.riot_client.config import REGION_URL, PLATFORM_URL
from src.riot_client.client import get
from src.riot_client.cache import load_match, save_match


def get_puuid(game_name: str, tag_line: str) -> str:
    account = get(f"{REGION_URL}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}")
    return account["puuid"]


def get_match_ids(puuid: str, count: int = 5, start_time: int | None = None, end_time: int | None = None) -> list[str]:
    params = {"queue": 420, "count": count}

    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    match_ids = get(f"{REGION_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids", params=params)
    return match_ids


def get_match(match_id: str) -> dict:
    cached = load_match(match_id)

    if cached is not None:
        return cached

    match = get(f"{REGION_URL}/lol/match/v5/matches/{match_id}")
    save_match(match_id, match)
    return match

def get_league_entries(tier: str, division: str) -> list[dict]:
    league_entries = get(f"{PLATFORM_URL}/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}")
    return league_entries

def get_apex_league(tier: str) -> dict:
    apex_league = get(f"{PLATFORM_URL}/lol/league/v4/{tier.lower()}leagues/by-queue/RANKED_SOLO_5x5")
    return apex_league


def get_solo_queue_entry(puuid: str) -> dict | None:
    """The player's Ranked Solo/Duo entry (tier, rank, leaguePoints, wins, losses), or None if unranked."""
    entries = get(f"{PLATFORM_URL}/lol/league/v4/entries/by-puuid/{puuid}")
    return next((e for e in entries if e["queueType"] == "RANKED_SOLO_5x5"), None)


def get_champion_mastery(puuid: str, champion_id: int) -> dict | None:
    """Mastery of one champion for one player, or None if they've never played it (Riot answers 404)."""
    try:
        return get(f"{PLATFORM_URL}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}")
    except requests.HTTPError as error:
        if is_not_found(error):
            return None
        raise


def is_not_found(error: requests.HTTPError) -> bool:
    """True if Riot answered 404. Only a 404 is safe to skip: a 403 means the key expired,
    and skipping that would silently save incomplete data."""
    return error.response is not None and error.response.status_code == 404
