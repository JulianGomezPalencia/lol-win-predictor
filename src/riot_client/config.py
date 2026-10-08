"""Loads settings from the .env file so no secret is ever hardcoded in the code."""
import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root and puts its values in os.environ

API_KEY = os.getenv("RIOT_API_KEY")
PLATFORM = os.getenv("RIOT_PLATFORM", "la1")
REGION = os.getenv("RIOT_REGION", "americas")

if not API_KEY:
    raise RuntimeError("RIOT_API_KEY is missing. Copy .env.example to .env and paste your key.")

# Two kinds of base URL — this is the "routing" detail from the plan.
PLATFORM_URL = f"https://{PLATFORM}.api.riotgames.com"  # league-v4, champion-mastery-v4, spectator-v5
REGION_URL = f"https://{REGION}.api.riotgames.com"      # account-v1, match-v5
