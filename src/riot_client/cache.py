"""
EXERCISE 3 — a disk cache for match details.

A finished match never changes, so once we've downloaded it we keep it forever:
    data/raw/matches/LA1_1712891757.json.gz

Why .json.gz? A match JSON is ~100 KB of text. gzip shrinks it ~10x, which matters
when you have tens of thousands of matches. Python's `gzip` module reads/writes it
almost exactly like a normal file.

How to check your work:
    pytest tests/test_cache.py -v
"""
import gzip
import json
from pathlib import Path

# Absolute path to <project root>/data/raw/matches, no matter which folder you run Python from.
#   __file__            = .../lol-win-predictor/src/riot_client/cache.py
#   .resolve().parents[2] = .../lol-win-predictor
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "matches"


def cache_path(match_id: str) -> Path:
    """Where the file for this match lives (it may not exist yet)."""
    # TODO: return CACHE_DIR / f"..."   -> e.g. CACHE_DIR / "LA1_1712891757.json.gz"
    # Hint: with pathlib, the `/` operator joins paths. No string concatenation needed.
    return CACHE_DIR / f"{match_id}.json.gz"


def load_match(match_id: str) -> dict | None:
    """Return the cached match, or None if we've never saved it."""
    # TODO:
    #   1. path = cache_path(match_id)
    #   2. if the file doesn't exist -> return None          (Hint: path.exists())
    #   3. otherwise open it and return the parsed JSON:
    #        with gzip.open(path, "rt", encoding="utf-8") as f:
    #            return json.load(f)
    #      "rt" = read text. The `with` block closes the file for you, even if an error happens.
    path = cache_path(match_id)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_match(match_id: str, match: dict) -> None:
    """Write the match to the cache."""
    # TODO:
    #   1. path = cache_path(match_id)
    #   2. make sure the folder exists:  path.parent.mkdir(parents=True, exist_ok=True)
    #        parents=True  -> also create data/ and data/raw/ if missing
    #        exist_ok=True -> don't crash if the folder is already there
    #   3. write it:  gzip.open(path, "wt", encoding="utf-8")  +  json.dump(match, f)
    path = cache_path(match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(match, f)
        


