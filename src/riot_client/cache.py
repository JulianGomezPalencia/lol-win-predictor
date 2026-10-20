"""
A disk cache for match details.

A finished match never changes, so once we've downloaded it we keep it forever:
    data/raw/matches/LA1_1712891757.json.gz

Why .json.gz? A match JSON is ~100 KB of text. gzip shrinks it ~10x, which matters
when you have tens of thousands of matches. Python's `gzip` module reads/writes it
almost exactly like a normal file.

Tests: pytest tests/test_cache.py -v
"""
import gzip
import json
import os
from pathlib import Path

# Absolute path to <project root>/data/raw/matches, no matter which folder you run Python from.
#   __file__            = .../lol-win-predictor/src/riot_client/cache.py
#   .resolve().parents[2] = .../lol-win-predictor
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "matches"


def cache_path(match_id: str) -> Path:
    """Where the file for this match lives (it may not exist yet)."""
    return CACHE_DIR / f"{match_id}.json.gz"


def load_match(match_id: str) -> dict | None:
    """Return the cached match, or None if we've never saved it (or the file is damaged)."""
    path = cache_path(match_id)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, EOFError, ValueError):
        # A damaged file (e.g. from a crash in an older version) — treat it as missing,
        # so the match is simply downloaded again and the file overwritten.
        return None


def save_match(match_id: str, match: dict) -> None:
    """Write the match to the cache.

    We write to a temporary file first and then rename it. A rename is "atomic": it either
    fully happens or not at all. So if the program is stopped mid-write, the only thing left
    behind is a .tmp file, never a half-written .json.gz that would break load_match().
    """
    path = cache_path(match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(match, f)
    os.replace(tmp_path, path)
