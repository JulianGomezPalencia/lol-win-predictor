"""Shared pytest setup. pytest loads this file automatically before any test.

config.py refuses to start without RIOT_API_KEY (good for real runs). The tests never call
Riot, so they get a dummy key here — that way the suite also runs on a fresh clone with no .env.
setdefault() keeps your real key if one is already loaded.
"""
import os

os.environ.setdefault("RIOT_API_KEY", "test-key-not-used")
