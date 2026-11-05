"""
Role inference from PRE-GAME information only.

Riot's teamPosition is assigned after the game, and the spectator endpoint (live games) has no
position at all. So we guess each player's role from two things that are known before the match:

  - the summoner spells picked in champion select (Smite = jungler), and
  - the positions the player played in their previous games (history, remakes excluded).

infer_positions() works on ONE team of up to five players and returns a one-to-one assignment of
players to positions. It is pure: no API, no files, no key. Same input -> same output.

The same function is used to build the training table and to predict, so the two always agree.

Tests:    pytest tests/test_roles.py -v
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SMITE = 11
SMITE_BONUS = 2.0      # strong: Smite almost always means jungle
NO_SMITE_PENALTY = 0.5  # smaller: a jungler without Smite is rare but possible
TIE_BREAK = 1e-6       # makes players with no history resolve the same way every time


def _scores(player: dict) -> np.ndarray:
    """Score of one player for each position in POSITIONS (higher = more likely)."""
    history = [p for p in player.get("history_positions") or [] if p in POSITIONS]
    scores = np.array([history.count(p) / len(history) if history else 0.0 for p in POSITIONS])
    jungle = POSITIONS.index("JUNGLE")
    if SMITE in (player.get("spell_ids") or ()):
        scores[jungle] += SMITE_BONUS
    else:
        scores[jungle] -= NO_SMITE_PENALTY
    return scores


def infer_positions(players: list[dict]) -> dict:
    """Assign each player of one team a different position.

    players: dicts with `key` (any hashable id), `spell_ids` (summoner spell ids) and
    `history_positions` (positions from games BEFORE the match; blanks and remakes already gone).
    Returns {key: position}. Fewer than five players is fine: some positions stay unused.
    """
    ordered = sorted(players, key=lambda p: str(p["key"]))  # input order never changes the answer
    if not ordered:
        return {}
    score = np.array([_scores(p) for p in ordered])
    # Fixed, tiny preference by (player order, position order): only matters on exact ties.
    tie = np.array([[TIE_BREAK * (i * len(POSITIONS) + j) for j in range(len(POSITIONS))]
                    for i in range(len(ordered))])
    rows, cols = linear_sum_assignment(-(score) + tie)
    return {ordered[r]["key"]: POSITIONS[c] for r, c in zip(rows, cols)}
