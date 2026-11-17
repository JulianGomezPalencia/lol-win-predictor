"""
Role inference from PRE-GAME information only.

Riot's teamPosition is assigned after the game, and the spectator endpoint (live games) has no
position at all. So we guess each player's role from three things that are known before the match:

  - the summoner spells picked in champion select (Smite = jungler), and
  - the positions the player played in their previous games (history, remakes excluded), and
  - optionally the champion: where that champion is usually played (a table made from earlier games,
    champion_pool.champion_position_shares). This helps players who are off-role or have no history
    (Pyke is usually support even when a player has no earlier games).

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
CHAMPION_WEIGHT = 1.0   # chosen only by agreement with Riot's team_position (see NOTES.md), never by win results
TIE_BREAK = 1e-6       # makes players with no history resolve the same way every time


def _scores(player: dict, champion_positions: dict | None = None) -> np.ndarray:
    """Score of one player for each position in POSITIONS (higher = more likely)."""
    history = [p for p in player.get("history_positions") or [] if p in POSITIONS]
    scores = np.array([history.count(p) / len(history) if history else 0.0 for p in POSITIONS])
    if champion_positions:
        shares = champion_positions.get(player.get("champion_id")) or {}
        scores = scores + CHAMPION_WEIGHT * np.array([shares.get(p, 0.0) for p in POSITIONS])
    jungle = POSITIONS.index("JUNGLE")
    if SMITE in (player.get("spell_ids") or ()):
        scores[jungle] += SMITE_BONUS
    else:
        scores[jungle] -= NO_SMITE_PENALTY
    return scores


def infer_positions(players: list[dict], champion_positions: dict | None = None) -> dict:
    """Assign each player of one team a different position.

    players: dicts with `key` (any hashable id), `spell_ids` (summoner spell ids) and
    `history_positions` (positions from games BEFORE the match; blanks and remakes already gone) and,
    when `champion_positions` is given, `champion_id`. Without the table the result is exactly the
    one from spells and history alone; a champion missing from the table adds nothing.
    Returns {key: position}. Fewer than five players is fine: some positions stay unused.
    """
    ordered = sorted(players, key=lambda p: str(p["key"]))  # input order never changes the answer
    if not ordered:
        return {}
    score = np.array([_scores(p, champion_positions) for p in ordered])
    # Fixed, tiny preference by (player order, position order). For a full team it adds the same sum to
    # every assignment (so it changes nothing); it only matters when fewer than five players are given.
    tie = np.array([[TIE_BREAK * (i * len(POSITIONS) + j) for j in range(len(POSITIONS))]
                    for i in range(len(ordered))])
    rows, cols = linear_sum_assignment(-(score) + tie)
    return {ordered[r]["key"]: POSITIONS[c] for r, c in zip(rows, cols)}
