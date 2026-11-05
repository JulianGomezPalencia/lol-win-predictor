"""
Champion pool: how well a champion does in a position, as known strictly BEFORE a given moment.

The pool is every distinct earlier game seen in player_history (one row per game and player, so a
game that shows up in several players' windows is counted once). It holds NO player column: only
when the game started, the champion, the position and whether it was won.

  build_pool(history)                  history rows -> pool DataFrame
  ChampionPool(pool).champion_position_rate(champion, position, t)
                                       win rate of the champion in that position over pool games with
                                       game_start_ts < t, shrunk (k = 20) to the champion's own rate,
                                       and that to the pool's overall rate before t
  champion_position_shares(pool)       {champion_id: {position: share}}, used by roles.infer_positions

Training, the forward test and a live game all call the same code. A live game has `t` after every
row of the pool, so it just uses everything the pool holds.

Tests:    pytest tests/test_champion_pool.py -v
"""
import numpy as np
import pandas as pd

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
POOL_COLUMNS = ["game_start_ts", "champion_id", "team_position", "win"]
SHRINKAGE = 20  # pseudo-games of the prior


def build_pool(history: pd.DataFrame) -> pd.DataFrame:
    """Distinct non-remake games with a known position, sorted by start. No player column."""
    games = history[history["remake"] == 0].drop_duplicates(["history_match_id", "player_id"])
    games = games[games["team_position"].isin(POSITIONS)].sort_values("game_start_ts", kind="stable")
    return games[POOL_COLUMNS].reset_index(drop=True)


def champion_position_shares(pool: pd.DataFrame) -> dict:
    """{champion_id: {position: share of that champion's pool games in that position}}."""
    counts = pool.groupby(["champion_id", "team_position"]).size()
    totals = counts.groupby(level="champion_id").transform("sum")
    shares = {}
    for (champion, position), share in (counts / totals).items():
        shares.setdefault(int(champion), {})[position] = float(share)
    return shares


class ChampionPool:
    """Cumulative wins per champion (and champion x position) over time, queried strictly before t."""

    def __init__(self, pool: pd.DataFrame):
        pool = pool.sort_values("game_start_ts", kind="stable")
        self._all = self._cumulative(pool)
        self._by_champion = {key: self._cumulative(g) for key, g in pool.groupby("champion_id")}
        self._by_pair = {key: self._cumulative(g) for key, g in pool.groupby(["champion_id", "team_position"])}

    @staticmethod
    def _cumulative(rows: pd.DataFrame) -> tuple:
        return rows["game_start_ts"].to_numpy(), np.cumsum(rows["win"].to_numpy(dtype=float))

    @staticmethod
    def _before(store: tuple, t) -> tuple[int, float]:
        """(games, wins) among rows with game_start_ts < t."""
        times, wins = store
        i = int(np.searchsorted(times, t, side="left"))
        return (i, float(wins[i - 1])) if i else (0, 0.0)

    def _rate(self, store, t, prior: float, k: int = SHRINKAGE) -> float:
        n, wins = self._before(store, t) if store is not None else (0, 0.0)
        return (wins + k * prior) / (n + k)

    def overall_rate(self, t) -> float:
        n, wins = self._before(self._all, t)
        return wins / n if n else 0.5

    def champion_rate(self, champion, t) -> float:
        return self._rate(self._by_champion.get(champion), t, self.overall_rate(t))

    def champion_position_rate(self, champion, position, t) -> float:
        return self._rate(self._by_pair.get((champion, position)), t, self.champion_rate(champion, t))
