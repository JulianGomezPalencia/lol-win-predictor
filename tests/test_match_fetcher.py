"""
Tests for Exercise 7: Agent 3 (match fetcher).

`make_match()` builds a fake match with the same shape as Riot's real JSON. Each test
changes one thing (e.g. a short duration) to check that one rule.

Run with:  pytest tests/test_match_fetcher.py -v
"""
import pytest
import requests

from src.agents import match_fetcher
from src.riot_client import endpoints
from src.storage import db

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def make_match(queue=420, duration=1800, blue_wins=True, early_surrender=False, n_players=10,
               version="16.8.766.8562"):
    participants = []
    for i in range(n_players):
        team = 100 if i < 5 else 200
        participants.append({
            "puuid": f"p{i}",
            "teamId": team,
            "teamPosition": POSITIONS[i % 5],
            "championId": 100 + i,
            "championName": f"Champ{i}",
            "win": (team == 100) == blue_wins,
            "gameEndedInEarlySurrender": early_surrender,
        })
    return {
        "metadata": {"matchId": "LA1_1"},
        "info": {
            "queueId": queue,
            "gameDuration": duration,
            "gameStartTimestamp": 1_777_076_935_212,
            "gameVersion": version,
            "teams": [{"teamId": 100, "win": blue_wins}, {"teamId": 200, "win": not blue_wins}],
            "participants": participants,
        },
    }


# --- Part 1: pure functions ------------------------------------------------------------


def test_valid_match_has_no_reason():
    assert match_fetcher.invalid_reason(make_match()) is None


def test_wrong_queue():
    assert match_fetcher.invalid_reason(make_match(queue=450)) == "wrong queue"  # 450 = ARAM


def test_not_ten_players():
    assert match_fetcher.invalid_reason(make_match(n_players=9)) == "not 10 players"


def test_short_game_is_remake():
    assert match_fetcher.invalid_reason(make_match(duration=200)) == "remake"


def test_early_surrender_is_remake():
    assert match_fetcher.invalid_reason(make_match(early_surrender=True)) == "remake"


def test_missing_position():
    match = make_match()
    match["info"]["participants"][3]["teamPosition"] = ""
    assert match_fetcher.invalid_reason(match) == "missing position"


def test_summarize_blue_win():
    assert match_fetcher.summarize(make_match(blue_wins=True)) == {
        "game_start_ts": 1_777_076_935_212, "patch": "16.8", "blue_win": 1,
    }


def test_summarize_red_win_and_two_digit_minor_patch():
    summary = match_fetcher.summarize(make_match(blue_wins=False, version="16.12.1.1"))
    assert summary["blue_win"] == 0
    assert summary["patch"] == "16.12", "split on dots — don't cut a fixed number of characters"


def test_extract_participants():
    players = match_fetcher.extract_participants(make_match(blue_wins=False))
    assert len(players) == 10
    assert players[0] == {
        "puuid": "p0", "team_id": 100, "team_position": "TOP",
        "champion_id": 100, "champion_name": "Champ0", "win": 0,
    }
    assert players[5]["win"] == 1 and isinstance(players[5]["win"], int)


# --- Part 2: database ------------------------------------------------------------------
# These use a real FILE and reopen it — that's how we catch a missing commit (Exercise 6!).


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(path)
    db.add_target_match(conn, "LA1_1", "seed", "GOLD")
    conn.close()
    return path


def test_mark_match_invalid_is_saved(db_path):
    conn = db.connect(db_path)
    db.mark_match_invalid(conn, "LA1_1")
    conn.close()
    assert db.count_target_matches_by_status(db.connect(db_path)) == {"invalid": 1}


def test_save_fetched_match_is_saved(db_path):
    match = make_match()
    conn = db.connect(db_path)
    db.save_fetched_match(conn, "LA1_1", match_fetcher.summarize(match),
                          match_fetcher.extract_participants(match))
    conn.close()

    conn = db.connect(db_path)
    row = conn.execute("SELECT * FROM target_matches WHERE match_id = 'LA1_1'").fetchone()
    assert (row["status"], row["patch"], row["blue_win"]) == ("fetched", "16.8", 1)
    assert row["game_start_ts"] == 1_777_076_935_212
    assert conn.execute("SELECT COUNT(*) FROM match_participants").fetchone()[0] == 10


# --- Part 3: the agent -----------------------------------------------------------------


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def test_fetch_pending(conn, monkeypatch):
    fake_matches = {
        "LA1_GOOD": make_match(),
        "LA1_REMAKE": make_match(duration=120),
        "LA1_ARAM": make_match(queue=450),
    }
    monkeypatch.setattr(endpoints, "get_match", lambda match_id: fake_matches[match_id])
    for match_id in fake_matches:
        db.add_target_match(conn, match_id, "seed", "GOLD")

    counts = match_fetcher.fetch_pending(conn)

    assert counts == {"fetched": 1, "remake": 1, "wrong queue": 1}
    assert db.count_target_matches_by_status(conn) == {"fetched": 1, "invalid": 2}
    assert match_fetcher.fetch_pending(conn) == {}, "nothing pending left -> nothing to do"


class FakeHTTPResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_fetch_pending_marks_missing_match_invalid_instead_of_crashing(conn, monkeypatch):
    def fake_get_match(match_id):
        if match_id == "LA1_GONE":
            raise requests.HTTPError("404", response=FakeHTTPResponse(404))
        return make_match()
    monkeypatch.setattr(endpoints, "get_match", fake_get_match)
    db.add_target_match(conn, "LA1_GONE", "seed", "GOLD")
    db.add_target_match(conn, "LA1_GOOD", "seed", "GOLD")

    assert match_fetcher.fetch_pending(conn) == {"not found": 1, "fetched": 1}
    assert db.count_target_matches_by_status(conn) == {"fetched": 1, "invalid": 1}


def test_fetch_pending_stops_on_expired_key(conn, monkeypatch):
    def expired_key(match_id):
        raise requests.HTTPError("403", response=FakeHTTPResponse(403))
    monkeypatch.setattr(endpoints, "get_match", expired_key)
    db.add_target_match(conn, "LA1_1", "seed", "GOLD")
    with pytest.raises(requests.HTTPError):
        match_fetcher.fetch_pending(conn)
    assert db.count_target_matches_by_status(conn) == {"pending": 1}, "still pending -> retried next run"
