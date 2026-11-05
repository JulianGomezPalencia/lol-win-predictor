"""Tests for src/features/roles.py (pure, no API). Run: pytest tests/test_roles.py -v"""
from src.features import roles

POS = roles.POSITIONS


def player(key, spells=(4, 14), history=()):
    return {"key": key, "spell_ids": spells, "history_positions": list(history)}


def team_with_history():
    return [player("a", history=["TOP"] * 8), player("b", spells=(11, 4), history=["MIDDLE"] * 5),
            player("c", history=["MIDDLE"] * 9), player("d", history=["BOTTOM"] * 7), player("e", history=["UTILITY"] * 7)]


def test_history_decides_and_smite_wins_jungle():
    assert roles.infer_positions(team_with_history()) == {
        "a": "TOP", "b": "JUNGLE", "c": "MIDDLE", "d": "BOTTOM", "e": "UTILITY"}


def test_smite_beats_a_history_of_another_role():
    team = team_with_history()
    team[1]["history_positions"] = ["TOP"] * 10  # b looks like a top, but carries Smite
    result = roles.infer_positions(team)
    assert result["b"] == "JUNGLE"


def test_output_is_one_to_one():
    result = roles.infer_positions(team_with_history())
    assert sorted(result.values()) == sorted(POS)


def test_no_history_gives_valid_assignment_for_one_and_for_all():
    team = team_with_history()
    team[0]["history_positions"] = []
    assert sorted(roles.infer_positions(team).values()) == sorted(POS)
    nobody = [player(k) for k in "abcde"]
    assert sorted(roles.infer_positions(nobody).values()) == sorted(POS)


def test_deterministic_and_independent_of_input_order():
    nobody = [player(k) for k in "abcde"]
    first = roles.infer_positions(nobody)
    assert roles.infer_positions(list(nobody)) == first
    assert roles.infer_positions(list(reversed(nobody))) == first
    team = team_with_history()
    assert roles.infer_positions(list(reversed(team))) == roles.infer_positions(team)


def test_fewer_than_five_players():
    assert roles.infer_positions([]) == {}
    two = roles.infer_positions([player("a", history=["TOP"] * 3), player("b", history=["UTILITY"] * 3)])
    assert two == {"a": "TOP", "b": "UTILITY"}
    assert len(set(roles.infer_positions([player(k) for k in "abc"]).values())) == 3


def test_blank_and_unknown_history_entries_are_ignored():
    p = [player("a", history=["", "", "Invalid"])]
    assert roles.infer_positions(p)["a"] in POS


# --- champion in role inference ---------------------------------------------------------------

PYKE, RIVEN = 555, 266
TABLE = {PYKE: {"UTILITY": 0.9, "JUNGLE": 0.1}, RIVEN: {"TOP": 0.95, "MIDDLE": 0.05}}


def swap_team(first, second):
    """Riven and Pyke have no history; three other players fill JUNGLE, MIDDLE and BOTTOM."""
    return [{**player(first, history=()), "champion_id": RIVEN}, {**player(second, history=()), "champion_id": PYKE},
            player("jg", spells=(11, 4), history=["JUNGLE"] * 6), player("mid", history=["MIDDLE"] * 6),
            player("adc", history=["BOTTOM"] * 6)]


def test_no_table_or_unknown_champions_give_the_old_result_exactly():
    team = team_with_history()
    old = roles.infer_positions(team)
    assert roles.infer_positions(team, None) == old and roles.infer_positions(team, {}) == old
    with_ids = [{**p, "champion_id": 99} for p in team]
    assert roles.infer_positions(with_ids, TABLE) == old  # champion 99 is not in the table


def test_champion_table_fixes_a_pyke_riven_swap_for_players_without_history():
    ok = []
    for first, second in (("a", "b"), ("b", "a")):  # swap which key comes first
        ok.append(roles.infer_positions(swap_team(first, second))[first] == "TOP")
        fixed = roles.infer_positions(swap_team(first, second), TABLE)
        assert fixed[first] == "TOP" and fixed[second] == "UTILITY"
    assert not all(ok)  # without the table the tie is broken by order, so one of the two orders is wrong


def test_smite_still_wins_jungle_with_a_champion_table_and_it_stays_one_to_one_and_deterministic():
    team = swap_team("a", "b")
    team[2]["champion_id"] = RIVEN  # the Smite player plays a champion usually seen in TOP
    team[0]["champion_id"] = PYKE
    result = roles.infer_positions(team, TABLE)
    assert result["jg"] == "JUNGLE"
    assert sorted(result.values()) == sorted(POS)
    assert result == roles.infer_positions(team, TABLE)
