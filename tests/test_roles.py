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
