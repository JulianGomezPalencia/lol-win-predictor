"""
Tests for src/models/train.py on small synthetic tables (no real data needed).

Run with:  pytest tests/test_train.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.models import train


def synthetic_table(n=200, seed=0):
    """Matches where blue wins more often when dif_kda_medio is positive (a learnable signal)."""
    rng = np.random.default_rng(seed)
    kda = rng.normal(size=n)
    blue_win = (kda + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({
        "match_id": [f"M{i}" for i in range(n)],
        "game_start_ts": rng.permutation(n) * 1000,  # shuffled on purpose
        "seed_tier": rng.choice(["GOLD", "MASTER", "MASTER"], size=n),
        "blue_win": blue_win,
        "dif_kda_medio": kda,
        "dif_winrate_10": rng.normal(size=n),
        "top_dif_winrate_10": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
    })


def test_split_is_chronological_and_disjoint():
    train_part, test_part = train.chronological_split(synthetic_table())
    assert len(train_part) == 160 and len(test_part) == 40
    assert train_part["game_start_ts"].max() < test_part["game_start_ts"].min()
    assert set(train_part["match_id"]).isdisjoint(test_part["match_id"])


def test_tier_weights_give_every_tier_the_same_total():
    tiers = pd.Series(["GOLD"] * 10 + ["MASTER"] * 30)
    weights = train.tier_weights(tiers)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[:10].sum() == pytest.approx(weights[10:].sum())


def test_baseline_majority_predicts_training_rate():
    proba = train.baseline_majority(pd.Series([1, 0, 0, 0]), 3)
    assert proba.tolist() == [0.25, 0.25, 0.25]


def test_baseline_better_form_and_ties():
    X = pd.DataFrame({"dif_winrate_10": [0.2, -0.1, 0.0, np.nan]})
    proba = train.baseline_better_form(X, pd.Series([0, 0, 1]))  # majority: red (0)
    assert proba.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_evaluate_perfect_and_constant_predictions():
    y = pd.Series([0, 1, 1, 0])
    perfect = train.evaluate(y, np.array([0.1, 0.9, 0.8, 0.2]))
    assert perfect["accuracy"] == 1.0 and perfect["roc_auc"] == 1.0
    constant = train.evaluate(y, np.full(4, 0.4))
    assert constant["accuracy"] == 0.5 and np.isnan(constant["roc_auc"])
    assert constant["confusion"].tolist() == [[2, 0], [2, 0]]


def test_run_learns_a_real_signal_and_handles_missing_values():
    result = train.run(synthetic_table(n=400))
    metrics = result["metrics"]
    assert metrics["Regresión logística"]["accuracy"] > 0.75
    assert metrics["Bosque aleatorio"]["accuracy"] > 0.7
    assert set(result["features"]) == {"dif_kda_medio", "dif_winrate_10", "top_dif_winrate_10"}


def test_temporal_validation_uses_only_older_matches_and_finds_the_signal():
    table = train.temporal_validation(synthetic_table(n=300))
    assert set(table.index) == {"Siempre la clase mayoritaria", "Regresión logística", "Bosque aleatorio"}
    assert table.loc["Regresión logística", "accuracy"] > 0.75
