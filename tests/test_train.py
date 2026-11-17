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


def test_temporal_validation_finds_the_signal_and_reports_auc_per_round():
    result = train.temporal_validation(synthetic_table(n=300))
    table = result["tabla"]
    assert set(table.index) == {"Siempre la clase mayoritaria", "Regresión logística", "Bosque aleatorio"}
    assert table.loc["Regresión logística", "accuracy"] > 0.75
    assert np.isnan(table.loc["Siempre la clase mayoritaria", "roc_auc"]), "a constant has no AUC"
    assert len(result["auc_por_corte"]) == 5
    assert result["auc_por_corte"]["Siempre la clase mayoritaria"].isna().all()
    assert (result["auc_por_corte"]["Regresión logística"] > 0.7).all()


def test_auc_interval_contains_the_auc():
    y = np.array([0, 1] * 50)
    proba = np.where(y == 1, 0.6, 0.4) + np.random.default_rng(1).normal(scale=0.2, size=100)
    low, high = train.bootstrap_auc_ci(y, proba)
    m = train.evaluate(pd.Series(y), proba)
    assert low <= m["roc_auc"] <= high


def test_paired_difference_counts_only_disagreements():
    y = np.array([1, 1, 0, 0])
    model = np.array([0.9, 0.9, 0.1, 0.9])      # 3 right
    baseline = np.array([0.1, 0.1, 0.1, 0.1])  # 2 right
    result = train.paired_accuracy_difference(y, model, baseline)
    assert result["diferencia"] == pytest.approx(0.25)
    assert result["ic95"][0] <= 0.25 <= result["ic95"][1]


def test_position_features_picks_only_the_position_variables():
    names = ["dif_en_linea_habitual", "top_dif_en_linea_habitual", "dif_fuera_de_posicion", "dif_kda_medio"]
    assert train.position_features(names) == names[:3]


def test_paired_auc_difference_on_identical_and_better_predictions():
    y = np.array([0, 1] * 50)
    noise = np.random.default_rng(2).normal(size=100)
    same = train.paired_auc_difference(y, noise, noise)
    assert same["diferencia"] == 0 and same["ic95"] == (0, 0)
    better = train.paired_auc_difference(y, y + 0.1 * noise, noise)
    assert better["ic95"][0] <= better["diferencia"] <= better["ic95"][1]
    assert better["ic95"][0] > 0


def test_position_sensitivity_shows_when_the_signal_lives_in_position():
    table = synthetic_table(n=300).rename(columns={"dif_kda_medio": "dif_en_linea_habitual"})
    result = train.position_sensitivity(table)
    aucs = result["tabla"]["roc_auc"]
    assert aucs["Con todas las variables"] > 0.8
    assert aucs["Sin variables de posición"] < 0.65
    assert result["diferencia_auc"]["ic95"][0] > 0


def test_temporal_validation_returns_aligned_seed_tier():
    table = synthetic_table()
    result = train.temporal_validation(table, n_splits=3)
    n = len(result["y"])
    assert len(result["seed_tier"]) == n == len(result["probabilidades"]["Regresión logística"])
    # the tiers must follow the same matches as y: check against the sorted table's tail
    ordered = table.sort_values("game_start_ts", kind="stable").reset_index(drop=True)
    assert result["seed_tier"].tolist() == ordered["seed_tier"].iloc[-n:].tolist()
    assert result["y"].tolist() == ordered["blue_win"].iloc[-n:].tolist()


def test_metrics_by_tier_band_counts_and_nan_for_one_class_band():
    rng = np.random.default_rng(1)
    tiers = ["GOLD"] * 60 + ["DIAMOND"] * 40 + ["MASTER"] * 50 + ["CHALLENGER"] * 10
    y = np.r_[rng.integers(0, 2, 60), np.ones(40, dtype=int), rng.integers(0, 2, 60)]
    proba = np.clip(y * 0.3 + rng.random(len(y)) * 0.5, 0, 1)
    table = train.metrics_by_tier_band(y, proba, tiers)
    assert table["partidas"].to_dict() == {"Hierro a Platino": 60, "Esmeralda y Diamante": 40, "Maestro o superior": 60}
    assert np.isnan(table.loc["Esmeralda y Diamante", "roc_auc"])  # one class only
    assert not np.isnan(table.loc["Esmeralda y Diamante", "accuracy"])
    assert 0.5 < table.loc["Hierro a Platino", "roc_auc"] <= 1.0


def test_metrics_by_tier_band_empty_band_is_nan():
    table = train.metrics_by_tier_band([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.6], ["GOLD"] * 4)
    assert table.loc["Maestro o superior", "partidas"] == 0
    assert np.isnan(table.loc["Maestro o superior", "accuracy"])


def test_metrics_by_tier_band_unknown_tier_raises():
    with pytest.raises(ValueError, match="UNRANKED"):
        train.metrics_by_tier_band([0, 1], [0.2, 0.8], ["GOLD", "UNRANKED"])


def test_calibration_summary_and_overconfidence():
    from scipy.special import expit, logit
    # Every probability group has exactly its advertised outcome rate.
    p = np.repeat(np.arange(.1, 1., .1), 100)
    y = np.concatenate([np.r_[np.ones(k*10), np.zeros(100-k*10)] for k in range(1, 10)])
    good = train.calibration_summary(y, p, n_boot=20)
    assert abs(good['intercept']['estimate']) < .01
    assert abs(good['slope']['estimate']-1) < .01
    assert good['ece']['estimate'] < 1e-10
    confident = train.calibration_summary(y, expit(2*logit(p)), n_boot=20)
    assert confident['slope']['estimate'] < .6
    assert all(len(v['ic95']) == 2 for v in good.values())


def test_ece_fixed_width_ten_bins():
    p = np.array([.11, .19, .21, .29])
    y = np.array([0, 1, 0, 0])
    result = train.calibration_summary(y, p, n_boot=10)
    assert result['ece']['estimate'] == pytest.approx(.3)
    constant = train.calibration_summary(y, np.full(4, .5), n_boot=10)
    assert constant['slope']['estimate'] is None
    assert constant['slope']['ic95'] is None


def test_selective_accuracy_counts_and_empty():
    result = train.selective_accuracy([0, 0, 1, 0], [.7, .4, .56, .51])
    assert result.n.tolist() == [3, 2, 1]
    assert result.coverage.tolist() == [.75, .5, .25]
    assert result.accuracy.tolist() == [2/3, .5, 0.]
    assert result.iloc[2].lo == pytest.approx(0.)
    assert result.iloc[2].hi == pytest.approx(.7934506856)
    empty = train.selective_accuracy([1], [.5])
    assert empty.n.eq(0).all() and empty.accuracy.isna().all()
