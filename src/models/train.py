"""
Train and evaluate the win predictors on the match-level feature table.

  1. Chronological split: the oldest 80 % of matches train, the newest 20 % test. A random split
     would let the model learn from games played after the ones it is tested on.
  2. Two baselines a model must beat: "always the majority class" and "the team with the better
     recent win rate wins".
  3. Logistic regression and random forest, each a scikit-learn Pipeline (impute -> [scale] ->
     model), so the imputation medians and scaling are learned from the training rows only.
     Hyperparameters are chosen with TimeSeriesSplit cross-validation inside the training set.
  4. Tier weights: Master and above make up about half the matches, so each match is weighted by
     1 / (share of its tier) and every tier counts the same in training.

Run it:   python -m src.models.train      # needs data/features/partidas.csv (python -m src.features.build)
Tests:    pytest tests/test_train.py -v
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build import FEATURES_PATH, feature_columns

TEST_SIZE = 0.2
SEED = 42


def chronological_split(table: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oldest matches for training, newest for testing (no match in both)."""
    ordered = table.sort_values("game_start_ts", kind="stable").reset_index(drop=True)
    cut = int(round(len(ordered) * (1 - test_size)))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def tier_weights(tiers: pd.Series) -> np.ndarray:
    """Weight = 1 / share of the tier, scaled so the weights average 1. Every tier then sums the same."""
    share = tiers.map(tiers.value_counts(normalize=True))
    weights = 1 / share
    return (weights / weights.mean()).to_numpy()


def baseline_majority(y_train: pd.Series, n: int) -> np.ndarray:
    """Probability of a blue win = the blue win rate seen in training, for every match."""
    return np.full(n, y_train.mean())


def baseline_better_form(X: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Blue wins if its average recent win rate is higher; ties go to the training majority.

    Returned as hard 0/1 "probabilities": this rule has no notion of confidence.
    """
    diff = X["dif_winrate_10"].fillna(0).to_numpy()
    majority = int(y_train.mean() >= 0.5)
    return np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, float(majority)))


def logistic_model() -> GridSearchCV:
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000))])
    return GridSearchCV(pipe, {"model__C": [0.001, 0.01, 0.1, 1.0]},
                        cv=TimeSeriesSplit(n_splits=5), scoring="neg_log_loss")


def forest_model() -> GridSearchCV:
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))])
    return GridSearchCV(pipe, {"model__max_depth": [2, 4, 6], "model__min_samples_leaf": [10, 30]},
                        cv=TimeSeriesSplit(n_splits=5), scoring="neg_log_loss")


def bootstrap_accuracy_ci(y: np.ndarray, pred: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """95 % interval for the accuracy, resampling the test matches with replacement."""
    rng = np.random.default_rng(seed)
    hits = (np.asarray(y) == np.asarray(pred)).astype(float)
    samples = rng.choice(hits, size=(n_boot, len(hits)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def evaluate(y: pd.Series, proba: np.ndarray) -> dict:
    """Test-set metrics from the predicted probability of a blue win."""
    y = np.asarray(y)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)
    low, high = bootstrap_accuracy_ci(y, pred)
    has_spread = len(np.unique(proba)) > 1
    return {
        "accuracy": accuracy_score(y, pred),
        "accuracy_ic95": (low, high),
        "log_loss": log_loss(y, proba, labels=[0, 1]),
        "brier": brier_score_loss(y, proba),
        "roc_auc": roc_auc_score(y, proba) if has_spread else float("nan"),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]),
    }


def run(table: pd.DataFrame) -> dict:
    """Split, fit both models and the baselines, and return everything the notebook shows."""
    train, test = chronological_split(table)
    features = feature_columns(table)
    X_train, y_train, X_test, y_test = train[features], train["blue_win"], test[features], test["blue_win"]
    weights = tier_weights(train["seed_tier"])

    models = {"Regresión logística": logistic_model(), "Bosque aleatorio": forest_model()}
    probabilities = {
        "Siempre la clase mayoritaria": baseline_majority(y_train, len(test)),
        "Gana el equipo con mejor forma": baseline_better_form(X_test, y_train),
    }
    for name, search in models.items():
        search.fit(X_train, y_train, model__sample_weight=weights)
        probabilities[name] = search.predict_proba(X_test)[:, 1]

    return {
        "train": train, "test": test, "features": features, "models": models,
        "probabilities": probabilities,
        "metrics": {name: evaluate(y_test, proba) for name, proba in probabilities.items()},
    }


def temporal_validation(table: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """A steadier estimate than one test set: walk forward through time over ALL matches.

    Each of the n_splits rounds trains on everything before a cut and predicts the next block, so
    every prediction is still made only from older matches. With 920 matches that gives ~765
    predictions instead of 184, and much narrower intervals.
    """
    ordered = table.sort_values("game_start_ts", kind="stable").reset_index(drop=True)
    features = feature_columns(ordered)
    X, y = ordered[features], ordered["blue_win"]
    collected = {"Siempre la clase mayoritaria": [], "Regresión logística": [], "Bosque aleatorio": []}
    answers = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        weights = tier_weights(ordered["seed_tier"].iloc[train_idx])
        answers.append(y.iloc[test_idx].to_numpy())
        collected["Siempre la clase mayoritaria"].append(baseline_majority(y_tr, len(test_idx)))
        for name, search in (("Regresión logística", logistic_model()), ("Bosque aleatorio", forest_model())):
            search.fit(X_tr, y_tr, model__sample_weight=weights)
            collected[name].append(search.predict_proba(X.iloc[test_idx])[:, 1])
    y_all = pd.Series(np.concatenate(answers))
    return metrics_table({name: evaluate(y_all, np.concatenate(parts)) for name, parts in collected.items()})


def metrics_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        low, high = m["accuracy_ic95"]
        rows.append({"modelo": name, "accuracy": m["accuracy"], "ic95_bajo": low, "ic95_alto": high,
                     "log_loss": m["log_loss"], "brier": m["brier"], "roc_auc": m["roc_auc"]})
    return pd.DataFrame(rows).set_index("modelo").round(3)


def main():
    table = pd.read_csv(FEATURES_PATH)
    result = run(table)
    train, test = result["train"], result["test"]
    print(f"Entrenamiento: {len(train)} partidas | prueba: {len(test)} partidas "
          f"(desde {pd.to_datetime(test['game_start_ts'].min(), unit='ms', utc=True):%Y-%m-%d %H:%M})")
    for name, search in result["models"].items():
        print(f"{name}: mejores hiperparámetros {search.best_params_}")
    print()
    print(metrics_table(result["metrics"]).to_string())
    print()
    print("Validación temporal sobre todas las partidas (5 cortes):")
    print(temporal_validation(table).to_string())


if __name__ == "__main__":
    main()
