import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.features.build import FINAL_FEATURES
from src.models.explain import contributions


def test_contributions_reconstruct_logit_with_imputation():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(rng.normal(size=(60, 10)), columns=FINAL_FEATURES)
    frame.iloc[0, 0] = np.nan
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression())])
    pipe.fit(frame, (frame.iloc[:, 2] > 0).astype(int))
    row = frame.iloc[0].to_dict()
    factors = contributions({"features": FINAL_FEATURES, "model": pipe}, row)
    assert np.isclose(sum(c["aporte"] for c in factors) + pipe[-1].intercept_[0], pipe.decision_function(frame.iloc[:1])[0])
    assert [abs(c["aporte"]) for c in factors] == sorted([abs(c["aporte"]) for c in factors], reverse=True)
    assert len(factors) == 10
    assert all("dif_" not in c["frase"] for c in factors)
