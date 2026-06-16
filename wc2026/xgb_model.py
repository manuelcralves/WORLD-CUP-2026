"""XGBoost (gradient boosting) variant with a Poisson objective.

Trains a boosting regressor on the SAME stacked representation (each team's
perspective) and with the same features as the Poisson model, predicting the
expected goals. Exposes `predict_lambda`/`rho` to plug into the same evaluation
and simulation tools.

Meant to answer: does "classic ML" beat the statistical model? (Spoiler: with
only 6 highly informative features and low counts, they end up practically
tied — and an ensemble of the two is the safest bet.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from . import data as D
from . import features as F
from . import model as M
from . import validate as V


class XGBModel:
    """Interface compatible with PoissonModel (.predict_lambda, .rho)."""

    def __init__(self, booster, features, rho=0.0):
        self.booster = booster
        self.features = features
        self.rho = rho

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(X[self.features], feature_names=list(self.features))
        return self.booster.predict(d)

    def lambdas_for(self, state, default, home, away, neutral):
        rh, ra = F.fixture_rows(state, default, home, away, neutral)
        lam = self.predict_lambda(pd.DataFrame([rh, ra]))
        return float(lam[0]), float(lam[1])


class EnsembleModel:
    """Average of the expected goals from several models (e.g. Poisson + XGBoost)."""

    def __init__(self, models, rho=0.0):
        self.models = models
        self.features = models[0].features
        self.rho = rho

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_lambda(X) for m in self.models], axis=0)

    def lambdas_for(self, state, default, home, away, neutral):
        rh, ra = F.fixture_rows(state, default, home, away, neutral)
        lam = self.predict_lambda(pd.DataFrame([rh, ra]))
        return float(lam[0]), float(lam[1])


def fit_xgb(X, y, w, features=None, num_round=350) -> XGBModel:
    features = list(features) if features is not None else list(F.FEATURES)
    dtrain = xgb.DMatrix(X[features], label=y, weight=w, feature_names=features)
    params = {
        "objective": "count:poisson", "eval_metric": "poisson-nloglik",
        "max_depth": 4, "eta": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.9, "min_child_weight": 8,
        "lambda": 1.0, "seed": 42, "nthread": 4,
    }
    booster = xgb.train(params, dtrain, num_boost_round=num_round)
    return XGBModel(booster, features)


def compare(bundle, cutoff="2022-01-01", end="2026-06-10") -> pd.DataFrame:
    """Compares Poisson vs XGBoost vs ensemble on the same temporal holdout."""
    train, test = V._split(bundle, cutoff, end)
    Xtr, ytr, wtr = F.build_training(train, halflife=M.BEST_PARAMS["halflife"])

    pois = M.fit(Xtr, ytr, wtr, l2=M.BEST_PARAMS["l2"])
    M.fit_rho(pois, train)
    xgbm = fit_xgb(Xtr, ytr, wtr)
    V._fit_rho_generic(xgbm, train)

    home, away = V._match_frames(test)
    actual = V._outcomes(test)
    p_pois = V._probs_from_model(pois, home, away)
    p_xgb = V._probs_from_model(xgbm, home, away)
    p_ens = (p_pois + p_xgb) / 2

    rows = []
    for name, p in [("poisson", p_pois), ("xgboost", p_xgb), ("ensemble", p_ens)]:
        rows.append({"modelo": name, "accuracy": V.accuracy(p, actual),
                     "log_loss": V.log_loss(p, actual), "RPS": V.rps(p, actual)})
    return pd.DataFrame(rows).set_index("modelo")


def train_xgb_full(bundle, **kw):
    """Trains XGBoost on all the data and returns a 'trained' object usable
    by the simulation (same structure as model.train_full, but with the XGBModel)."""
    base = M.train_full(bundle, **kw)
    xgbm = fit_xgb(base["X"], base["y"], base["w"])
    V._fit_rho_generic(xgbm, base["played_aug"])
    base = dict(base)
    base["model"] = xgbm
    return base


def train_ensemble_full(bundle, **kw):
    """Trains the Poisson + XGBoost ensemble (the best on the holdout) for the simulation."""
    base = M.train_full(bundle, **kw)
    xgbm = fit_xgb(base["X"], base["y"], base["w"])
    ens = EnsembleModel([base["model"], xgbm])
    V._fit_rho_generic(ens, base["played_aug"])
    base = dict(base)
    base["model"] = ens
    return base


if __name__ == "__main__":
    b = D.load_all()
    print("Comparing Poisson vs XGBoost vs ensemble (holdout 2022-2026)...")
    res = compare(b)
    print(res.round(4).to_string())
    best = res["RPS"].idxmin()
    print(f"\nBest RPS: '{best}'. (RPS and log-loss: lower = better.)")
