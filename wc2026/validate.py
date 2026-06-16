"""Honest model validation with a TEMPORAL holdout.

Training uses only matches before a cutoff date and evaluation is on the
following matches (which the model never saw). Three predictors are compared:

  * naive   : always predicts the historical W/D/L frequency (base rate)
  * elo     : Poisson with only (Elo difference + home)  -> strong baseline
  * full    : the full model (Elo + form + home + friendly)

Metrics (lower is better, except accuracy):
  * accuracy : % of correct calls on the most likely outcome
  * log-loss : penalizes confident, wrong predictions
  * RPS      : Ranked Probability Score, the standard metric in football
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import elo as E
from . import features as F
from . import model as M


def _match_frames(df_aug: pd.DataFrame):
    """Feature DataFrames (home and away) for each match, with an extra elo_diff."""
    venue = np.where(df_aug["neutral"].values, 0.0, 1.0)
    friendly = df_aug["tournament"].str.lower().eq("friendly").astype(float).values
    home = pd.DataFrame({
        "elo_self": df_aug["home_elo_pre"].values,
        "elo_opp": df_aug["away_elo_pre"].values,
        "venue": venue, "is_friendly": friendly,
        "self_gf": df_aug["home_gf_form"].values,
        "opp_ga": df_aug["away_ga_form"].values,
    })
    away = pd.DataFrame({
        "elo_self": df_aug["away_elo_pre"].values,
        "elo_opp": df_aug["home_elo_pre"].values,
        "venue": -venue, "is_friendly": friendly,
        "self_gf": df_aug["away_gf_form"].values,
        "opp_ga": df_aug["home_ga_form"].values,
    })
    for fr in (home, away):
        fr["elo_diff"] = fr["elo_self"] - fr["elo_opp"]
    return home, away


def _outcomes(df_aug: pd.DataFrame) -> np.ndarray:
    """Actual class: 0=home win, 1=draw, 2=away win."""
    h = df_aug["home_score"].values
    a = df_aug["away_score"].values
    return np.where(h > a, 0, np.where(h == a, 1, 2))


def _probs_from_model(model: M.PoissonModel, home, away) -> np.ndarray:
    lh = model.predict_lambda(home)
    la = model.predict_lambda(away)
    return np.array([M.outcome_probs(a, b, model.rho) for a, b in zip(lh, la)])


def rps(probs: np.ndarray, actual: np.ndarray) -> float:
    """Mean Ranked Probability Score (order: home, draw, away)."""
    oh = np.eye(3)[actual]
    cp = np.cumsum(probs, axis=1)
    ca = np.cumsum(oh, axis=1)
    return float(np.mean(np.sum((cp - ca) ** 2, axis=1) / 2.0))


def log_loss(probs: np.ndarray, actual: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(actual)), actual], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def accuracy(probs: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == actual))


def _split(bundle: dict, cutoff: str, test_end: str):
    """Elo (with tuned parameters) + form, split into train/test over time."""
    df_elo, _, _ = E.compute_elo(bundle["matches"],
                                 home_adv=M.BEST_PARAMS["home_adv"],
                                 k_scale=M.BEST_PARAMS["k_scale"])
    played = df_elo[df_elo["home_score"].notna() & df_elo["away_score"].notna()].copy()
    played_aug, _ = F.add_form(played)
    played_aug = played_aug.dropna(
        subset=["home_gf_form", "away_gf_form", "home_ga_form", "away_ga_form"]
    )
    cut, end = pd.Timestamp(cutoff), pd.Timestamp(test_end)
    train = played_aug[played_aug["date"] < cut]
    test = played_aug[(played_aug["date"] >= cut) & (played_aug["date"] <= end)]
    return train, test


def evaluate(bundle: dict, cutoff: str = "2022-01-01",
             test_end: str = "2026-06-10") -> pd.DataFrame:
    train, test = _split(bundle, cutoff, test_end)
    hl = M.BEST_PARAMS["halflife"]
    Xtr, ytr, wtr = F.build_training(train, halflife=hl)
    actual = _outcomes(test)
    home_te, away_te = _match_frames(test)

    # --- full model -------------------------------------------------------- #
    full = M.fit(Xtr, ytr, wtr)
    M.fit_rho(full, train)
    p_full = _probs_from_model(full, home_te, away_te)

    # --- Elo-only baseline (Elo difference + home) ------------------------- #
    Xtr_elo = pd.DataFrame({"elo_diff": Xtr["elo_self"] - Xtr["elo_opp"],
                            "venue": Xtr["venue"]})
    elo = M.fit(Xtr_elo, ytr, wtr, features=["elo_diff", "venue"],
                continuous=["elo_diff"])
    # rho for the baseline: uses the same routine but with its own features
    _fit_rho_generic(elo, train)
    p_elo = _probs_from_model(elo, home_te, away_te)

    # --- naive baseline (training base rate) ------------------------------- #
    base = np.bincount(_outcomes(train), minlength=3) / len(train)
    p_naive = np.tile(base, (len(test), 1))

    rows = []
    for name, p in [("naive", p_naive), ("elo", p_elo), ("full", p_full)]:
        rows.append({
            "modelo": name,
            "accuracy": accuracy(p, actual),
            "log_loss": log_loss(p, actual),
            "RPS": rps(p, actual),
        })
    res = pd.DataFrame(rows).set_index("modelo")
    res.attrs["n_train"] = len(train)
    res.attrs["n_test"] = len(test)
    res.attrs["base_rates"] = base
    return res


def _fit_rho_generic(model: M.PoissonModel, df_aug: pd.DataFrame):
    """Estimates rho for a model with arbitrary features."""
    from scipy.optimize import minimize_scalar
    home, away = _match_frames(df_aug)
    lh = model.predict_lambda(home)
    la = model.predict_lambda(away)
    hs = df_aug["home_score"].values
    as_ = df_aug["away_score"].values

    def neg_ll(rho):
        tau = M.dc_tau(hs, as_, lh, la, rho)
        return -np.sum(np.log(np.clip(tau, 1e-9, None)))

    model.rho = float(minimize_scalar(neg_ll, bounds=(-0.2, 0.2),
                                      method="bounded").x)


def calibration(bundle: dict, cutoff: str = "2022-01-01",
                test_end: str = "2026-06-10", bins: int = 10) -> pd.DataFrame:
    """Reliability diagram: predicted probability vs observed frequency.

    Pools the 3 probabilities (W/D/L) of all test matches and compares, per
    probability bin, the prediction with what actually happened. A well-calibrated
    model has prev_media ≈ freq_real in each bin.
    """
    train, test = _split(bundle, cutoff, test_end)
    Xtr, ytr, wtr = F.build_training(train, halflife=M.BEST_PARAMS["halflife"])
    full = M.fit(Xtr, ytr, wtr, l2=M.BEST_PARAMS["l2"])
    M.fit_rho(full, train)
    home, away = _match_frames(test)
    p = _probs_from_model(full, home, away)
    onehot = np.eye(3)[_outcomes(test)]
    pred, hit = p.ravel(), onehot.ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pred, edges) - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        m = idx == k
        if m.sum():
            rows.append({"escalao": f"{edges[k]:.1f}-{edges[k+1]:.1f}",
                         "prev_media": round(pred[m].mean(), 3),
                         "freq_real": round(hit[m].mean(), 3), "n": int(m.sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    b = D.load_all()
    res = evaluate(b)
    print(f"Train: {res.attrs['n_train']} matches (<2022) | "
          f"Test: {res.attrs['n_test']} matches (2022 to Jun/2026)")
    br = res.attrs["base_rates"]
    print(f"Training base rate -> home {br[0]*100:.1f}%  "
          f"draw {br[1]*100:.1f}%  away {br[2]*100:.1f}%\n")
    print((res * 1).round(4).to_string())
    print("\n(accuracy: higher = better | log_loss and RPS: lower = better)")
    print("\n=== Calibration (predicted vs actual) ===")
    print(calibration(b).to_string(index=False))
