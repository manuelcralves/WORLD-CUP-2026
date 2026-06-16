"""Goals model: Poisson regression (Dixon-Coles) by maximum likelihood.

Trains two "sides" of the same Poisson model (the same equation serves both the
home team and the away team, thanks to the stacked representation of the
features) and estimates the Dixon-Coles correction for low scores (0-0, 1-0,
0-1, 1-1), where pure independence between the two teams' goals breaks down.

From each team's expected goals (lambda) we obtain the grid of probabilities
for each exact score and, from there, the probability of a win, draw or loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from . import features as F


@dataclass
class PoissonModel:
    beta: np.ndarray          # coefficients (includes intercept)
    mu: pd.Series             # means of the continuous features (normalization)
    sd: pd.Series             # standard deviations of the continuous features
    rho: float                # Dixon-Coles dependence parameter
    features: list            # feature order
    continuous: list          # subset of features to normalize

    # ---- lambda prediction (expected goals) ------------------------------ #
    def _design(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.features].copy()
        for c in self.continuous:
            Z[c] = (Z[c] - self.mu[c]) / self.sd[c]
        return np.column_stack([np.ones(len(Z)), Z.values])

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self._design(X) @ self.beta)

    def lambdas_for(self, state, default, home, away, neutral):
        """Expected goals (lambda_home, lambda_away) for a specific match."""
        rh, ra = F.fixture_rows(state, default, home, away, neutral)
        X = pd.DataFrame([rh, ra])
        lam = self.predict_lambda(X)
        return float(lam[0]), float(lam[1])

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": ["intercept"] + self.features,
                             "beta": self.beta})


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def fit(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, l2: float = 1e-3,
        features: list | None = None, continuous: list | None = None) -> PoissonModel:
    """Fits the Poisson by weighted maximum likelihood (with L2 regularization)."""
    features = list(features) if features is not None else list(F.FEATURES)
    continuous = list(continuous) if continuous is not None else list(F.CONTINUOUS)
    mu = X[continuous].mean()
    sd = X[continuous].std().replace(0, 1.0)
    model = PoissonModel(beta=None, mu=mu, sd=sd, rho=0.0,
                         features=features, continuous=continuous)
    M = model._design(X)
    w = w / w.mean()  # normalize the weights

    def nll(b):
        eta = M @ b
        eta = np.clip(eta, -10, 10)  # numerical stability
        lam = np.exp(eta)
        val = np.sum(w * (lam - y * eta)) + l2 * np.sum(b[1:] ** 2)
        grad = M.T @ (w * (lam - y)) + 2 * l2 * np.r_[0.0, b[1:]]
        return val, grad

    b0 = np.zeros(M.shape[1])
    b0[0] = np.log(max(np.average(y, weights=w), 0.1))
    res = minimize(nll, b0, jac=True, method="L-BFGS-B")
    model.beta = res.x
    return model


def fit_rho(model: PoissonModel, played_aug: pd.DataFrame) -> float:
    """Estimates Dixon-Coles rho by maximizing the likelihood of the real matches."""
    lam_h, lam_a = _match_lambdas(model, played_aug)
    hs = played_aug["home_score"].values
    as_ = played_aug["away_score"].values
    ok = ~(np.isnan(lam_h) | np.isnan(lam_a) | np.isnan(hs) | np.isnan(as_))
    hs, as_, lam_h, lam_a = hs[ok], as_[ok], lam_h[ok], lam_a[ok]

    def neg_ll(rho):
        tau = dc_tau(hs, as_, lam_h, lam_a, rho)
        return -np.sum(np.log(np.clip(tau, 1e-9, None)))

    res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
    model.rho = float(res.x)
    return model.rho


def _match_lambdas(model: PoissonModel, df_aug: pd.DataFrame):
    """home and away lambda for each match (uses features already attached)."""
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
    return model.predict_lambda(home), model.predict_lambda(away)


# --------------------------------------------------------------------------- #
# Dixon-Coles and score grid
# --------------------------------------------------------------------------- #
def dc_tau(h, a, lh, la, rho):
    """Dixon-Coles correction factor for the low scores."""
    h = np.asarray(h); a = np.asarray(a)
    lh = np.asarray(lh, float); la = np.asarray(la, float)
    tau = np.ones(np.broadcast(h, a, lh, la).shape, float)
    m = (h == 0) & (a == 0); tau[m] = 1 - lh[m] * la[m] * rho
    m = (h == 0) & (a == 1); tau[m] = 1 + lh[m] * rho
    m = (h == 1) & (a == 0); tau[m] = 1 + la[m] * rho
    m = (h == 1) & (a == 1); tau[m] = 1 - rho
    return tau


def score_grid(lh: float, la: float, rho: float, maxg: int = 10) -> np.ndarray:
    """Matrix P[i, j] = probability that the score is i-j (home-away)."""
    i = np.arange(maxg + 1)
    ph = poisson.pmf(i, lh)
    pa = poisson.pmf(i, la)
    G = np.outer(ph, pa)
    # Dixon-Coles correction in the low-scores corner
    G[0, 0] *= 1 - lh * la * rho
    G[0, 1] *= 1 + lh * rho
    G[1, 0] *= 1 + la * rho
    G[1, 1] *= 1 - rho
    G = np.clip(G, 0, None)
    return G / G.sum()


def outcome_probs(lh: float, la: float, rho: float = 0.0, maxg: int = 10):
    """(P_home_win, P_draw, P_away_win)."""
    G = score_grid(lh, la, rho, maxg)
    home = np.tril(G, -1).sum()
    draw = np.trace(G)
    away = np.triu(G, 1).sum()
    return float(home), float(draw), float(away)


# --------------------------------------------------------------------------- #
# Full training pipeline (data -> model + simulation state)
# --------------------------------------------------------------------------- #
# Optimal hyperparameters (found by wc2026.tune; see README).
BEST_PARAMS = {"home_adv": 70.0, "k_scale": 0.8, "halflife": 5.0, "l2": 1e-3}


def train_full(bundle: dict, home_adv=BEST_PARAMS["home_adv"],
               k_scale=BEST_PARAMS["k_scale"], halflife=BEST_PARAMS["halflife"],
               l2=BEST_PARAMS["l2"]):
    """Takes the result of data.load_all() and returns model + current state.

    Uses the tuned hyperparameters by default. Returns a dict with: model, state,
    default, ratings, played_aug, df_elo, shootout (penalty model), params.
    """
    from . import elo as E
    from . import penalties as P

    matches = bundle["matches"]
    df_elo, ratings, _ = E.compute_elo(matches, home_adv=home_adv, k_scale=k_scale)
    # restrict to the matches played and attach form
    played = df_elo[df_elo["home_score"].notna() & df_elo["away_score"].notna()].copy()
    played_aug, long = F.add_form(played)

    X, y, w = F.build_training(played_aug, halflife=halflife)
    model = fit(X, y, w, l2=l2)
    fit_rho(model, played_aug)

    state = F.current_state(ratings, F.current_form(long))
    default = F.default_state(played)
    shootout = P.fit_shootout_model(P.load_shootouts(), df_elo)
    return {
        "model": model, "state": state, "default": default,
        "ratings": ratings, "played_aug": played_aug, "df_elo": df_elo,
        "shootout": shootout,
        "params": {"home_adv": home_adv, "k_scale": k_scale,
                   "halflife": halflife, "l2": l2},
        "X": X, "y": y, "w": w,
    }


if __name__ == "__main__":
    from . import data as D

    b = D.load_all()
    out = train_full(b)
    model, state, default = out["model"], out["state"], out["default"]
    print("Poisson model coefficients:")
    print(model.coef_table().to_string(index=False))
    print(f"\nrho (Dixon-Coles): {model.rho:.4f}\n")

    def show(home, away, neutral):
        lh, la = model.lambdas_for(state, default, home, away, neutral)
        ph, pd_, pa = outcome_probs(lh, la, model.rho)
        loc = "neutral" if neutral else f"{home} at home"
        print(f"{home} vs {away} ({loc}): "
              f"xG {lh:.2f}-{la:.2f} | "
              f"W {ph*100:4.1f}% D {pd_*100:4.1f}% L {pa*100:4.1f}%")

    print("Sanity check — some matches:")
    show("Germany", "Japan", True)
    show("Spain", "Cape Verde", True)
    show("Brazil", "Scotland", True)
    show("United States", "Paraguay", False)  # host at home
    show("Argentina", "France", True)
