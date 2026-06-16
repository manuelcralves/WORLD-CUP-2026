"""Penalty shootout tiebreak model, built from the 677 historical shootouts.

Instead of the "almost 50/50 with a slight Elo lean" guess, we learn from the
data: each shootout is matched to the (drawn) match in the results to obtain the
teams' pre-match Elo, and a logistic regression is fitted as
P(team A wins) = sigmoid(b0 + b1 * (Elo_A - Elo_B)).

Spoiler: penalties are almost a coin toss — Elo barely helps, just as the
studies say. We also measure the (famous) advantage of shooting first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import data as D
from . import elo as E


def load_shootouts(data_dir=D.DATA_DIR) -> pd.DataFrame:
    s = pd.read_csv(f"{data_dir}/shootouts.csv", parse_dates=["date"])
    mapping = D.name_mapping(D.load_former_names(data_dir))
    for c in ("home_team", "away_team", "winner"):
        s[c] = s[c].replace(mapping)
    return s


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(x)), x])

    def nll(b):
        z = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    return minimize(nll, np.zeros(2), method="L-BFGS-B").x


def fit_shootout_model(shootouts: pd.DataFrame, df_elo: pd.DataFrame) -> dict:
    """Fits P(home wins the shootout) ~ pre-match Elo difference."""
    cols = ["date", "home_team", "away_team", "home_elo_pre", "away_elo_pre"]
    merged = shootouts.merge(df_elo[cols], on=["date", "home_team", "away_team"],
                             how="inner")
    elo_diff = (merged["home_elo_pre"] - merged["away_elo_pre"]).values
    home_win = (merged["winner"] == merged["home_team"]).astype(float).values
    b0, b1 = _fit_logistic(elo_diff, home_win)
    return {"b0": float(b0), "b1": float(b1),
            "n": int(len(merged)), "home_win_rate": float(home_win.mean())}


def shootout_prob(params: dict, elo_a: np.ndarray, elo_b: np.ndarray,
                  neutral: bool = True):
    """P(A wins on penalties). On a neutral ground the intercept (b0) is ignored."""
    b0 = 0.0 if neutral else params["b0"]
    z = b0 + params["b1"] * (np.asarray(elo_a) - np.asarray(elo_b))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def first_shooter_advantage(shootouts: pd.DataFrame) -> dict:
    """Does whoever shoots first win more often? (when the data exists)."""
    s = shootouts.dropna(subset=["first_shooter"])
    s = s[s["first_shooter"].astype(str).str.strip() != ""]
    wins_first = (s["first_shooter"] == s["winner"]).mean()
    return {"n": int(len(s)), "win_rate_shooting_first": float(wins_first)}


def team_records(shootouts: pd.DataFrame, min_games: int = 4) -> pd.DataFrame:
    """Record of each national team on penalties (wins-losses)."""
    played, won = {}, {}
    for m in shootouts.itertuples(index=False):
        for t in (m.home_team, m.away_team):
            played[t] = played.get(t, 0) + 1
        won[m.winner] = won.get(m.winner, 0) + 1
    rows = [{"team": t, "shootouts": n, "wins": won.get(t, 0),
             "losses": n - won.get(t, 0), "win_pct": won.get(t, 0) / n}
            for t, n in played.items() if n >= min_games]
    return pd.DataFrame(rows).sort_values(["win_pct", "shootouts"],
                                          ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    b = D.load_all()
    df_elo, _, _ = E.compute_elo(b["matches"])
    s = load_shootouts()
    m = fit_shootout_model(s, df_elo)
    print(f"Shootouts with Elo: {m['n']} | 'home' win rate: "
          f"{m['home_win_rate']*100:.1f}%")
    print(f"Model: P(A wins) = sigmoid({m['b0']:.4f} + {m['b1']:.5f} * Elo_diff)")
    print(f"  E.g.: a +200 Elo favorite wins "
          f"{shootout_prob(m, 200, 0)*100:.1f}% of penalty shootouts (almost 50/50!)")
    fs = first_shooter_advantage(s)
    print(f"\nShooting first: wins {fs['win_rate_shooting_first']*100:.1f}% "
          f"(n={fs['n']})")
    print("\nBest national teams on penalties (min. 4):")
    print(team_records(s).head(8).to_string(index=False))
