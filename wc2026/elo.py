"""Elo ratings for national teams (World Football Elo Ratings style).

Makes a chronological pass over all matches played and keeps a rating per
team. Each team's PRE-match rating is the model's strongest feature (and has
no information leakage: it only uses the past).

Formula:
    E_home = 1 / (1 + 10 ** (-(R_home + home_advantage - R_away) / 400))
    R' = R + K * G * (S - E)
where K depends on the importance of the tournament and G scales the
adjustment according to the goal difference (blowouts move the rating more).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Initial rating for a team appearing for the first time.
BASE_RATING = 1500.0
# Home-field advantage, in Elo points (applied only if the venue is not neutral).
HOME_ADVANTAGE = 70.0

# Weight (base K) by tournament importance. Matched by substring in the name.
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 60,
    "Copa América": 50,
    "Copa America": 50,
    "UEFA Euro": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "Confederations Cup": 50,
    "qualification": 40,
    "Nations League": 40,
    "Finals": 40,
    "Friendly": 20,
}
DEFAULT_WEIGHT = 30  # smaller tournaments / not classified above


def tournament_weight(name: str) -> float:
    for key, w in TOURNAMENT_WEIGHT.items():
        if key.lower() in name.lower():
            return w
    return DEFAULT_WEIGHT


def expected_score(r_a: float, r_b: float) -> float:
    """Elo probability of A scoring (1=win, 0.5=draw) against B."""
    return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b) / 400.0))


def goal_diff_multiplier(gd: int) -> float:
    """G multiplier from the World Football Elo formula."""
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def compute_elo(
    matches: pd.DataFrame,
    base: float = BASE_RATING,
    home_adv: float = HOME_ADVANTAGE,
    k_scale: float = 1.0,
    keep_history: bool = False,
):
    """Compute Elo chronologically.

    Parameters
    ----------
    matches : all matches (played or not), already sorted by date. Matches with
        no result get the current pre-match rating but do NOT update the Elo.

    Returns
    -------
    df : copy of `matches` with columns `home_elo_pre`, `away_elo_pre`.
    ratings : dict team -> final rating ("current" state).
    history : DataFrame (date, team, rating) if keep_history, else None.
    """
    ratings: dict[str, float] = {}
    home_pre = np.empty(len(matches))
    away_pre = np.empty(len(matches))
    history: list[tuple] = [] if keep_history else None

    for i, m in enumerate(matches.itertuples(index=False)):
        rh = ratings.get(m.home_team, base)
        ra = ratings.get(m.away_team, base)
        home_pre[i] = rh
        away_pre[i] = ra

        # Matches not yet played: record the pre-match rating but don't update.
        if pd.isna(m.home_score) or pd.isna(m.away_score):
            continue

        adv = 0.0 if m.neutral else home_adv
        exp_home = expected_score(rh + adv, ra)

        hs, as_ = int(m.home_score), int(m.away_score)
        if hs > as_:
            s_home = 1.0
        elif hs < as_:
            s_home = 0.0
        else:
            s_home = 0.5

        k = k_scale * tournament_weight(m.tournament) * goal_diff_multiplier(hs - as_)
        delta = k * (s_home - exp_home)
        ratings[m.home_team] = rh + delta
        ratings[m.away_team] = ra - delta  # zero-sum game

        if keep_history:
            history.append((m.date, m.home_team, ratings[m.home_team]))
            history.append((m.date, m.away_team, ratings[m.away_team]))

    df = matches.copy()
    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre

    hist_df = (
        pd.DataFrame(history, columns=["date", "team", "rating"])
        if keep_history
        else None
    )
    return df, ratings, hist_df


def ranking(ratings: dict[str, float], top: int | None = None) -> pd.DataFrame:
    s = pd.Series(ratings).sort_values(ascending=False)
    if top:
        s = s.head(top)
    return s.rename_axis("team").reset_index(name="elo")


if __name__ == "__main__":
    from . import data as D

    d = D.load_all()
    df, ratings, _ = compute_elo(d["matches"])
    print("Top 20 teams by Elo (current state):")
    print(ranking(ratings, 20).to_string(index=False))
