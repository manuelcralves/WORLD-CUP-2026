"""Feature engineering (without information leakage).

Each match is converted into TWO rows — one from each team's perspective —
where the target is the goals scored by that team. The features describe
what was known BEFORE the match:

    elo_self     : pre-match Elo of the team itself
    elo_opp      : pre-match Elo of the opponent
    venue        : +1 at home, -1 away, 0 on neutral ground
    is_friendly  : 1 if it is a friendly (different effort/intensity)
    self_gf      : recent attacking form (moving average of goals scored)
    opp_ga       : opponent's recent defensive form (goals conceded)

Each row has a weight = tournament importance x recency decay, so that the
model reflects above all modern, competitive football.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import elo as E

FEATURES = ["elo_self", "elo_opp", "venue", "is_friendly", "self_gf", "opp_ga"]
CONTINUOUS = ["elo_self", "elo_opp", "self_gf", "opp_ga"]  # to be normalized

FORM_SPAN = 10          # no. of matches in the exponential moving average of form
RECENCY_HALFLIFE = 8.0  # years: older matches weigh less
REF_YEAR = 2026


# --------------------------------------------------------------------------- #
# Recent form (exponential moving average of goals, pre-match)
# --------------------------------------------------------------------------- #
def _long_table(played: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (match, team) with goals scored/conceded."""
    n = len(played)
    midx = np.arange(n)
    home = pd.DataFrame({
        "midx": midx, "date": played["date"].values,
        "team": played["home_team"].values,
        "gf": played["home_score"].values, "ga": played["away_score"].values,
        "is_home": 1,
    })
    away = pd.DataFrame({
        "midx": midx, "date": played["date"].values,
        "team": played["away_team"].values,
        "gf": played["away_score"].values, "ga": played["home_score"].values,
        "is_home": 0,
    })
    return pd.concat([home, away], ignore_index=True)


def add_form(played: pd.DataFrame):
    """Add form columns to each match and also return the long table.

    `home_gf_form`/`home_ga_form`/`away_gf_form`/`away_ga_form` are the
    exponential moving averages BEFORE the match (via shift(1)), so no leakage.
    """
    played = played.reset_index(drop=True)
    long = _long_table(played).sort_values(
        ["team", "date", "midx"], kind="stable"
    ).reset_index(drop=True)

    grp = long.groupby("team", sort=False)
    long["gf_form"] = grp["gf"].transform(
        lambda s: s.shift(1).ewm(span=FORM_SPAN, min_periods=1).mean()
    )
    long["ga_form"] = grp["ga"].transform(
        lambda s: s.shift(1).ewm(span=FORM_SPAN, min_periods=1).mean()
    )

    home = long[long["is_home"] == 1].set_index("midx")
    away = long[long["is_home"] == 0].set_index("midx")
    out = played.copy()
    out["home_gf_form"] = out.index.map(home["gf_form"])
    out["home_ga_form"] = out.index.map(home["ga_form"])
    out["away_gf_form"] = out.index.map(away["gf_form"])
    out["away_ga_form"] = out.index.map(away["ga_form"])
    return out, long


def current_form(long: pd.DataFrame) -> pd.DataFrame:
    """"Current" form of each team (EWMA including the last match played)."""
    def last(s):
        return s.ewm(span=FORM_SPAN, min_periods=1).mean().iloc[-1]

    g = long.sort_values(["team", "date", "midx"], kind="stable").groupby("team")
    return pd.DataFrame({
        "gf_form": g["gf"].apply(last),
        "ga_form": g["ga"].apply(last),
    })


# --------------------------------------------------------------------------- #
# Training matrix (stacked across both perspectives)
# --------------------------------------------------------------------------- #
def sample_weight(tournament: pd.Series, year: np.ndarray,
                  halflife: float = RECENCY_HALFLIFE,
                  ref_year: int = REF_YEAR) -> np.ndarray:
    imp = tournament.map(E.tournament_weight).values / 60.0
    recency = 0.5 ** ((ref_year - year) / halflife)
    return imp * recency


def build_training(played_aug: pd.DataFrame, halflife: float = RECENCY_HALFLIFE,
                   ref_year: int = REF_YEAR):
    """Build (X, y, w) from matches with Elo and form already attached."""
    df = played_aug
    venue = np.where(df["neutral"].values, 0.0, 1.0)
    friendly = df["tournament"].str.lower().eq("friendly").astype(float).values
    year = df["date"].dt.year.values
    w = sample_weight(df["tournament"], year, halflife, ref_year)

    home = pd.DataFrame({
        "elo_self": df["home_elo_pre"].values,
        "elo_opp": df["away_elo_pre"].values,
        "venue": venue,
        "is_friendly": friendly,
        "self_gf": df["home_gf_form"].values,
        "opp_ga": df["away_ga_form"].values,
    })
    away = pd.DataFrame({
        "elo_self": df["away_elo_pre"].values,
        "elo_opp": df["home_elo_pre"].values,
        "venue": -venue,
        "is_friendly": friendly,
        "self_gf": df["away_gf_form"].values,
        "opp_ga": df["home_ga_form"].values,
    })
    X = pd.concat([home[FEATURES], away[FEATURES]], ignore_index=True)
    y = np.concatenate([df["home_score"].values, df["away_score"].values]).astype(float)
    w = np.concatenate([w, w])

    ok = X.notna().all(axis=1).values & ~np.isnan(y)
    return X[ok].reset_index(drop=True), y[ok], w[ok]


# --------------------------------------------------------------------------- #
# "Current" state to feed the simulation
# --------------------------------------------------------------------------- #
def current_state(ratings: dict, form: pd.DataFrame) -> dict:
    """state[team] = {elo, gf_form, ga_form} with the best current knowledge."""
    state = {}
    for team, elo in ratings.items():
        gf = form["gf_form"].get(team, np.nan)
        ga = form["ga_form"].get(team, np.nan)
        state[team] = {"elo": elo, "gf_form": gf, "ga_form": ga}
    return state


def default_state(played: pd.DataFrame) -> dict:
    """Fallback values for a team with no history."""
    avg = float(np.nanmean(np.r_[played["home_score"], played["away_score"]]))
    return {"elo": E.BASE_RATING, "gf_form": avg, "ga_form": avg}


def fixture_rows(state: dict, default: dict, home: str, away: str, neutral: bool):
    """Return (home_row, away_row) of features for a match to simulate."""
    sh = state.get(home, default)
    sa = state.get(away, default)
    venue = 0.0 if neutral else 1.0
    row_home = {
        "elo_self": sh["elo"], "elo_opp": sa["elo"], "venue": venue,
        "is_friendly": 0.0, "self_gf": sh["gf_form"], "opp_ga": sa["ga_form"],
    }
    row_away = {
        "elo_self": sa["elo"], "elo_opp": sh["elo"], "venue": -venue,
        "is_friendly": 0.0, "self_gf": sa["gf_form"], "opp_ga": sh["ga_form"],
    }
    return row_home, row_away
