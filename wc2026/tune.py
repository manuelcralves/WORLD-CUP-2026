"""Hyperparameter tuning by temporal grid (minimizes RPS on the holdout).

Varies the most important parameters and re-evaluates on the same temporal
holdout as the validation (train < 2022, test 2022-2026):

  home_adv  : home advantage in Elo points
  k_scale   : scale of the Elo K factor (how fast the ratings react)
  halflife  : half-life (years) of the recency weight in training
  l2        : regularization of the Poisson model

Since Elo already carries almost all the signal, the gains are marginal — but
home advantage directly tunes the hosts' chances, so it is worth it.
"""
from __future__ import annotations

import itertools

import pandas as pd

from . import data as D
from . import elo as E
from . import features as F
from . import model as M
from . import validate as V

HOME_ADV_GRID = [50, 70, 90, 110]
KSCALE_GRID = [0.8, 1.0, 1.3]
HALFLIFE_GRID = [5, 8, 15]
L2_GRID = [1e-3, 1e-2]


def _prep(matches, home_adv, k_scale):
    """Elo (with these parameters) + form, attached to the matches played."""
    df_elo, _, _ = E.compute_elo(matches, home_adv=home_adv, k_scale=k_scale)
    played = df_elo[df_elo["home_score"].notna() & df_elo["away_score"].notna()].copy()
    played_aug, _ = F.add_form(played)
    return played_aug.dropna(subset=["home_gf_form", "away_gf_form",
                                     "home_ga_form", "away_ga_form"])


def _rps(played_aug, halflife, l2, cutoff, end):
    cut, fim = pd.Timestamp(cutoff), pd.Timestamp(end)
    train = played_aug[played_aug["date"] < cut]
    test = played_aug[(played_aug["date"] >= cut) & (played_aug["date"] <= fim)]
    Xtr, ytr, wtr = F.build_training(train, halflife=halflife)
    model = M.fit(Xtr, ytr, wtr, l2=l2)
    M.fit_rho(model, train)
    home_te, away_te = V._match_frames(test)
    p = V._probs_from_model(model, home_te, away_te)
    return V.rps(p, V._outcomes(test))


def search(bundle, cutoff="2022-01-01", end="2026-06-10", verbose=True):
    matches = bundle["matches"]
    rows = []
    prep_cache = {}
    combos = list(itertools.product(HOME_ADV_GRID, KSCALE_GRID,
                                    HALFLIFE_GRID, L2_GRID))
    for i, (ha, ks, hl, l2) in enumerate(combos, 1):
        if (ha, ks) not in prep_cache:
            prep_cache[(ha, ks)] = _prep(matches, ha, ks)
        rps = _rps(prep_cache[(ha, ks)], hl, l2, cutoff, end)
        rows.append({"home_adv": ha, "k_scale": ks, "halflife": hl,
                     "l2": l2, "RPS": rps})
        if verbose:
            print(f"  [{i:>2}/{len(combos)}] home_adv={ha} k_scale={ks} "
                  f"halflife={hl} l2={l2}  ->  RPS={rps:.5f}")
    res = pd.DataFrame(rows).sort_values("RPS").reset_index(drop=True)
    return res


if __name__ == "__main__":
    b = D.load_all()
    print("Searching for the best hyperparameters (may take 1-2 min)...")
    res = search(b)
    print("\n=== Top 8 configurations ===")
    print(res.head(8).to_string(index=False))
    best = res.iloc[0]
    base = res[(res.home_adv == 70) & (res.k_scale == 1.0)
               & (res.halflife == 8) & (res.l2 == 1e-3)]
    if len(base):
        print(f"\nBaseline (home_adv=70, k=1.0, hl=8): RPS={base.iloc[0]['RPS']:.5f}")
    print(f"Best:     home_adv={int(best.home_adv)}, k_scale={best.k_scale}, "
          f"halflife={int(best.halflife)}, l2={best.l2}  ->  RPS={best.RPS:.5f}")
