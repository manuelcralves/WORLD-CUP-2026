"""Mega backtest: model skill across ALL major international tournaments.

Trains the Poisson model on data before a cutoff, then predicts every match of
the big finals tournaments since then — World Cups, Euros, Copa América, Nations
League, Asian/African Cups, Gold Cups — and aggregates the skill (accuracy /
RPS) vs the naive base rate. Hundreds of real tournament matches.

The Elo and form features are chronological (pre-match, no leakage); the Poisson
coefficients are stable over time, so a single fit on pre-cutoff data is a fair,
honest test of out-of-sample skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import elo as E
from . import features as F
from . import model as M
from . import validate as V

BEST = M.BEST_PARAMS

# Finals tournaments only (qualification excluded). Copa América has two spellings.
MAJORS = {
    "FIFA World Cup", "UEFA Euro", "Copa América", "Copa America",
    "African Cup of Nations", "AFC Asian Cup", "Gold Cup", "UEFA Nations League",
    "Confederations Cup", "FIFA Confederations Cup", "CONCACAF Championship",
}
DISPLAY = {"Copa America": "Copa América", "AFC Asian Cup": "AFC Asian Cup",
           "African Cup of Nations": "Africa Cup of Nations"}


def _fit_until(played_aug, before):
    """Fit the model on every match strictly before `before` (walk-forward)."""
    train = played_aug[played_aug["date"] < before]
    model = M.fit(*F.build_training(train, halflife=BEST["halflife"]), l2=BEST["l2"])
    M.fit_rho(model, train)
    base = np.bincount(V._outcomes(train), minlength=3) / len(train)
    return model, base


def run_all(bundle, start_year=2002, min_n=20, min_train=3000) -> dict:
    """Honest walk-forward test: for **each** finals tournament since
    `start_year`, the model is retrained on everything that happened *before
    that tournament kicked off* and then asked to predict it blind — exactly
    how you would have used it in real time. Skill is aggregated over all the
    editions and broken down by competition.
    """
    df_elo, _, _ = E.compute_elo(bundle["matches"], home_adv=BEST["home_adv"],
                                 k_scale=BEST["k_scale"])
    played = df_elo[df_elo["home_score"].notna() & df_elo["away_score"].notna()].copy()
    played_aug, _ = F.add_form(played)
    played_aug = played_aug.dropna(subset=["home_gf_form", "away_gf_form",
                                           "home_ga_form", "away_ga_form"])
    played_aug = played_aug.sort_values("date").reset_index(drop=True)
    played_aug["comp"] = played_aug["tournament"].replace(DISPLAY)

    test = played_aug[(played_aug["tournament"].isin(MAJORS))
                      & (played_aug["date"].dt.year >= start_year)]
    P, PN, A, C = [], [], [], []
    n_editions = 0
    cache = {}  # reuse a fit across editions that start in the same month
    for (comp, _year), sub in test.groupby(["comp", test["date"].dt.year]):
        start = sub["date"].min()
        if len(played_aug[played_aug["date"] < start]) < min_train:
            continue
        key = (start.year, start.month)
        if key not in cache:
            cache[key] = _fit_until(played_aug, start)
        model, base = cache[key]
        home, away = V._match_frames(sub)
        P.append(V._probs_from_model(model, home, away))
        PN.append(np.tile(base, (len(sub), 1)))
        A.append(V._outcomes(sub))
        C += [comp] * len(sub)
        n_editions += 1

    P = np.vstack(P); PN = np.vstack(PN)
    A = np.concatenate(A); C = np.array(C)

    def metrics(mask):
        return {"n": int(mask.sum()),
                "acc": V.accuracy(P[mask], A[mask]), "rps": V.rps(P[mask], A[mask]),
                "acc_naive": V.accuracy(PN[mask], A[mask]),
                "rps_naive": V.rps(PN[mask], A[mask])}

    overall = metrics(np.ones(len(A), bool))
    overall["n_editions"] = n_editions
    by_comp = []
    for comp in pd.unique(C):
        mask = C == comp
        if mask.sum() >= min_n:
            by_comp.append({"competition": comp, **metrics(mask)})
    by_comp.sort(key=lambda r: -r["n"])
    return {"start_year": start_year, "overall": overall,
            "by_competition": by_comp, "walk_forward": True}


if __name__ == "__main__":
    import time
    b = D.load_all()
    print("Walk-forward backtest across all major tournaments (may take ~30s)...")
    t0 = time.time()
    r = run_all(b)
    o = r["overall"]
    print(f"\nSince {r['start_year']} (walk-forward): {o['n']} matches across "
          f"{o['n_editions']} tournament editions  [{time.time()-t0:.0f}s]")
    print(f"  Model : accuracy {o['acc']*100:.1f}%  RPS {o['rps']:.3f}")
    print(f"  Naive : accuracy {o['acc_naive']*100:.1f}%  RPS {o['rps_naive']:.3f}")
    print("\nBy competition:")
    for c in r["by_competition"]:
        print(f"  {c['competition']:<24} {c['n']:>4} matches · "
              f"acc {c['acc']*100:.0f}% (naive {c['acc_naive']*100:.0f}%) · "
              f"RPS {c['rps']:.3f}")
