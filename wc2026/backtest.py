"""Backtesting: would the model have been right about the 2018 and 2022 World Cups?

Two evaluations, both training ONLY on data prior to the tournament:

1. match_skill      — predicts all 64 actual World Cup matches and measures skill
                      (accuracy / RPS / log-loss) vs the naive base rate.
2. champion_forecast — freezes the pre-tournament state, simulates the World Cup N
                      times (8 groups, standard 32-team bracket) and checks where
                      the actual champion ranked among the favorites.

Note: a standard 32-team bracket is used and host advantage is ignored
(Russia 2018 / Qatar 2022 were not contenders), for simplicity.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import data as D
from . import elo as E
from . import features as F
from . import model as M
from . import validate as V

BEST = M.BEST_PARAMS


# --------------------------------------------------------------------------- #
def _wc_matches(matches, year):
    return matches[(matches.tournament == "FIFA World Cup")
                   & (matches.date.dt.year == year)].sort_values("date")


def actual_champion(matches, shootouts, year):
    wc = _wc_matches(matches, year)
    fin = wc.iloc[-1]
    if fin.home_score > fin.away_score:
        return fin.home_team
    if fin.away_score > fin.home_score:
        return fin.away_team
    # tied final -> decided on penalties
    s = shootouts[(shootouts.date == fin.date)
                  & (shootouts.home_team == fin.home_team)]
    return s.iloc[0].winner if len(s) else fin.home_team


def reconstruct_groups8(matches, year):
    wc = _wc_matches(matches, year).head(48)  # 48 group-stage matches
    adj = defaultdict(set)
    for m in wc.itertuples(index=False):
        adj[m.home_team].add(m.away_team)
        adj[m.away_team].add(m.home_team)
    seen, comps = set(), []
    for t in adj:
        if t in seen:
            continue
        st, c = [t], []
        while st:
            x = st.pop()
            if x in seen:
                continue
            seen.add(x); c.append(x); st += list(adj[x] - seen)
        comps.append(sorted(c))
    comps.sort(key=lambda c: c[0])
    return {chr(ord("A") + i): c for i, c in enumerate(comps)}


# --------------------------------------------------------------------------- #
def match_skill(bundle, year):
    """Model skill on the 64 actual matches (trained only on pre-tournament data)."""
    df_elo, _, _ = E.compute_elo(bundle["matches"], home_adv=BEST["home_adv"],
                                 k_scale=BEST["k_scale"])
    played = df_elo[df_elo["home_score"].notna() & df_elo["away_score"].notna()].copy()
    played_aug, _ = F.add_form(played)
    played_aug = played_aug.dropna(subset=["home_gf_form", "away_gf_form",
                                           "home_ga_form", "away_ga_form"])
    wc = played_aug[(played_aug.tournament == "FIFA World Cup")
                    & (played_aug.date.dt.year == year)]
    cutoff = wc.date.min()
    train = played_aug[played_aug.date < cutoff]

    Xtr, ytr, wtr = F.build_training(train, halflife=BEST["halflife"])
    mdl = M.fit(Xtr, ytr, wtr, l2=BEST["l2"])
    M.fit_rho(mdl, train)
    home, away = V._match_frames(wc)
    p = V._probs_from_model(mdl, home, away)
    actual = V._outcomes(wc)
    base = np.bincount(V._outcomes(train), minlength=3) / len(train)
    p_naive = np.tile(base, (len(wc), 1))
    return {
        "year": year, "n": len(wc),
        "acc": V.accuracy(p, actual), "rps": V.rps(p, actual),
        "logloss": V.log_loss(p, actual),
        "acc_naive": V.accuracy(p_naive, actual), "rps_naive": V.rps(p_naive, actual),
    }


# --------------------------------------------------------------------------- #
def _pair_lambdas(teams, model, state, default):
    n = len(teams)
    LH = np.zeros((n, n)); LA = np.zeros((n, n))
    for i, a in enumerate(teams):
        for j, b in enumerate(teams):
            if i != j:
                LH[i, j], LA[i, j] = model.lambdas_for(state, default, a, b, True)
    return LH, LA


def champion_forecast(bundle, year, n_sims=10000, seed=42):
    """Pre-tournament champion forecast (8 groups, standard 32-team bracket)."""
    matches = bundle["matches"]
    cutoff = _wc_matches(matches, year).date.min()
    pre = M.train_full({"matches": matches[matches.date < cutoff]})
    groups8 = reconstruct_groups8(matches, year)

    rng = np.random.default_rng(seed)
    teams = sorted({t for ts in groups8.values() for t in ts})
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    elo_arr = np.array([pre["state"].get(t, pre["default"])["elo"] for t in teams])
    LH, LA = _pair_lambdas(teams, pre["model"], pre["state"], pre["default"])
    b1 = pre["shootout"]["b1"]
    N = n_sims

    def play(A, B):
        ga = rng.poisson(LH[A, B]); gb = rng.poisson(LA[A, B])
        winA = ga > gb; tie = ga == gb
        if tie.any():
            idx = np.where(tie)[0]
            a2 = ga[idx] + rng.poisson(LH[A[idx], B[idx]] / 3)
            b2 = gb[idx] + rng.poisson(LA[A[idx], B[idx]] / 3)
            wt = a2 > b2; still = a2 == b2
            pa = 1 / (1 + np.exp(-b1 * (elo_arr[A[idx]] - elo_arr[B[idx]])))
            wt = np.where(still, rng.random(idx.size) < pa, wt)
            winA[idx] = wt
        return np.where(winA, A, B)

    # group stage (round-robin) -> 1st and 2nd
    win, ru = {}, {}
    for L, ts in groups8.items():
        gti = np.array([ti[t] for t in ts])
        pts = np.zeros((4, N)); gd = np.zeros((4, N)); gf = np.zeros((4, N))
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = gti[i], gti[j]
                ga = rng.poisson(np.full(N, LH[a, b]))
                gb = rng.poisson(np.full(N, LA[a, b]))
                hw = ga > gb; dr = ga == gb
                pts[i] += 3 * hw + dr; pts[j] += 3 * (gb > ga) + dr
                gd[i] += ga - gb; gd[j] += gb - ga
                gf[i] += ga; gf[j] += gb
        key = pts * 1e6 + gd * 1e3 + gf + rng.random((4, N)) * 0.1
        order = np.argsort(-key, axis=0)
        win[L] = gti[order[0]]; ru[L] = gti[order[1]]

    # standard 32-team bracket
    L = list(groups8.keys())  # A..H
    r16 = [(win[L[0]], ru[L[1]]), (win[L[2]], ru[L[3]]),
           (win[L[4]], ru[L[5]]), (win[L[6]], ru[L[7]]),
           (win[L[1]], ru[L[0]]), (win[L[3]], ru[L[2]]),
           (win[L[5]], ru[L[4]]), (win[L[7]], ru[L[6]])]
    w16 = [play(a, b) for a, b in r16]
    qf = [play(w16[0], w16[1]), play(w16[4], w16[5]),
          play(w16[2], w16[3]), play(w16[6], w16[7])]
    sf = [play(qf[0], qf[1]), play(qf[2], qf[3])]
    champ_idx = play(sf[0], sf[1])

    champ = np.zeros(n)
    np.add.at(champ, champ_idx, 1)
    tab = pd.DataFrame({"team": teams, "p_champion": champ / N}) \
        .sort_values("p_champion", ascending=False).reset_index(drop=True)
    return tab


def run(bundle, year, n_sims=10000):
    sk = match_skill(bundle, year)
    champ_real = actual_champion(bundle["matches"], D_shootouts(), year)
    fc = champion_forecast(bundle, year, n_sims)
    rank = int(fc.index[fc.team == champ_real][0]) + 1
    p = float(fc[fc.team == champ_real]["p_champion"].iloc[0])
    print(f"=== World Cup {year} (actual champion: {champ_real}) ===")
    print(f"Skill over {sk['n']} matches: accuracy {sk['acc']*100:.0f}% "
          f"(naive {sk['acc_naive']*100:.0f}%) | RPS {sk['rps']:.3f} "
          f"(naive {sk['rps_naive']:.3f})")
    print("Model's top-5 pre-tournament favorites:")
    for _, r in fc.head(5).iterrows():
        star = "  <-- CHAMPION" if r.team == champ_real else ""
        print(f"  {r.team:<14} {r.p_champion*100:4.1f}%{star}")
    print(f"The actual champion ({champ_real}) was the #{rank} favorite "
          f"({p*100:.1f}%).\n")


def summary(bundle, year, n_sims=10000) -> dict:
    """Summary of one year's backtest, ready for the dashboard."""
    sk = match_skill(bundle, year)
    champ_real = actual_champion(bundle["matches"], D_shootouts(), year)
    fc = champion_forecast(bundle, year, n_sims)
    rank = int(fc.index[fc.team == champ_real][0]) + 1
    p = float(fc[fc.team == champ_real]["p_champion"].iloc[0])
    return {
        "year": year, "n": sk["n"], "acc": sk["acc"], "rps": sk["rps"],
        "acc_naive": sk["acc_naive"], "rps_naive": sk["rps_naive"],
        "champ_real": champ_real, "champ_rank": rank, "champ_p": p,
        "top5": [{"team": r.team, "p": float(r.p_champion)}
                 for _, r in fc.head(5).iterrows()],
    }


def D_shootouts():
    from . import penalties as P
    return P.load_shootouts()


if __name__ == "__main__":
    b = D.load_all()
    for yr in (2018, 2022):
        run(b, yr)
