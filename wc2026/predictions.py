"""Rich predictions from the trained model and the simulation.

  - match_report          : most likely score, top-5 scorelines, W/D/L, markets
  - group_stage_predictions: prediction of each remaining group-stage match
  - expected_standings    : expected standings of each group
  - opponents_for         : a team's most likely opponents per round
  - most_likely_bracket   : the "favourites" bracket of the knockout rounds
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import model as M
from .tournament import (HOSTS, LATER, OFFICIAL_GROUPS, R32, _match_thirds)

THIRD_SLOTS = [74, 77, 79, 80, 81, 82, 85, 87]


# --------------------------------------------------------------------------- #
def match_report(trained, home, away, neutral=True, maxg=10) -> dict:
    """Full report of a tie (doubles as an H2H simulator)."""
    model, state, default = trained["model"], trained["state"], trained["default"]
    lh, la = model.lambdas_for(state, default, home, away, neutral)
    G = M.score_grid(lh, la, model.rho, maxg)
    I, J = np.indices(G.shape)

    flat = sorted(((G[i, j], i, j) for i in range(maxg + 1) for j in range(maxg + 1)),
                  reverse=True)
    top = [{"score": f"{i}-{j}", "p": float(p)} for p, i, j in flat[:5]]
    ml = flat[0]
    return {
        "home": home, "away": away, "neutral": neutral,
        "xg_home": float(lh), "xg_away": float(la),
        "p_home": float(np.tril(G, -1).sum()),
        "p_draw": float(np.trace(G)),
        "p_away": float(np.triu(G, 1).sum()),
        "most_likely": f"{ml[1]}-{ml[2]}", "p_most_likely": float(ml[0]),
        "top_scorelines": top,
        "p_over25": float(G[(I + J) >= 3].sum()),
        "p_btts": float(G[(I >= 1) & (J >= 1)].sum()),
    }


def format_match(rep: dict) -> str:
    return (f"{rep['home']} {rep['most_likely']} {rep['away']}  "
            f"(most likely score {rep['p_most_likely']*100:.0f}%) | "
            f"W {rep['p_home']*100:.0f}%  D {rep['p_draw']*100:.0f}%  "
            f"L {rep['p_away']*100:.0f}% | xG {rep['xg_home']:.2f}-{rep['xg_away']:.2f}")


# --------------------------------------------------------------------------- #
def group_stage_predictions(bundle, trained, only_remaining=True) -> pd.DataFrame:
    """Prediction of each group-stage match (by default only the remaining ones)."""
    team_letter = {t: L for L, ts in OFFICIAL_GROUPS.items() for t in ts}
    games = bundle["wc_remaining"] if only_remaining else bundle["wc"]
    rows = []
    for g in games.itertuples(index=False):
        rep = match_report(trained, g.home_team, g.away_team, neutral=bool(g.neutral))
        rows.append({
            "date": pd.Timestamp(g.date).date(),
            "group": team_letter.get(g.home_team, "?"),
            "home": g.home_team, "away": g.away_team,
            "xg_home": round(rep["xg_home"], 2), "xg_away": round(rep["xg_away"], 2),
            "p_home": round(rep["p_home"], 3), "p_draw": round(rep["p_draw"], 3),
            "p_away": round(rep["p_away"], 3),
            "ml_score": rep["most_likely"], "p_ml": round(rep["p_most_likely"], 3),
            "top3": ",".join(f"{s['score']}:{round(s['p'] * 100)}"
                             for s in rep["top_scorelines"][:3]),
        })
    cols = ["date", "group", "home", "away", "xg_home", "xg_away",
            "p_home", "p_draw", "p_away", "ml_score", "p_ml", "top3"]
    df = pd.DataFrame(rows, columns=cols)            # keep the header even with 0 remaining games
    return df if df.empty else df.sort_values(["date", "group"]).reset_index(drop=True)


def match_predictions(bundle, trained, topn=3) -> list:
    """Richer per-match prediction for the dashboard (incl. top-N scorelines)."""
    team_letter = {t: L for L, ts in OFFICIAL_GROUPS.items() for t in ts}
    rows = []
    for g in bundle["wc_remaining"].itertuples(index=False):
        rep = match_report(trained, g.home_team, g.away_team, neutral=bool(g.neutral))
        rows.append({
            "date": str(pd.Timestamp(g.date).date()),
            "group": team_letter.get(g.home_team, "?"),
            "home": g.home_team, "away": g.away_team,
            "xg_home": round(rep["xg_home"], 2), "xg_away": round(rep["xg_away"], 2),
            "p_home": rep["p_home"], "p_draw": rep["p_draw"], "p_away": rep["p_away"],
            "top": [{"score": s["score"], "p": s["p"]}
                    for s in rep["top_scorelines"][:topn]],
        })
    rows.sort(key=lambda r: (r["date"], r["group"]))
    return rows


# --------------------------------------------------------------------------- #
def expected_standings(table: pd.DataFrame) -> pd.DataFrame:
    """Expected standings of each group (sorted by expected points)."""
    out = []
    for L in OFFICIAL_GROUPS:
        g = table[table.group == L].sort_values("exp_points", ascending=False)
        for rank, (_, r) in enumerate(g.iterrows(), 1):
            out.append({
                "group": L, "pos": rank, "team": r["team"],
                "exp_points": r["exp_points"],
                "p_1st": round(r["p_1st"], 3), "p_2nd": round(r["p_2nd"], 3),
                "p_advance": round(r["p_ko"], 3),
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
def opponents_for(table: pd.DataFrame, team: str, topn=4) -> dict:
    """Most likely knockout opponents of `team`, round by round."""
    if "opp_matrix" not in table.attrs:
        return {}
    teams = table.attrs["teams"]
    if team not in teams:
        return {}
    idx = teams.index(team)
    n = table.attrs["n_sims"]
    opp_mat = table.attrs["opp_matrix"]
    reach = table.attrs["reach_counts"]
    labels = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-finals",
              "SF": "Semi-finals", "F": "Final"}
    res = {}
    for rnd, lab in labels.items():
        played = reach[rnd][team]
        if played == 0:
            continue
        row = pd.Series(opp_mat[rnd][idx], index=teams)
        s = row[row > 0].sort_values(ascending=False).head(topn)
        res[lab] = {
            "p_reach": round(float(played) / n, 3),
            "opponents": [{"team": t, "p_cond": round(float(c) / played, 3)}
                          for t, c in s.items()],
        }
    return res


# --------------------------------------------------------------------------- #
def most_likely_bracket(table: pd.DataFrame, trained: dict) -> list:
    """"Favourites" bracket: fills each slot with the modal team and resolves
    each tie in favour of the favourite (probability of advancing).

    Returns a list of rounds, each with its ties.
    """
    by_group = {L: table[table.group == L] for L in OFFICIAL_GROUPS}
    winner, runnerup, third = {}, {}, {}
    for L, g in by_group.items():
        w = g.sort_values("p_1st", ascending=False)["team"].iloc[0]
        ru = g.sort_values("p_2nd", ascending=False)["team"].tolist()
        runnerup[L] = ru[0] if ru[0] != w else ru[1]
        winner[L] = w
        third[L] = g.sort_values("p_3rd", ascending=False)["team"].iloc[0]

    # 8 best third-placed teams — rank by the sim's qualifying probability (which
    # applies the full FIFA tie-breakers: pts, GD, GF, ...), not raw expected
    # points (which can't break a points tie -> could pick the wrong third).
    strength = {L: float(by_group[L].set_index("team").loc[third[L], "p_ko"])
                for L in OFFICIAL_GROUPS}
    qual = sorted(strength, key=strength.get, reverse=True)[:8]
    assign = _match_thirds(sorted(qual), THIRD_SLOTS)  # {slot: group}
    slot_third = {slot: third[L] for slot, L in assign.items()}

    def team_of(spec):
        typ, key = spec
        return {"W": winner, "RU": runnerup}[typ][key] if typ in ("W", "RU") \
            else slot_third[key]

    def advance(a, b):
        host = (a in HOSTS) ^ (b in HOSTS)
        rep = match_report(trained, a, b, neutral=not host)
        pa = rep["p_home"] + rep["p_draw"] / 2  # includes extra time/penalties ~ 50/50
        return (a, pa) if pa >= 0.5 else (b, 1 - pa)

    win = {}
    bracket = []
    # Round of 32
    r32 = []
    for mno, (sa, sb) in R32.items():
        a, b = team_of(sa), team_of(sb)
        adv, p = advance(a, b)
        win[mno] = adv
        r32.append({"home": a, "away": b, "advances": adv, "p": round(p, 3),
                    "m": mno})
    bracket.append(("Round of 32", r32))

    names = {(89, 96): "Round of 16", (97, 100): "Quarter-finals",
             (101, 102): "Semi-finals", (104, 104): "Final"}
    for (lo, hi), label in names.items():
        rnd = []
        for mno in range(lo, hi + 1):
            if mno not in LATER:
                continue
            m1, m2 = LATER[mno]
            a, b = win[m1], win[m2]
            adv, p = advance(a, b)
            win[mno] = adv
            rnd.append({"home": a, "away": b, "advances": adv, "p": round(p, 3),
                        "m": mno})
        bracket.append((label, rnd))
    return bracket


def recent_form(bundle, teams, n: int = 5) -> dict:
    """Last-N results (W/D/L string, most recent last) for each team."""
    res = {t: [] for t in teams}
    for m in bundle["played"].itertuples(index=False):
        for team, gf, ga in ((m.home_team, m.home_score, m.away_score),
                             (m.away_team, m.away_score, m.home_score)):
            if team in res:
                res[team].append("W" if gf > ga else ("D" if gf == ga else "L"))
    return {t: "".join(v[-n:]) for t, v in res.items()}


def recent_results(bundle, teams, n: int = 5) -> dict:
    """Last-N actual results per team (most recent first): opponent + scoreline."""
    res = {t: [] for t in teams}
    for m in bundle["played"].itertuples(index=False):
        for team, opp, gf, ga, home in (
                (m.home_team, m.away_team, m.home_score, m.away_score, True),
                (m.away_team, m.home_team, m.away_score, m.home_score, False)):
            if team in res:
                res[team].append({"date": str(m.date)[:10], "opp": opp,
                                  "gf": int(gf), "ga": int(ga), "home": home})
    return {t: v[-n:][::-1] for t, v in res.items()}


def h2h_records(bundle, teams, recent: int = 5) -> dict:
    """All-time head-to-head record between every pair of these teams, plus the
    last `recent` meetings (date + scoreline) so the dashboard can show *how*
    the recent games actually went, not only the aggregate W-D-L count.
    """
    tset = set(teams)
    rec = {}
    for m in bundle["played"].itertuples(index=False):
        h, a = m.home_team, m.away_team
        if h in tset and a in tset:
            t1, t2 = sorted([h, a])
            r = rec.setdefault("|".join([t1, t2]),
                               {"t1": t1, "t2": t2, "w1": 0, "d": 0, "w2": 0,
                                "n": 0, "recent": []})
            r["n"] += 1
            if m.home_score == m.away_score:
                r["d"] += 1
            else:
                winner = h if m.home_score > m.away_score else a
                r["w1" if winner == t1 else "w2"] += 1
            r["recent"].append({"date": str(pd.Timestamp(m.date).date()),
                                 "home": h, "away": a,
                                 "hs": int(m.home_score), "as": int(m.away_score)})
    for r in rec.values():  # keep only the most recent meetings, newest first
        r["recent"] = sorted(r["recent"], key=lambda x: x["date"],
                             reverse=True)[:recent]
    return rec


def played_review(bundle, trained) -> list:
    """For each World Cup match already played: the model's *pre-match* W/D/L
    probabilities and most likely score vs. the actual result.

    `trained` should be the pre-tournament model (trained before the first
    World Cup match) so the prediction is a genuine blind call, not in-sample.
    """
    team_letter = {t: L for L, ts in OFFICIAL_GROUPS.items() for t in ts}
    out = []
    for g in bundle["wc_played"].itertuples(index=False):
        rep = match_report(trained, g.home_team, g.away_team, neutral=bool(g.neutral))
        hs, as_ = int(g.home_score), int(g.away_score)
        actual = "home" if hs > as_ else ("away" if as_ > hs else "draw")
        probs = {"home": rep["p_home"], "draw": rep["p_draw"], "away": rep["p_away"]}
        pred = max(probs, key=probs.get)
        out.append({
            "date": str(pd.Timestamp(g.date).date()),
            "group": team_letter.get(g.home_team, "?"),
            "home": g.home_team, "away": g.away_team, "hs": hs, "as": as_,
            "p_home": rep["p_home"], "p_draw": rep["p_draw"], "p_away": rep["p_away"],
            "ml_score": rep["most_likely"], "p_ml": rep["p_most_likely"],
            "actual": actual, "pred": pred, "hit": pred == actual,
            "p_actual": probs[actual],
        })
    out.sort(key=lambda r: (r["date"], r["group"]))
    return out


if __name__ == "__main__":
    from . import data as D

    b = D.load_all()
    tr = M.train_full(b)
    from . import tournament as T
    table = T.simulate(b, tr, n_sims=20000)

    print("=== H2H simulator ===")
    print(format_match(match_report(tr, "Spain", "Brazil", neutral=True)))
    print(format_match(match_report(tr, "Argentina", "France", neutral=True)))

    fav, fav_group = table.iloc[0]["team"], table.iloc[0]["group"]
    print(f"\n=== Upcoming matches in Group {fav_group} (favourite: {fav}) ===")
    gs = group_stage_predictions(b, tr)
    print(gs[gs.group == fav_group].to_string(index=False))

    print(f"\n=== Expected standings of Group {fav_group} ===")
    es = expected_standings(table)
    print(es[es.group == fav_group].to_string(index=False))

    print(f"\n=== {fav}'s opponents by round ===")
    for rnd, info in opponents_for(table, fav).items():
        opps = ", ".join(f"{o['team']} {o['p_cond']*100:.0f}%" for o in info["opponents"])
        print(f"{rnd} (reaches {info['p_reach']*100:.0f}%): {opps}")

    print("\n=== Most likely bracket (start) ===")
    brk = most_likely_bracket(table, tr)
    for label, matches in brk[:2]:
        print(f"-- {label} --")
        for m in matches:
            print(f"  {m['home']} vs {m['away']} -> {m['advances']} ({m['p']*100:.0f}%)")
