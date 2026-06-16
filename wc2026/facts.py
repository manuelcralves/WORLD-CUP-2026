"""Historical trivia from the 49 thousand matches (1872–2026).

Ties back to the "fun stuff" from the start: biggest upsets of all time
(by Elo), goal trend across the decades, evolution of home advantage,
penalty kings and whipping boys, and a historical "card" for any team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import goldenboot as GB
from . import penalties as PEN


# --------------------------------------------------------------------------- #
def global_facts(matches: pd.DataFrame) -> dict:
    m = matches[matches["home_score"].notna()].copy()
    m["diff"] = (m["home_score"] - m["away_score"]).abs()
    m["total"] = m["home_score"] + m["away_score"]
    blow = m.loc[m["diff"].idxmax()]
    high = m.loc[m["total"].idxmax()]
    g = GB.load_goalscorers()
    top = g[~g["own_goal"]]["scorer"].value_counts().head(1)
    return {
        "n_matches": int(len(m)),
        "total_goals": int(m["total"].sum()),
        "biggest_win": (f"{blow.home_team} {int(blow.home_score)}-"
                        f"{int(blow.away_score)} {blow.away_team}",
                        str(blow["date"].date())),
        "highest_scoring": (f"{high.home_team} {int(high.home_score)}-"
                            f"{int(high.away_score)} {high.away_team}",
                            str(high["date"].date())),
        "top_scorer": (top.index[0], int(top.iloc[0])),
    }


def biggest_upsets(matches_elo: pd.DataFrame, n: int = 6,
                   since: str = "1960-01-01") -> list[dict]:
    m = matches_elo[matches_elo["home_score"].notna()
                    & (matches_elo["home_score"] != matches_elo["away_score"])
                    & (matches_elo["date"] >= pd.Timestamp(since))].copy()
    home_won = m["home_score"] > m["away_score"]
    m["w_elo"] = np.where(home_won, m["home_elo_pre"], m["away_elo_pre"])
    m["l_elo"] = np.where(home_won, m["away_elo_pre"], m["home_elo_pre"])
    m["gap"] = m["l_elo"] - m["w_elo"]            # >0 => the underdog won
    m = m[m["l_elo"] > 1850]                       # really strong favorite
    out = []
    for r in m.sort_values("gap", ascending=False).head(n).itertuples(index=False):
        hw = r.home_score > r.away_score
        winner = r.home_team if hw else r.away_team
        loser = r.away_team if hw else r.home_team
        out.append({"date": str(r.date.date()), "winner": winner, "loser": loser,
                    "score": f"{int(r.home_score)}-{int(r.away_score)}"
                             if hw else f"{int(r.away_score)}-{int(r.home_score)}",
                    "gap": int(r.gap), "tournament": r.tournament})
    return out


def goal_trend(matches: pd.DataFrame) -> list[dict]:
    m = matches[matches["home_score"].notna()].copy()
    m["decade"] = (m["date"].dt.year // 10) * 10
    g = m.groupby("decade").apply(
        lambda d: (d["home_score"] + d["away_score"]).mean(), include_groups=False)
    return [{"decade": int(k), "avg_goals": round(float(v), 2)}
            for k, v in g.items() if k >= 1900]


def home_advantage_trend(matches: pd.DataFrame) -> list[dict]:
    m = matches[matches["home_score"].notna() & (~matches["neutral"])].copy()
    m["decade"] = (m["date"].dt.year // 10) * 10
    m["home_win"] = m["home_score"] > m["away_score"]
    g = m.groupby("decade")["home_win"].mean()
    return [{"decade": int(k), "home_win_pct": round(float(v) * 100, 1)}
            for k, v in g.items() if k >= 1900]


def _team_matches(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    m = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    m = m[m["home_score"].notna()]
    is_home = m["home_team"] == team
    return pd.DataFrame({
        "date": m["date"].values,
        "opp": np.where(is_home, m["away_team"], m["home_team"]),
        "gf": np.where(is_home, m["home_score"], m["away_score"]),
        "ga": np.where(is_home, m["away_score"], m["home_score"]),
    }).sort_values("date").reset_index(drop=True)


def team_card(matches_elo: pd.DataFrame, team: str) -> dict:
    t = _team_matches(matches_elo, team)
    if t.empty:
        return {"team": team}
    w = int((t.gf > t.ga).sum()); d = int((t.gf == t.ga).sum())
    loss = int((t.gf < t.ga).sum())
    t = t.assign(margin=t.gf - t.ga)
    bw = t.loc[t["margin"].idxmax()]
    bl = t.loc[t["margin"].idxmin()]
    # longest unbeaten run
    best = run = 0
    for ok in (t.gf >= t.ga).values:
        run = run + 1 if ok else 0
        best = max(best, run)
    rival = t["opp"].value_counts().index[0]
    rv = t[t.opp == rival]
    # Elo peak
    he = matches_elo[matches_elo.home_team == team][["date", "home_elo_pre"]] \
        .rename(columns={"home_elo_pre": "elo"})
    ae = matches_elo[matches_elo.away_team == team][["date", "away_elo_pre"]] \
        .rename(columns={"away_elo_pre": "elo"})
    elo = pd.concat([he, ae])
    peak = elo.loc[elo["elo"].idxmax()] if len(elo) else None
    return {
        "team": team, "games": len(t), "w": w, "d": d, "l": loss,
        "win_pct": round(w / len(t) * 100, 1),
        "biggest_win": f"{int(bw.gf)}-{int(bw.ga)} vs {bw.opp} ({str(bw.date.date())[:4]})",
        "worst_loss": f"{int(bl.gf)}-{int(bl.ga)} vs {bl.opp} ({str(bl.date.date())[:4]})",
        "longest_unbeaten": int(best),
        "rival": rival, "rival_record": f"{int((rv.gf>rv.ga).sum())}W-"
                                        f"{int((rv.gf==rv.ga).sum())}D-"
                                        f"{int((rv.gf<rv.ga).sum())}L in {len(rv)}",
        "peak_elo": (int(peak["elo"]), str(peak["date"].date())[:4]) if peak is not None else None,
    }


def penalty_kings(min_games: int = 5) -> dict:
    s = PEN.load_shootouts()
    rec = PEN.team_records(s, min_games)
    fs = PEN.first_shooter_advantage(s)
    return {
        "best": [{"team": r.team, "pct": round(r.win_pct * 100), "n": int(r.shootouts)}
                 for r in rec.head(5).itertuples(index=False)],
        "worst": [{"team": r.team, "pct": round(r.win_pct * 100), "n": int(r.shootouts)}
                  for r in rec.tail(5).iloc[::-1].itertuples(index=False)],
        "first_shooter_pct": round(fs["win_rate_shooting_first"] * 100, 1),
    }


def fun_facts(matches_elo: pd.DataFrame, team: str = None) -> dict:
    out = {
        "global": global_facts(matches_elo),
        "upsets": biggest_upsets(matches_elo),
        "goal_trend": goal_trend(matches_elo),
        "home_trend": home_advantage_trend(matches_elo),
        "penalties": penalty_kings(),
    }
    if team is not None:
        out["team_card"] = team_card(matches_elo, team)
    return out


if __name__ == "__main__":
    from . import data as D, elo as E
    b = D.load_all()
    df_elo, ratings, _ = E.compute_elo(b["matches"])
    top_team = max(ratings, key=ratings.get)   # current Elo leader (neutral example)
    f = fun_facts(df_elo, top_team)
    g = f["global"]
    print(f"Biggest win: {g['biggest_win'][0]} ({g['biggest_win'][1]})")
    print(f"Top scorer: {g['top_scorer'][0]} ({g['top_scorer'][1]} goals)")
    print("\nBiggest upsets (underdog won, by Elo):")
    for u in f["upsets"]:
        print(f"  {u['date']}  {u['winner']} {u['score']} {u['loser']}  "
              f"(Δelo {u['gap']}, {u['tournament']})")
    print("\nGoal trend by decade:",
          ", ".join(f"{d['decade']}s:{d['avg_goals']}" for d in f["goal_trend"][-5:]))
    print("Home advantage by decade:",
          ", ".join(f"{d['decade']}s:{d['home_win_pct']}%" for d in f["home_trend"][-5:]))
    pk = f["penalties"]
    print(f"\nPenalties — shooting first wins {pk['first_shooter_pct']}%")
    print("Penalty kings:", ", ".join(f"{x['team']} {x['pct']}%" for x in pk["best"][:3]))
    c = f["team_card"]
    print(f"\n{c['team']} card: {c['w']}W-{c['d']}D-{c['l']}L ({c['win_pct']}%), "
          f"biggest win {c['biggest_win']}, unbeaten run {c['longest_unbeaten']}, "
          f"rival {c['rival']} ({c['rival_record']}), Elo peak {c['peak_elo']}")
