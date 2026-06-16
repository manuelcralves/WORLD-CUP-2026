"""Match goal timelines + an "anatomy of a goal" breakdown.

Built only from goalscorers.csv — every international goal since 1916, with the
scorer, the minute, and penalty / own-goal flags. Covers ALL internationals
(World Cups, qualifiers, friendlies, everything), no extra datasets needed.
"""
from __future__ import annotations

import pandas as pd

from .goldenboot import load_goalscorers

_G = None


def _all() -> pd.DataFrame:
    global _G
    if _G is None:
        g = load_goalscorers()
        g["penalty"] = g["penalty"].astype(str).str.upper().eq("TRUE")
        _G = g
    return _G


def match_goals(home: str, away: str, date) -> list:
    """Ordered goal timeline for one match (scorer, minute, pen/own-goal flags)."""
    g = _all()
    d = pd.Timestamp(date).normalize()
    sub = g[(g["date"].dt.normalize() == d) & (g["home_team"] == home)
            & (g["away_team"] == away)].sort_values("minute", na_position="last")
    return [{"scorer": r.scorer, "team": r.team,
             "minute": int(r.minute) if pd.notna(r.minute) else None,
             "pen": bool(r.penalty), "og": bool(r.own_goal)}
            for r in sub.itertuples(index=False)]


def scoring_analysis() -> dict:
    """When goals are scored + penalty / own-goal share, over every international."""
    g = _all()
    m = g["minute"].dropna()
    buckets = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
    by_min = [{"label": f"{lo}-{hi}", "count": int(((m >= lo) & (m <= hi)).sum())}
              for lo, hi in buckets]
    by_min.append({"label": "90+", "count": int((m > 90).sum())})
    return {"total_goals": int(len(g)), "since": int(g["date"].dt.year.min()),
            "pen_pct": round(float(g["penalty"].mean()) * 100, 1),
            "og_pct": round(float(g["own_goal"].mean()) * 100, 1),
            "by_minute": by_min}


if __name__ == "__main__":
    from . import data as D
    b = D.load_all()
    a = scoring_analysis()
    print(f"{a['total_goals']:,} goals since {a['since']} · {a['pen_pct']}% pens "
          f"· {a['og_pct']}% own goals")
    print("by minute:", [(x["label"], x["count"]) for x in a["by_minute"]])
    print("\nGoal timelines (played WC 2026 games):")
    for m in b["wc_played"].head(4).itertuples(index=False):
        gl = match_goals(m.home_team, m.away_team, m.date)
        s = ", ".join(f"{x['minute']}' {x['scorer']}" + ("(p)" if x["pen"] else "")
                      + ("(og)" if x["og"] else "") for x in gl)
        print(f"  {m.home_team} {m.home_score:.0f}-{m.away_score:.0f} {m.away_team}: {s}")
