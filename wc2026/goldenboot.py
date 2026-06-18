"""Golden Boot (top scorer) forecast for the 2026 World Cup.

We don't have the 2026 squads, so we use recent scorers (since 2023) as an
approximation of the likely roster. For each player:

  projected goals = player_share_of_team_goals
                    × team's_expected_goals_at_the_World_Cup

where the share comes from goalscorers.csv (what fraction of the national team's
recent goals are his) and expected goals = expected number of World Cup matches
(from the simulation) × the team's recent scoring rate (from the results).

Approximate, of course: players retire, new ones emerge, and the share may shift.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from . import data as D


def load_goalscorers(data_dir=D.DATA_DIR) -> pd.DataFrame:
    g = pd.read_csv(f"{data_dir}/goalscorers.csv", parse_dates=["date"])
    mapping = D.name_mapping(D.load_former_names(data_dir))
    for c in ("home_team", "away_team", "team"):
        g[c] = g[c].replace(mapping)
    g["own_goal"] = g["own_goal"].astype(str).str.upper().eq("TRUE")
    return g


def _team_scoring_rate(played, teams, since):
    p = played[played["date"] >= since]
    gf, n = defaultdict(float), defaultdict(int)
    for m in p.itertuples(index=False):
        if m.home_team in teams:
            gf[m.home_team] += m.home_score; n[m.home_team] += 1
        if m.away_team in teams:
            gf[m.away_team] += m.away_score; n[m.away_team] += 1
    return {t: (gf[t] / n[t] if n[t] else 1.2) for t in teams}


def predict(bundle, table, since="2023-01-01", before=None, topn=20) -> pd.DataFrame:
    teams = set(table["team"])
    # expected number of World Cup matches (3 group + knockout matches)
    exp_matches = {r.team: 3 + r.p_ko + r.p_r16 + r.p_qf + r.p_sf + r.p_final
                   for r in table.itertuples(index=False)}
    rate = _team_scoring_rate(bundle["played"], teams, pd.Timestamp(since))

    g = load_goalscorers()
    g = g[(g["date"] >= pd.Timestamp(since)) & (~g["own_goal"])
          & (g["team"].isin(teams))]
    if before is not None:  # pre-tournament mode: exclude goals from the World Cup itself
        g = g[g["date"] < pd.Timestamp(before)]
    player_goals = g.groupby("scorer").size()
    player_team = g.groupby("scorer")["team"].agg(lambda s: s.mode().iloc[0])
    team_goals = g.groupby("team").size()

    # goals actually scored in this World Cup so far (played matches only, live mode).
    # Identify WC matches by joining on (date, home, away) against bundle["wc_played"]
    # — a date cutoff alone would wrongly count June friendlies as World Cup goals.
    wc_goals = {}
    wcp = bundle.get("wc_played")
    if before is None and wcp is not None and len(wcp):
        mp = D.name_mapping(D.load_former_names(D.DATA_DIR))
        wc_keys = {(d, mp.get(h, h), mp.get(a, a))
                   for d, h, a in zip(wcp["date"], wcp["home_team"], wcp["away_team"])}
        in_wc = [(d, h, a) in wc_keys
                 for d, h, a in zip(g["date"], g["home_team"], g["away_team"])]
        wc_goals = g[pd.Series(in_wc, index=g.index)].groupby("scorer").size().to_dict()

    rows = []
    for scorer, goals in player_goals.items():
        team = player_team[scorer]
        share = goals / team_goals[team]
        proj = share * exp_matches[team] * rate[team]
        rows.append({"scorer": scorer, "team": team, "recent_goals": int(goals),
                     "wc_goals": int(wc_goals.get(scorer, 0)),
                     "exp_team_goals": round(exp_matches[team] * rate[team], 1),
                     "proj_goals": round(proj, 2)})
    return (pd.DataFrame(rows).sort_values("proj_goals", ascending=False)
            .head(topn).reset_index(drop=True))


if __name__ == "__main__":
    from . import model as M, tournament as T
    b = D.load_all()
    tr = M.train_full(b)
    table = T.simulate(b, tr, n_sims=20000)
    gb = predict(b, table)
    print("=== 2026 Golden Boot Forecast ===")
    print(gb.to_string(index=False))
    top = gb.iloc[0]
    print(f"\nTop projected scorer: {top['scorer']} ({top['team']}, "
          f"~{top['proj_goals']} goals)")
