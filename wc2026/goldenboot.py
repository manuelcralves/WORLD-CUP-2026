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
    # expected number of World Cup matches (3 group + knockout). Everyone who reaches the
    # semis plays one more game whatever happens — the final if they win, the third-place
    # play-off if they lose — so the semi-onward games are 2*p_sf, not p_sf + p_final (the
    # old formula silently assumed the losing semi-finalists went home).
    exp_matches = {r.team: 3 + r.p_ko + r.p_r16 + r.p_qf + 2 * r.p_sf
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

    # Goals already scored in this World Cup, and how many WC matches each team
    # has played — identified by joining on (date, home, away) against
    # bundle["wc_played"] (a date cutoff alone would count June friendlies too).
    # load_all already normalised every name, so the keys line up directly.
    wc_goals, played_wc = {}, {}
    wcp = bundle.get("wc_played")
    ko = bundle.get("knockout")                                   # knockout games live OUTSIDE wc_played now (world_cup_2026 is head(72))
    kop = ko[ko["home_score"].notna()] if ko is not None and len(ko) else None
    frames = [f for f in (wcp, kop) if f is not None and len(f)]
    wcplayed = pd.concat(frames, ignore_index=True) if frames else None   # every played WC game: group + knockout
    if before is None and wcplayed is not None and len(wcplayed):
        wc_keys = {(d, h, a) for d, h, a in
                   zip(wcplayed["date"], wcplayed["home_team"], wcplayed["away_team"])}
        in_wc = [(d, h, a) in wc_keys
                 for d, h, a in zip(g["date"], g["home_team"], g["away_team"])]
        wc_goals = g[pd.Series(in_wc, index=g.index)].groupby("scorer").size().to_dict()
        for h, a in zip(wcplayed["home_team"], wcplayed["away_team"]):
            played_wc[h] = played_wc.get(h, 0) + 1
            played_wc[a] = played_wc.get(a, 0) + 1

    # Recent form: each player's goals in his national team's last 10 matches.
    played = bundle["played"]
    last10 = set()
    for t in teams:
        tm = played[(played["home_team"] == t) | (played["away_team"] == t)]
        for d in tm.sort_values("date")["date"].tail(10):
            last10.add((t, d))
    in_l10 = [(t, d) in last10 for t, d in zip(g["team"], g["date"])]
    form10 = g[pd.Series(in_l10, index=g.index)].groupby("scorer").size().to_dict()

    rows = []
    for scorer, goals in player_goals.items():
        team = player_team[scorer]
        share = goals / team_goals[team]
        banked = int(wc_goals.get(scorer, 0))
        # final-total forecast: goals already banked + expected in the games still to play
        remaining = max(0.0, exp_matches[team] - played_wc.get(team, 0))
        proj = banked + share * remaining * rate[team]
        rows.append({"scorer": scorer, "team": team, "recent_goals": int(goals),
                     "form10": int(form10.get(scorer, 0)), "wc_goals": banked,
                     "exp_remaining": round(share * remaining * rate[team], 2),
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
