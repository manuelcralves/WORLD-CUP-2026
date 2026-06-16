"""Extra analyses derived from the simulation table.

  - group_of_death  : groups ranked by combined strength
  - dark_horses     : low-Elo teams with surprising deep-run odds
  - likely_finals   : the most probable final match-ups (from the opp matrix)
  - path_difficulty : average Elo of the opponents each team is likely to face
  - champion_ci     : 95% Monte Carlo confidence interval on the title odds
"""
from __future__ import annotations


def group_of_death(table) -> list:
    """Each group ranked by raw strength, the way fans actually judge a "group
    of death": the average Elo of its four teams, and how strong the 3rd-best
    team is — a high 3rd-seed Elo means even a good side risks elimination, so
    it captures the *difficulty of advancing*, not just star power.
    """
    rows = []
    for L in sorted(table["group"].unique()):
        g = table[table["group"] == L].sort_values("elo", ascending=False)
        elos = [int(e) for e in g["elo"]]
        rows.append({
            "group": L,
            "avg_elo": int(round(g["elo"].mean())),
            "third_elo": elos[2] if len(elos) >= 3 else elos[-1],
            "elos": elos,
            "sum_champion": round(float(g["p_champion"].sum()), 4),
            "teams": list(g["team"]),  # strongest first, by Elo
        })
    # toughest = highest average Elo, tie-broken by the strength of the 3rd seed
    return sorted(rows, key=lambda r: (-r["avg_elo"], -r["third_elo"]))


def dark_horses(table, n: int = 5, elo_rank_min: int = 12) -> list:
    """Lower-ranked teams (by Elo) with the highest chance of a deep run."""
    t = table.copy()
    t["elo_rank"] = t["elo"].rank(ascending=False, method="min")
    dh = t[t["elo_rank"] > elo_rank_min].sort_values("p_qf", ascending=False).head(n)
    return [{"team": r.team, "elo": int(r.elo), "elo_rank": int(r.elo_rank),
             "p_qf": float(r.p_qf), "p_champion": float(r.p_champion)}
            for r in dh.itertuples(index=False)]


def likely_finals(table, n: int = 6) -> list:
    """Most probable final match-ups, from the round-by-round opponent matrix."""
    if "opp_matrix" not in table.attrs:
        return []
    teams = table.attrs["teams"]
    M = table.attrs["opp_matrix"]["F"]
    N = table.attrs["n_sims"]
    pairs = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            c = M[i, j]
            if c > 0:
                pairs.append((teams[i], teams[j], float(c) / N))
    pairs.sort(key=lambda x: -x[2])
    return [{"a": a, "b": b, "p": round(p, 4)} for a, b, p in pairs[:n]]


def path_difficulty(table, n: int = 8) -> list:
    """Average Elo of the opponents each team is likely to face in the knockouts."""
    if "opp_matrix" not in table.attrs:
        return []
    teams = table.attrs["teams"]
    elo = dict(zip(table["team"], table["elo"]))
    opp = table.attrs["opp_matrix"]
    rows = []
    for i, t in enumerate(teams):
        tot = wsum = 0.0
        for r in opp:
            row = opp[r][i]
            for j, c in enumerate(row):
                if c:
                    tot += c
                    wsum += c * elo[teams[j]]
        if tot > 0:
            rows.append({"team": t, "avg_opp_elo": int(round(wsum / tot)),
                         "elo": int(elo[t])})
    return sorted(rows, key=lambda r: -r["avg_opp_elo"])[:n]


def champion_ci(table, top: int = 12) -> list:
    """95% Monte Carlo confidence interval on the title probability."""
    N = table.attrs.get("n_sims", 1)
    out = []
    for r in table.head(top).itertuples(index=False):
        p = float(r.p_champion)
        se = (p * (1 - p) / N) ** 0.5
        out.append({"team": r.team, "p": p, "lo": max(0.0, p - 1.96 * se),
                    "hi": p + 1.96 * se})
    return out


if __name__ == "__main__":
    from . import data as D, model as M, tournament as T
    b = D.load_all()
    table = T.simulate(b, M.train_full(b), n_sims=20000)
    god = group_of_death(table)
    print("Group of death:", god[0]["group"],
          f"({god[0]['sum_champion']*100:.0f}% combined title prob,",
          ", ".join(god[0]["teams"][:2]), "…)")
    print("\nDark horses:")
    for d in dark_horses(table):
        print(f"  {d['team']} (Elo #{d['elo_rank']}): {d['p_qf']*100:.0f}% to reach the QF")
    print("\nMost likely finals:")
    for f in likely_finals(table):
        print(f"  {f['a']} vs {f['b']}: {f['p']*100:.1f}%")
    print("\nToughest paths:", ", ".join(
        f"{r['team']} (opp Elo {r['avg_opp_elo']})" for r in path_difficulty(table)[:4]))
