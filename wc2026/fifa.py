"""Official FIFA/Coca-Cola Men's World Ranking for the 48 finalists, to compare
with the model's own (Elo-based) strength ranking.

Snapshot: 11 June 2026 — the last edition published before the World Cup
(Argentina #1). Team names are normalised to the dataset's spellings.
"""
from __future__ import annotations

RANK_DATE = "11 Jun 2026"

# {dataset team name: (FIFA world rank, FIFA points)}
FIFA = {
    "Argentina": (1, 1877.27), "Spain": (2, 1874.71), "France": (3, 1870.70),
    "England": (4, 1828.02), "Portugal": (5, 1767.85), "Brazil": (6, 1765.86),
    "Morocco": (7, 1755.10), "Netherlands": (8, 1753.57), "Belgium": (9, 1742.24),
    "Germany": (10, 1735.77), "Croatia": (11, 1714.87), "Colombia": (13, 1698.35),
    "Mexico": (14, 1687.48), "Senegal": (15, 1684.07), "Uruguay": (16, 1673.07),
    "United States": (17, 1671.23), "Japan": (18, 1661.58), "Switzerland": (19, 1650.06),
    "Iran": (20, 1619.58), "Turkey": (22, 1605.73), "Ecuador": (23, 1598.52),
    "Austria": (24, 1597.40), "South Korea": (25, 1591.63), "Australia": (27, 1579.34),
    "Algeria": (28, 1571.03), "Egypt": (29, 1562.37), "Canada": (30, 1559.48),
    "Norway": (31, 1557.44), "Ivory Coast": (33, 1540.87), "Panama": (34, 1539.16),
    "Sweden": (38, 1509.79), "Czech Republic": (40, 1505.74), "Paraguay": (41, 1505.35),
    "Scotland": (42, 1503.34), "Tunisia": (45, 1476.41), "DR Congo": (46, 1474.43),
    "Uzbekistan": (50, 1458.73), "Qatar": (56, 1450.31), "Iraq": (57, 1446.28),
    "South Africa": (60, 1428.38), "Saudi Arabia": (61, 1423.88), "Jordan": (63, 1387.74),
    "Bosnia and Herzegovina": (64, 1387.22), "Cape Verde": (67, 1371.11),
    "Ghana": (73, 1346.88), "Curaçao": (82, 1294.77), "Haiti": (83, 1293.10),
    "New Zealand": (85, 1275.58),
}


def compare(table) -> dict:
    """Compare the model's Elo ranking with the FIFA ranking, both re-ranked
    1..N among the finalists. `edge` > 0 means the model rates the team higher
    (better) than FIFA does. Also returns the Spearman rank correlation.
    """
    teams = [t for t in table["team"] if t in FIFA]
    elo = dict(zip(table["team"], table["elo"]))

    by_elo = sorted(teams, key=lambda t: -elo[t])
    model_rank = {t: i + 1 for i, t in enumerate(by_elo)}
    by_fifa = sorted(teams, key=lambda t: FIFA[t][0])
    fifa_rank48 = {t: i + 1 for i, t in enumerate(by_fifa)}

    rows = []
    for t in teams:
        fr, fp = FIFA[t]
        rows.append({
            "team": t, "fifa_rank": fr, "fifa_pts": fp,
            "fifa_rank48": fifa_rank48[t], "model_rank": model_rank[t],
            "elo": int(elo[t]), "edge": fifa_rank48[t] - model_rank[t],
        })
    rows.sort(key=lambda r: r["model_rank"])

    n = len(rows)
    d2 = sum((r["model_rank"] - r["fifa_rank48"]) ** 2 for r in rows)
    spearman = 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else 1.0
    return {"date": RANK_DATE, "n": n, "spearman": round(spearman, 3), "rows": rows}


def rank_of(team: str):
    """Global FIFA world rank of a team (or None)."""
    return FIFA[team][0] if team in FIFA else None


if __name__ == "__main__":
    from . import data as D, model as M, tournament as T
    b = D.load_all()
    tab = T.simulate(b, M.train_full(b), n_sims=8000)
    c = compare(tab)
    print(f"Model vs FIFA ({c['date']}) — Spearman correlation {c['spearman']}")
    print("\nModel's biggest believers (rates higher than FIFA):")
    for r in sorted(c["rows"], key=lambda r: -r["edge"])[:5]:
        print(f"  {r['team']:<14} model #{r['model_rank']:<2} vs FIFA #{r['fifa_rank48']:<2} "
              f"(world #{r['fifa_rank']})  +{r['edge']}")
    print("\nModel's biggest doubters (rates lower than FIFA):")
    for r in sorted(c["rows"], key=lambda r: r["edge"])[:5]:
        print(f"  {r['team']:<14} model #{r['model_rank']:<2} vs FIFA #{r['fifa_rank48']:<2} "
              f"(world #{r['fifa_rank']})  {r['edge']}")
