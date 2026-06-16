"""Generates the narrated notebook World_Cup_2026.ipynb (JSON v4, without needing nbformat).

Run once:  python make_notebook.py
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    cells.append({"cell_type": "code", "metadata": {},
                  "execution_count": None, "outputs": [], "source": text})


md("""# 🏆 2026 World Cup — *Machine Learning* Prediction

This notebook builds, step by step, a model that predicts the 2026 World Cup
from the dataset *International football results from 1872 to 2026*.

The recipe:
1. **Elo** — a strength rating for each team, updated match by match since 1872.
2. **Poisson model (Dixon-Coles)** — learns how many goals each team scores,
   as a function of Elo, recent form and the home factor.
3. **Validation** — we prove that the model beats baselines on matches it has never seen.
4. **Monte Carlo simulation** — we run the tournament thousands of times to obtain
   each team's probability of becoming champion.

> Run the cells in order. You need `numpy`, `pandas`, `scipy` and `matplotlib`
> (the `wc2026` package is in the same folder).""")

code("""%matplotlib inline
import pandas as pd
from wc2026 import data as D, elo as E, model as M, validate as V, tournament as T, viz
pd.set_option("display.width", 140)

bundle = D.load_all()
print(f"{len(bundle['played']):,} matches with a result")
print(f"2026 World Cup: {len(bundle['wc_played'])} played, "
      f"{len(bundle['wc_remaining'])} to play")
pd.DataFrame({L: ", ".join(t) for L, t in bundle['groups'].items()}.items(),
             columns=["group", "teams"])""")

md("""## 1. Elo ratings

For each match we compute the win probability expected from Elo and adjust the
ratings according to the result (big wins move them more; important matches weigh
more; the home team gets a bonus). It is the model's strongest feature — and it has
no data leakage, because the *pre-match* rating only uses the past.""")

code("""df_elo, ratings, _ = E.compute_elo(bundle["matches"])
E.ranking(ratings, 15)""")

md("""## 2. The goals model (Poisson / Dixon-Coles)

Each match becomes **two rows** (one per team), where the target is the goals scored.
By maximum likelihood we fit the expected number of goals `λ` as a function of:
own Elo, opponent Elo, home/away, friendly and recent form. The **Dixon-Coles**
correction (`ρ`) adjusts the dependence in low scores (0-0, 1-0, 1-1).""")

code("""trained = M.train_full(bundle)
mdl, state, default = trained["model"], trained["state"], trained["default"]
print("rho (Dixon-Coles) =", round(mdl.rho, 4))
mdl.coef_table()""")

code("""# Sanity check: a few ties
for h, a, neutral in [("Brazil", "Morocco", True),
                      ("Spain", "Cape Verde", True),
                      ("Argentina", "France", True)]:
    lh, la = mdl.lambdas_for(state, default, h, a, neutral)
    ph, pe, pa = M.outcome_probs(lh, la, mdl.rho)
    print(f"{h:>10} vs {a:<12} xG {lh:.2f}-{la:.2f} | "
          f"V {ph*100:4.0f}%  E {pe*100:4.0f}%  D {pa*100:4.0f}%")""")

md("""## 3. Validation (temporal holdout)

We train only on matches **before 2022** and evaluate on the following ones (which the
model has never seen). We compare with the naive base rate and an Elo-only baseline.
Metrics: accuracy, *log-loss* and **RPS** (the standard metric in football; the lower,
the better).""")

code("""val = V.evaluate(bundle)
val.round(4)""")

md("""Result: ~**60% accuracy** and **RPS ≈ 0.17**, well above the naive guess
(0.23) — a level comparable to bookmaker models. (Elo already carries almost all the
signal; recent form helps above all with the *realism of the exact scores*.)""")

md("""## 4. Monte Carlo simulation of the tournament

We fix the 12 matches already played and simulate N times the remaining 60 group-stage
matches + the knockout rounds. Qualification by the FIFA criteria (points → goal
difference → goals scored), the 8 best third-placed teams and the **official 2026
bracket** (with extra time and penalties in ties).""")

code("""table = T.simulate(bundle, trained, n_sims=20000)

show = table.copy()
for c in T.STAGES + ["p_win_group"]:
    show[c] = (show[c] * 100).round(1)
show.head(16)""")

md("## 5. Results")

code("fig = viz.champion_chart(table, top=15)")

code('''# Path of the favourite (top of the ranking)
fav = table.iloc[0]
fig = viz.path_chart(table)          # defaults to the favourite
print(f"{fav.team}: {fav.p_champion*100:.1f}% champion, "
      f"{fav.p_ko*100:.0f}% gets past the group stage, "
      f"{fav.p_final*100:.1f}% reaches the final.")''')

md("""## 6. Richer predictions

Most likely score of each match, expected group standings, and any team's most likely
opponents in each round (here, the favourite's).""")

code('''from wc2026 import predictions as PR

# Match simulator (any pair of teams)
print(PR.format_match(PR.match_report(trained, "Spain", "Brazil")))
print(PR.format_match(PR.match_report(trained, "Argentina", "France")))''')

code('''# Upcoming matches in the favourite's group: most likely score + W/D/L
fav_group = table.iloc[0].group
PR.group_stage_predictions(bundle, trained).query("group == @fav_group")''')

code('''# Expected standings of the favourite's group
PR.expected_standings(table).query("group == @fav_group")''')

code('''# The favourite's most likely opponents, round by round
fav_name = table.iloc[0].team
for rnd, info in PR.opponents_for(table, fav_name).items():
    opps = ", ".join(f"{o['team']} {o['p_cond']*100:.0f}%"
                     for o in info["opponents"])
    print(f"{rnd} (reaches {info['p_reach']*100:.0f}%): {opps}")''')

md("""## 7. Backtesting — does it work?

Trains the model only on data before each World Cup and compares the prediction with
what actually happened (2018 and 2022).""")

code('''from wc2026 import backtest as BT
for yr in (2018, 2022):
    BT.run(bundle, yr)''')

md("""## 8. Golden Boot & Elo through history

Who scores the most at the World Cup? And how has the teams' strength evolved since 1960?""")

code('''from wc2026 import goldenboot as GB
GB.predict(bundle, table).head(12)''')

code('''# Elo bar chart race (generates and shows the GIF)
viz.elo_race_gif(bundle["matches"], "outputs/elo_race.gif")
from IPython.display import Image
Image("outputs/elo_race.gif")''')

md("""## 9. Interactive dashboard

Generates the interactive page (clickable ranking, groups, match-by-match, drawn
bracket, backtesting, Golden Boot and Elo) that opens in any browser.""")

code('''from wc2026 import dashboard as DASH
viz.save_charts(table, "outputs")
data = DASH.collect(bundle, trained, table, val,
                    [BT.summary(bundle, y) for y in (2018, 2022)])
DASH.build_interactive(data, "outputs/dashboard.html")
print("Interactive dashboard at outputs/dashboard.html")''')

md("""---
### Notes and limitations

- The model is **data-driven** and probabilistic: a "3% chance of being champion"
  is not a guess, but there is always enormous uncertainty in a tournament.
- The hosts' home advantage is halved in the knockout rounds (they are played in more
  neutral stadiums).
- The assignment of the 8 best third-placed teams respects the official eligibility lists
  (no group rematches); FIFA's exact table may differ in extreme cases.
- Just for fun. ⚽""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "World_Cup_2026.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook written: {out} ({len(cells)} cells)")
