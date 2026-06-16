# 🏆 FIFA World Cup 2026 — Machine Learning Prediction

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Model](https://img.shields.io/badge/RPS-0.17%20(bookmaker%20level)-orange)
![Matches](https://img.shields.io/badge/data-49k%20matches%20since%201872-lightgrey)

A model that predicts the 2026 FIFA World Cup from the *International football
results from 1872 to 2026* dataset (Kaggle), with an **interactive dashboard**
(ranking, click-through team profiles, a live head-to-head **Match Lab**),
match-by-match predictions and proven **backtesting**.

Three pieces at the core of the model:

1. **Elo ratings** — each team's strength, updated match by match since 1872.
2. **Poisson model (Dixon-Coles)** — learns, by maximum likelihood, how many goals
   each team scores given the Elo, recent form and home advantage. From that come
   the probabilities of every exact scoreline.
3. **Monte Carlo simulation** — fixes the matches already played and runs the
   tournament tens of thousands of times (groups with FIFA tiebreakers, the 8 best
   third-placed teams and the **official 2026 bracket**, with extra time and
   **penalties modelled from 677 real shootouts**).

## Results (example, 30 000 simulations)

| # | Team | Champion |
|---|------|---------:|
| 1 | 🇦🇷 Argentina | 20.5% |
| 2 | 🇪🇸 Spain | 14.0% |
| 3 | 🏴 England | 8.9% |
| 4 | 🇫🇷 France | 8.4% |
| 5 | 🇩🇪 Germany | 6.0% |
| 6 | 🇲🇽 Mexico | 5.2% |

In the interactive dashboard you can **click any team** to see its path to the final,
its most likely opponents round by round, and its history card.

*(Numbers move as results come in — see the two versions below.)*

## How to run

Requirements: `numpy`, `pandas`, `scipy`, `matplotlib` (already installed).
XGBoost/scikit-learn are optional (only for the `ensemble`/`xgboost` model).

### Option A — script (generates everything)

```bash
python run_pipeline.py                  # live version, 30 000 simulations
python run_pipeline.py 50000 both       # BOTH versions (live + pre-tournament)
python run_pipeline.py pretournament    # pre-tournament version only (blind)
python run_pipeline.py 30000 ensemble   # use the Poisson+XGBoost ensemble (needs xgboost)
```

Arguments can come in any order (number of simulations, `poisson`/`xgboost`/
`ensemble`, and `snapshot`/`pretournament`/`both`).

**One click:** double-click `update_and_run.bat` to update the data and rebuild
both versions in one go.

### Two versions

- **Live (`snapshot`)** → `outputs/`: uses the known World Cup results as fixed and
  predicts the rest. The "as it stands" picture.
- **Pre-tournament (`pretournament`)** → `outputs_pretournament/`: trains **only on data
  before 11 Jun 2026** and simulates the 72 group matches from scratch, knowing no
  2026 result or scorer. The "blind" forecast.

Since the tournament has barely started (16 of 104 matches), the two are close at
the top — but they differ a lot for teams that have already played (e.g. Turkey,
which lost, is worth 3% to win its group in the live version vs. 33% pre-tournament).

### Option B — narrated notebook

Open `World_Cup_2026.ipynb` (VS Code or Jupyter) and run the cells in order.

### Updating the data

The CSVs come from the live **martj42** repository (the same source as Kaggle),
which is updated continuously. To fetch the latest version (with an automatic
backup of the current files in `_backup/`):

```bash
python update_data.py            # download the 4 latest CSVs
python run_pipeline.py both      # refresh the predictions with the new results
```

> 💡 During the World Cup, run these two commands each day (or double-click
> `update_and_run.bat`). The `tracker` accumulates the odds history in
> `history.csv` and the odds-over-time chart updates itself.

### Publishing it online (auto-deploy)

The repo ships with a **GitHub Actions** workflow (`.github/workflows/deploy.yml`)
that, **every day**, fetches the latest results, re-runs both versions and publishes
the site to GitHub Pages — no manual step and your PC doesn't need to be on.

To enable it: push the repo to GitHub, then **Settings → Pages → Source: GitHub
Actions**. The site goes live at `https://<user>.github.io/<repo>/` — a landing page
(`index.html`) links to the live dashboard, the pre-tournament one and the comparison.
The workflow also commits the refreshed data and `history.csv` back, so the
odds-over-time chart keeps accumulating.

Prefer manual? Drag the `outputs/` folder to
[Netlify Drop](https://app.netlify.com/drop). For social media, use the cards
`share_title.png` / `share_spotlight.png`.

## Structure

```
wc2026/
  data.py         loading, name normalisation, group reconstruction
  elo.py          chronological Elo ratings
  features.py     feature engineering (no leakage)
  model.py        Poisson model (Dixon-Coles) + training + scoreline grid
  validate.py     temporal validation (vs baselines) + calibration
  tune.py         hyperparameter tuning via a temporal grid
  penalties.py    penalty-shootout model (from the shootouts)
  tournament.py   Monte Carlo simulation (groups + official knockouts)
  predictions.py  likely scores, match-by-match, standings, bracket, H2H
  backtest.py     backtesting on the 2018 and 2022 World Cups
  goldenboot.py   top-scorer (Golden Boot) forecast
  xgb_model.py    XGBoost variant (gradient boosting) + ensemble
  tracker.py      prediction history + odds over time during the tournament
  facts.py        historical facts (team cards, upsets, penalties)
  analysis.py     group of death, dark horses, likely finals, confidence
  viz.py          matplotlib charts + Elo bar chart race (GIF)
  share.py        social cards + side-by-side comparison page
  dashboard.py    interactive HTML dashboard (ranking, Match Lab, ...)
run_pipeline.py   end-to-end pipeline (live + pre-tournament)
update_data.py    updates the CSVs from the live source (martj42)
update_and_run.bat one-click update + run (Windows)
make_notebook.py  generates World_Cup_2026.ipynb
index.html        landing page (links to both dashboards) — for GitHub Pages
requirements.txt · LICENSE · .github/workflows/deploy.yml  (daily auto-deploy)
```

Generated in each output folder (`outputs/` and/or `outputs_pretournament/`):
- **`dashboard.html`** — the **interactive** page: odds over time, sortable ranking
  (click a team to see its path and opponents), groups, match-by-match, drawn
  bracket, backtesting, Golden Boot, Elo through history and historical facts
- `share_title.png`, `share_portugal.png` — social cards
- `champions.png`, `portugal.png`, `elo_race.gif`, `odds_evolution.png`
- `predictions.csv`, `group_matches.csv`, `expected_standings.csv`,
  `golden_boot.csv`, `history.csv` (prediction time series)

And in the root, in `both` mode: `compare.html` (the two versions side by side).

## Model quality

**Validation** (temporal holdout, train < 2022, test 2022–2026, ~4 500 unseen
matches): ~**60% accuracy** and **RPS ≈ 0.17** — bookmaker level, well above the
naive base rate (RPS 0.23). **Calibration** is almost perfect: when the model says
70%, it happens ~74% of the time.

**Backtesting** (training only on pre-tournament data):

| World Cup | Actual champion | Model's rank | Match accuracy |
|-----------|-----------------|-------------:|---------------:|
| 2018 | 🇫🇷 France | #6 (4.9%) | 58% (vs 39% naive) |
| 2022 | 🇦🇷 Argentina | #2 (24%) | 50% (vs 44% naive) |

The model always beats the baseline and ranks the eventual champions among the
contenders — but tournaments are chaotic (France 2018 surprised almost every model).

**Tuning**: hyperparameters were optimised via a temporal grid minimising RPS
(`wc2026.tune`). It confirmed that Elo already carries almost all the signal — the
gain was marginal, which validates the initial choices.

**Poisson vs XGBoost** (`wc2026.xgb_model`, same holdout): essentially tied (RPS
0.1717 vs 0.1718), and the **ensemble of the two is best** (RPS 0.1714). With few
highly informative features, gradient boosting doesn't beat the statistical model —
but combining them gains a little. Poisson stays the default (fast, no extra deps).

## Limitations

- A **probabilistic, data-driven** model: there is always huge uncertainty in a
  tournament.
- Host advantage is halved in the knockout rounds.
- The assignment of the 8 best third-placed teams respects the official eligibility
  lists; FIFA's exact table may differ in edge cases.
- The backtesting uses a standard 32-team bracket and ignores host advantage.

*Just for fun.* ⚽
