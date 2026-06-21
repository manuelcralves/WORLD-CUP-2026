<div align="center">

# 🏆 Who Wins the World Cup 2026?

[![▶ Live demo](https://img.shields.io/badge/▶_live_demo-online-00e0a4?style=for-the-badge)](https://manuelcralves.github.io/WORLD-CUP-2026/)

![banner](https://manuelcralves.github.io/WORLD-CUP-2026/outputs/share_card.png)

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Model](https://img.shields.io/badge/RPS-0.17%20(bookmaker%20level)-orange)
![Data](https://img.shields.io/badge/data-49k%20matches%20since%201872-lightgrey)
![Deploy](https://img.shields.io/badge/deploy-every%202h%20·%20GitHub%20Actions-9cf)

**A machine-learning model — Elo + Dixon-Coles Poisson + Monte Carlo — trained on
150 years of international football, with a rich interactive dashboard. Just for fun.**

[**🔴 Live dashboard**](https://manuelcralves.github.io/WORLD-CUP-2026/outputs/dashboard.html) ·
[**🔮 Pre-tournament forecast**](https://manuelcralves.github.io/WORLD-CUP-2026/outputs_pretournament/dashboard.html) ·
[**⚖️ Compare the two**](https://manuelcralves.github.io/WORLD-CUP-2026/compare.html)

</div>

---

## How it works

Three pieces at the core of the model:

1. **Elo ratings** — each team's strength, updated match by match since 1872.
2. **Poisson model (Dixon-Coles)** — learns, by maximum likelihood, how many goals
   each team scores given the Elo, recent form and home advantage. From that come
   the probabilities of every exact scoreline.
3. **Monte Carlo simulation** — fixes the matches already played and runs the
   tournament **a million times**: the 12 groups with FIFA tiebreakers, the
   8 best third-placed teams (official eligibility lists) and the **official 2026
   bracket**, with extra time and **penalties modelled from 677 real shootouts**.

It runs in **two versions** that tell different stories:

- **🔴 Live** → uses the World Cup results already known as fixed and predicts the
  rest. The "as it stands" picture, **rebuilt automatically every 2 hours**.
- **🔮 Pre-tournament** → trained **only on data before 11 Jun 2026**, simulating all
  72 group matches from scratch with zero knowledge of any 2026 result. The "blind" call.

## ✨ What's inside the dashboard

A single, self-contained dark dashboard (no backend), organised into four tabs:

| Tab | Highlights |
|-----|-----------|
| **Overview** | Title odds + **odds-over-time** chart · today's matches with the model's pick (**click a played game** for its full match detail) · biggest surprises so far · the full **connected bracket** |
| **Teams** | Sortable ranking — **click any team** for its path to the final, round-by-round opponents, **squad** and history · all 12 groups (cards break ties via FIFA fair-play) · every match with **kickoff times (WEST)** |
| **Play** | 🆚 **Match Lab** (any head-to-head: win odds, top scorelines, betting-style markets, recent meetings) · 🏆 **Build your World Cup** (an interactive knockout bracket you control — the model seeds every tie, you override any result) · a **what-if group editor** (set real scorelines, watch qualification change) · 🎲 roll a full tournament |
| **Predict** | 🎯 **Beat the Machine** — predict every scoreline, climb a **live leaderboard** against other players and the model, and fill your own **knockout bracket** (Supabase-backed) |
| **Insights** | ✅ **Predicted vs actual** (how the model is doing) · 🏅 **Model vs FIFA ranking** (Spearman 0.93) · 👟 **Golden Boot** race · ⚠️ **discipline** (cards) ranking · goal-timeline analysis · **Elo through history** (bar-chart race) · historical facts |

Every played game opens a **live match-detail modal** (Highlightly API): a two-sided
**timeline** (goals · cards · subs), **line-ups** annotated with cards & substitutions,
and **match stats** (possession, shots, xG…).

Plus: rich **link previews** (Open Graph / Twitter cards), **add-to-home-screen** (PWA),
team **deep-links** (`?team=Portugal`) and a mobile-friendly responsive layout.

<div align="center">

*Elo through 150 years of football&nbsp;·&nbsp;and how the title odds move as results come in*

<img src="https://manuelcralves.github.io/WORLD-CUP-2026/outputs/elo_race.gif" width="49%" alt="Elo bar-chart race through history">
<img src="https://manuelcralves.github.io/WORLD-CUP-2026/outputs/odds_evolution.png" width="49%" alt="Title odds over time">

</div>

## Model quality

**Validation** (temporal holdout, train < 2022, test 2022–2026, ~4,500 unseen
matches): ~**60% accuracy** and **RPS ≈ 0.17** — bookmaker level, well above the
naive base rate (RPS 0.23). **Calibration** is almost perfect: when the model says
70%, it happens ~74% of the time.

**Walk-forward backtest** — the honest test: for every past tournament, retrain on
*only* the data available beforehand, then predict. Across **67 editions / 2,669
matches**, the model calls **55.8%** of results correctly vs **44.2%** for the
"always pick the higher seed" baseline.

**Champion backtest** (trained only on pre-tournament data):

| World Cup | Actual champion | Model's rank | Match accuracy |
|-----------|-----------------|-------------:|---------------:|
| 2018 | 🇫🇷 France | #6 (4.9%) | 58% (vs 39% naive) |
| 2022 | 🇦🇷 Argentina | #2 (24%) | 50% (vs 44% naive) |

The model always beats the baseline and ranks the eventual champions among the
contenders — but tournaments are chaotic (France 2018 surprised almost every model).

**Poisson vs XGBoost** (same holdout): essentially tied (RPS 0.1717 vs 0.1718), and
the **ensemble** is marginally best (0.1714). With few highly informative features,
gradient boosting doesn't beat the statistical model. Poisson stays the default
(fast, no extra deps).

## How to run

Requirements: `numpy`, `pandas`, `scipy`, `matplotlib`, `Pillow`. XGBoost /
scikit-learn are optional (only for the `ensemble` / `xgboost` model).

```bash
pip install -r requirements.txt

python run_pipeline.py                  # live version, 1,000,000 simulations
python run_pipeline.py 1000000 both     # BOTH versions (live + pre-tournament)
python run_pipeline.py pretournament    # pre-tournament only (blind)
```

Arguments can come in any order (number of simulations, `poisson`/`xgboost`/
`ensemble`, and `snapshot`/`pretournament`/`both`). On Windows, double-click
`update_and_run.bat` to update the data and rebuild both versions in one go.

### Updating the data

The CSVs come from the live **[martj42/international_results](https://github.com/martj42/international_results)**
repository (the same source as Kaggle), updated continuously:

```bash
python update_data.py            # download the 4 latest CSVs (backs up to _backup/)
python run_pipeline.py both      # refresh the predictions with the new results
```

### Publishing it online (auto-deploy)

The repo ships with a **GitHub Actions** workflow (`.github/workflows/deploy.yml`)
that **every 2 hours** fetches the latest results (martj42 + the **Highlightly API** for
live match detail, scores and goals), re-runs both versions and publishes the site to
GitHub Pages — no manual step, and your PC doesn't need to be on. It also commits the
refreshed data and `history.csv` back, so the odds-over-time chart keeps accumulating. To enable: push to GitHub, then **Settings → Pages → Source: GitHub
Actions**. The site goes live at `https://<user>.github.io/<repo>/`.

## Project structure

```
wc2026/
  data.py         loading, name normalisation, group reconstruction
  elo.py          chronological Elo ratings
  features.py     feature engineering (no leakage)
  model.py        Poisson model (Dixon-Coles) + training + scoreline grid
  validate.py     temporal validation (vs baselines) + calibration
  penalties.py    penalty-shootout model (from 677 real shootouts)
  tournament.py   Monte Carlo simulation (groups + official 2026 knockouts)
  predictions.py  likely scores, match-by-match, standings, bracket, H2H
  backtest.py     champion backtest on the 2018 & 2022 World Cups
  backtest_all.py walk-forward skill test across every past tournament
  fifa.py         official FIFA ranking + model-vs-ranking comparison
  schedule.py     official 2026 kickoff times (shown in WEST)
  goldenboot.py   top-scorer (Golden Boot) forecast
  goals.py        goal-timeline analysis (from goalscorers.csv)
  richdata.py     Highlightly rich data (match modal, squads, cards) → D.rich
  supa.py         Supabase match-table push (live "Beat the Machine" leaderboard)
  tracker.py      prediction history + odds over time during the tournament
  facts.py        historical facts (team cards, upsets, penalties)
  analysis.py     group of death, dark horses, likely finals, confidence
  viz.py          matplotlib charts + Elo bar-chart race (GIF) + favicon
  share.py        social cards (hero + spotlight) + comparison page
  dashboard.py    the interactive HTML dashboard (all four tabs)
run_pipeline.py    end-to-end pipeline (live + pre-tournament)
update_data.py     updates the CSVs from the live source (martj42)
fetch_wc_data.py   Highlightly rich-data fetcher (cached · quota-safe · self-healing)
feed_highlightly.py / feed_goals_highlightly.py   auto-feed WC results & goal scorers
reconcile_highlightly.py   daily dataset ↔ Highlightly sanity check (→ data_check.txt)
index.html         landing page (links to both dashboards) — for GitHub Pages
make_icons.py      generates the PWA app icons · manifest.json
.github/workflows/deploy.yml   daily auto-deploy to GitHub Pages
```

Each output folder (`outputs/`, `outputs_pretournament/`) gets a self-contained
`dashboard.html` plus the share cards (`share_card.png`, `share_spotlight.png`),
charts (`champions.png`, `elo_race.gif`, `odds_evolution.png`) and the data CSVs
(`predictions.csv`, `group_matches.csv`, `golden_boot.csv`, `history.csv`). In
`both` mode the root also gets `compare.html`.

## Limitations

- A **probabilistic, data-driven** model: there is always huge uncertainty in a
  knockout tournament.
- Host advantage is halved in the knockout rounds.
- The assignment of the 8 best third-placed teams respects the official eligibility
  lists; FIFA's exact table may differ in edge cases.
- The walk-forward backtest uses each era's standard bracket and ignores host advantage.

---

<div align="center">

*Data-driven, just for fun.* ⚽ · Data: [martj42/international_results](https://github.com/martj42/international_results)
(150 years of international football)

**[▶ Open the live dashboard](https://manuelcralves.github.io/WORLD-CUP-2026/)**

</div>
