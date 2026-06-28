"""Full pipeline for the 2026 World Cup.

    python run_pipeline.py [n_simulations] [model] [mode]

  model: poisson (default) | xgboost | ensemble
  mode:  snapshot (live, default) | pretournament (pre-tournament) | both

For each version it generates: interactive dashboard, charts, Elo GIF and CSVs.
  - snapshot      -> outputs/            (fixes the 12 matches already played)
  - pretournament -> outputs_pretournament/ (trains only on data before 11 Jun)

The arguments may come in any order (e.g. `run_pipeline.py both 50000`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # windowless backend (runs in terminal/headless)
import pandas as pd

from wc2026 import data as D
from wc2026 import model as M
from wc2026 import validate as V
from wc2026 import viz
from wc2026 import tournament as T
from wc2026 import predictions as PR
from wc2026 import backtest as BT
from wc2026 import backtest_all as BT_ALL
from wc2026 import dashboard as DASH
from wc2026 import tracker as TRK
from wc2026 import richdata as RICH
from wc2026 import goldenboot as GB
from wc2026 import share as SH

BASE = Path(__file__).resolve().parent


def _train(bundle, model):
    if model in ("ensemble", "xgboost"):
        from wc2026 import xgb_model as XGB
        return (XGB.train_ensemble_full(bundle) if model == "ensemble"
                else XGB.train_xgb_full(bundle))
    return M.train_full(bundle)


def _run(n_sims, model, cutoff, out_dir, label, backtests, mega, review):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {label} =====")
    bundle = D.load_all(cutoff=cutoff)
    if cutoff:
        print(f"  Training only on data before {cutoff}; "
              f"simulating the {len(bundle['wc_remaining'])} group-stage matches from scratch.")
    else:
        print(f"  {len(bundle['wc_played'])} matches fixed, "
              f"{len(bundle['wc_remaining'])} to simulate.")

    trained = _train(bundle, model)
    val = V.evaluate(bundle)
    fp = RICH.load_rich(BASE / "api_cache").get("fairplay", {}) if cutoff is None else {}
    table = T.simulate(bundle, trained, n_sims=n_sims, fairplay=fp)

    # Tracker (live version only): records the snapshot, the odds history and
    # the Golden Boot projection history (both filled by the same per-day backfill).
    evolution, odds_history, golden_history = None, None, None
    if cutoff is None:
        hist = out_dir / "history.csv"
        ghist = out_dir / "golden_history.csv"
        h0, gh0 = TRK.load_history(hist), TRK.load_history(ghist)
        if (h0.empty or h0["date"].nunique() < 3
                or gh0.empty or gh0["date"].nunique() < 3):   # one-time backfill if sparse
            TRK.backfill_history(hist, golden_path=ghist, n_sims=12000)
        asof = TRK.data_asof(bundle)
        TRK.record_snapshot(table, asof, hist)
        TRK.record_golden_snapshot(GB.predict(bundle, table, topn=100), asof, ghist)  # deep enough that a later top-6 riser still has back-history
        chart = TRK.evolution_chart(hist, out_dir / "odds_evolution.png")
        evolution = {"movers": TRK.movers(hist), "has_chart": chart is not None}
        odds_history = TRK.history_series(hist)
        golden_history = TRK.golden_history_series(ghist)

    viz.save_charts(table, out_dir)
    SH.share_cards(table, out_dir)
    viz.elo_race_gif(bundle["matches"], out_dir / "elo_race.gif")
    PR.group_stage_predictions(bundle, trained).to_csv(
        out_dir / "group_matches.csv", index=False, encoding="utf-8")
    PR.expected_standings(table).to_csv(
        out_dir / "expected_standings.csv", index=False, encoding="utf-8")
    data = DASH.collect(bundle, trained, table, val, backtests,
                        gb_before=cutoff, mode_label=label, evolution=evolution,
                        odds_history=odds_history, golden_history=golden_history,
                        mega_backtest=mega,
                        # "results so far" only makes sense for the live version
                        played_review=(review if cutoff is None else None))
    DASH.build_interactive(data, out_dir / "dashboard.html")
    pd.DataFrame(data["golden_boot"]).to_csv(
        out_dir / "golden_boot.csv", index=False, encoding="utf-8")
    table.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")
    # Most likely knockout opponents per team -> opponents.csv (for the tweet kit).
    # Cheap: reads the precomputed sim matrices in table.attrs, no re-simulation.
    opp_rows = [{"team": tm, "round": rnd, "p_reach": info["p_reach"],
                 "opponent": o["team"], "p_cond": o["p_cond"]}
                for tm in table["team"]
                for rnd, info in PR.opponents_for(table, tm).items()
                for o in info["opponents"]]
    if opp_rows:
        pd.DataFrame(opp_rows).to_csv(
            out_dir / "opponents.csv", index=False, encoding="utf-8")
    # Knockout fixtures (once the groups are done) -> knockout_matches.csv for the
    # tweet kit (KO previews + the R32 reveal). Reuses data["matches"]; empty in groups.
    ko_rows = [{"date": m["date"], "home": m["home"], "away": m["away"],
                "p_home": m["p_home"], "p_draw": m["p_draw"], "p_away": m["p_away"],
                "ml_score": m["ml_score"],
                "top3": ",".join(f"{s['score']}:{round(s['p'] * 100)}" for s in m.get("top", []))}
               for m in data["matches"] if m.get("stage") == "knockout"]
    if ko_rows:
        pd.DataFrame(ko_rows).to_csv(
            out_dir / "knockout_matches.csv", index=False, encoding="utf-8")
    if cutoff is None and data.get("played_review"):   # for the daily tweet (recap/upsets) -- incl. knockouts
        pd.DataFrame(data["played_review"]).drop(columns=["goals"], errors="ignore").to_csv(
            out_dir / "played_review.csv", index=False, encoding="utf-8")
    if cutoff is None:                           # push the match table for the leaderboard
        from wc2026 import supa as SUPA
        SUPA.push_matches(data)

    fav, sec = table.iloc[0], table.iloc[1]
    print(f"  Favourite: {fav['team']} {fav['p_champion']*100:.1f}% · "
          f"{sec['team']} {sec['p_champion']*100:.1f}%")
    print(f"  -> {out_dir.name}/dashboard.html")
    return table


def main(n_sims: int = 1000000, model: str = "poisson", mode: str = "snapshot"):
    print("Preparing backtesting (champion-rank + walk-forward skill test)...")
    live_bundle = D.load_all()
    backtests = [BT.summary(live_bundle, y) for y in (2018, 2022)]
    mega = BT_ALL.run_all(live_bundle)
    # blind pre-tournament model -> predicted-vs-actual for the games already played
    pre_trained = _train(D.load_all(cutoff="2026-06-11"), model)
    review = PR.played_review(live_bundle, pre_trained)

    runs = []
    if mode in ("snapshot", "both"):
        runs.append(("Live version (matches already played are fixed)",
                     None, BASE / "outputs"))
    if mode in ("pretournament", "both"):
        runs.append(("Pre-tournament version (no 2026 results known)",
                     "2026-06-11", BASE / "outputs_pretournament"))

    for label, cutoff, out_dir in runs:
        _run(n_sims, model, cutoff, out_dir, label, backtests, mega, review)

    snap = BASE / "outputs" / "predictions.csv"
    pre = BASE / "outputs_pretournament" / "predictions.csv"
    if snap.exists() and pre.exists():
        SH.comparison_page(snap, pre, BASE / "compare.html")
        print("Comparison page: compare.html")
    print("\nDone.")


if __name__ == "__main__":
    a = sys.argv[1:]
    n = next((int(x) for x in a if x.isdigit()), 1000000)
    mdl = next((x for x in a if x in ("poisson", "xgboost", "ensemble")), "poisson")
    md = next((x for x in a if x in ("snapshot", "pretournament", "both")), "snapshot")
    main(n, mdl, md)
