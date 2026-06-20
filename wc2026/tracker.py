"""History of predictions over the tournament + odds evolution.

Each run records a "snapshot" of the probabilities, dated by the
**data date** (the most recent World Cup match with a result). This way, as
the dataset is updated, a time series accumulates and we can:
 - plot the evolution of each team's title probability;
 - summarise what changed the most between the two latest updates.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import viz

KEYS = ["p_champion", "p_ko", "p_final"]


def data_asof(bundle) -> str:
    """Reference date: the most recent World Cup match with a result."""
    wc = bundle["wc"]
    played = wc[wc["home_score"].notna()]
    d = played["date"].max() if len(played) else bundle["played"]["date"].max()
    return str(pd.Timestamp(d).date())


def record_snapshot(table: pd.DataFrame, asof: str, path) -> Path:
    """Append (or replace) this date's snapshot to the history."""
    path = Path(path)
    snap = table[["team"] + KEYS].copy()
    snap.insert(0, "date", asof)
    if path.exists():
        hist = pd.read_csv(path)
        hist = hist[hist["date"] != asof]           # idempotent by date
        snap = pd.concat([hist, snap], ignore_index=True)
    snap.to_csv(path, index=False, encoding="utf-8")
    return path


def load_history(path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def movers(path, n: int = 6) -> list[dict]:
    """Largest title-probability changes between the 2 latest dates."""
    hist = load_history(path)
    if hist.empty:
        return []
    dates = sorted(hist["date"].unique())
    if len(dates) < 2:
        return []
    a = hist[hist.date == dates[-2]].set_index("team")["p_champion"]
    b = hist[hist.date == dates[-1]].set_index("team")["p_champion"]
    delta = (b - a).dropna()
    delta = delta.reindex(delta.abs().sort_values(ascending=False).index).head(n)
    return [{"team": t, "prev": float(a[t]), "now": float(b[t]),
             "delta": float(b[t] - a[t])} for t in delta.index]


def history_series(path, top: int = 8) -> dict:
    """Top teams' title-probability series over time, for the interactive chart."""
    hist = load_history(path)
    if hist.empty:
        return None
    dates = sorted(hist["date"].unique())
    piv = hist.pivot_table(index="date", columns="team",
                           values="p_champion").reindex(dates)
    leaders = piv.iloc[-1].sort_values(ascending=False).head(top).index
    return {"dates": list(dates),
            "series": [{"team": t,
                        "data": [round(float(piv.loc[d, t]), 4) for d in dates]}
                       for t in leaders]}


def record_golden_snapshot(gb: pd.DataFrame, asof: str, path) -> Path:
    """Append (or replace) this date's Golden Boot projections to the history."""
    path = Path(path)
    snap = pd.DataFrame({"date": asof, "scorer": gb["scorer"].values,
                         "proj": gb["proj_goals"].values})
    if path.exists():
        hist = pd.read_csv(path)
        hist = hist[hist["date"] != asof]            # idempotent by date
        snap = pd.concat([hist, snap], ignore_index=True)
    snap.to_csv(path, index=False, encoding="utf-8")
    return path


def golden_history_series(path, top: int = 6) -> dict:
    """Top scorers' projected-total series over time, for the interactive chart."""
    hist = load_history(path)
    if hist.empty or hist["date"].nunique() < 2:
        return None
    dates = sorted(hist["date"].unique())
    piv = hist.pivot_table(index="date", columns="scorer", values="proj").reindex(dates)
    leaders = piv.iloc[-1].sort_values(ascending=False).head(top).index
    return {"dates": list(dates),
            "series": [{"scorer": s,
                        "data": [None if pd.isna(piv.loc[d, s])
                                 else round(float(piv.loc[d, s]), 2) for d in dates]}
                       for s in leaders]}


def backfill_history(history_path, golden_path=None, n_sims: int = 15000) -> Path:
    """Rebuild history.csv with one snapshot per as-of date: the eve of the
    tournament plus each World Cup match-day we already have results for.

    This pre-populates the odds-over-time chart so it looks alive from day one.
    """
    from . import data as D, model as M, tournament as T
    full = D.load_all()
    played = full["wc"][full["wc"]["home_score"].notna()]
    if played.empty:
        return Path(history_path)
    days = sorted(played["date"].dt.normalize().unique())
    eve = pd.Timestamp(days[0]) - pd.Timedelta(days=1)
    asof_dates = [eve] + list(days)

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gpath = Path(golden_path) if golden_path else None
    if gpath:
        from . import goldenboot as GB
        if gpath.exists():
            gpath.unlink()
    for d in asof_dates:
        label = str(pd.Timestamp(d).date())
        b = D.load_all(asof=label)
        table = T.simulate(b, M.train_full(b), n_sims=n_sims)
        record_snapshot(table, label, path)
        if gpath:
            record_golden_snapshot(GB.predict(b, table, topn=100), label, gpath)  # keep back-history for players who later climb into the top 6
    return path


def evolution_chart(path, out_png, top: int = 8, highlight=None):
    """Time line of title probabilities (top-N + highlight of the favourite)."""
    hist = load_history(path)
    dates = sorted(hist["date"].unique()) if not hist.empty else []
    if len(dates) < 2:
        return None  # needs at least 2 snapshots

    piv = hist.pivot_table(index="date", columns="team",
                           values="p_champion").reindex(dates)
    latest = piv.iloc[-1].sort_values(ascending=False)
    if highlight is None:
        highlight = latest.index[0]  # the current favourite (data-driven)
    teams = list(latest.head(top).index)
    if highlight not in teams:
        teams.append(highlight)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    fig.patch.set_facecolor(viz.INK)
    for t in teams:
        if t not in piv.columns:
            continue
        is_pt = t == highlight
        ax.plot(range(len(dates)), piv[t].values * 100,
                marker="o", linewidth=3 if is_pt else 1.8,
                color=viz.GOLD if is_pt else None,
                label=f"{viz.flag(t)} {t}", zorder=3 if is_pt else 2)
        ax.text(len(dates) - 1 + 0.05, piv[t].iloc[-1] * 100,
                f" {t}", va="center", fontsize=8,
                color=viz.GOLD if is_pt else viz.TEXT)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[5:] for d in dates], rotation=0)
    ax.set_ylabel("Title probability (%)")
    ax.set_title("Title probability over time", color=viz.TEXT,
                 fontweight="bold")
    viz._style(ax)
    ax.grid(axis="y", color="#232c40", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=viz.INK, bbox_inches="tight")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    # Seed: generates 2 real snapshots (14 Jun backup + current 15 Jun data)
    import matplotlib
    matplotlib.use("Agg")
    from . import data as D, model as M, tournament as T

    hist_path = Path("outputs/history.csv")
    if hist_path.exists():
        hist_path.unlink()
    for data_dir in ["_backup", "."]:
        b = D.load_all(data_dir=data_dir)
        tr = M.train_full(b)
        table = T.simulate(b, tr, n_sims=20000)
        asof = data_asof(b)
        record_snapshot(table, asof, hist_path)
        print(f"Snapshot {asof}: favourite {table.iloc[0]['team']} "
              f"{table.iloc[0]['p_champion']*100:.1f}%")
    evolution_chart(hist_path, "outputs/odds_evolution.png")
    print("\nWhat changed:")
    for m in movers(hist_path):
        print(f"  {m['team']:<12} {m['prev']*100:4.1f}% -> {m['now']*100:4.1f}% "
              f"({m['delta']*100:+.1f})")
