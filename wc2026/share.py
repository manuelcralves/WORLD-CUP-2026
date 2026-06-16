"""Sharing: image cards for social media + a comparison page of the versions.

 - share_cards()      : square, branded PNGs (title race; favourite spotlight)
 - comparison_page()  : side-by-side HTML of the live version vs pre-tournament
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .tournament import STAGES, STAGE_LABELS
from .viz import GOLD, GREEN, INK, MUTED, TEXT, flag

FOOT = "World Cup 2026  ·  ML model (Poisson + Elo)  ·  data 1872–2026"


def _card(size=(7.2, 7.2)):
    fig, ax = plt.subplots(figsize=size, dpi=150)
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)
    fig.text(0.5, 0.035, FOOT, ha="center", color=MUTED, fontsize=9)
    return fig, ax


def _style(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=TEXT)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)


def champions_card(table, out, top=10):
    d = table.head(top).iloc[::-1]
    fig, ax = _card()
    lead = float(d["p_champion"].max())  # highlight the favourite
    colors = [GOLD if v == lead else GREEN for v in d["p_champion"]]
    ax.barh(list(d["team"]), d["p_champion"] * 100, color=colors)
    for i, v in enumerate(d["p_champion"] * 100):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", color=TEXT, fontsize=10)
    ax.set_title("Who will win the 2026 World Cup?", fontweight="bold",
                 fontsize=16, pad=16)
    ax.set_xlabel("Probability of winning the title (%)")
    _style(ax)
    fig.subplots_adjust(left=0.28, right=0.95, top=0.9, bottom=0.12)
    fig.savefig(out, facecolor=INK)
    plt.close(fig)
    return out


def spotlight_card(table, out, team=None):
    if team is None:
        team = table.iloc[0]["team"]  # the favourite, by default
    row = table[table["team"] == team].iloc[0]
    rank = int(table.index[table["team"] == team][0]) + 1
    fig, ax = _card()
    ax.axis("off")
    ax.text(0.5, 0.93, team.upper(), ha="center",
            fontsize=28, fontweight="bold", color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.85, f"at the 2026 World Cup  ·  Group {row['group']}", ha="center",
            fontsize=12, color=MUTED, transform=ax.transAxes)
    ax.text(0.5, 0.70, f"{row['p_champion']*100:.1f}%", ha="center", fontsize=58,
            fontweight="bold", color=GOLD, transform=ax.transAxes)
    ax.text(0.5, 0.62, f"to win the title  (#{rank} in the ranking)", ha="center",
            fontsize=13, color=TEXT, transform=ax.transAxes)
    # mini funnel of the path
    y0 = 0.46
    for i, s in enumerate(STAGES):
        p = row[s] * 100
        x = 0.12 + i * 0.13
        ax.add_patch(plt.Rectangle((x, y0), 0.10, 0.10 * p / 100,
                                   color=GOLD, transform=ax.transAxes))
        ax.text(x + 0.05, y0 - 0.03, STAGE_LABELS[s].split(" ")[0], ha="center",
                fontsize=8, color=MUTED, transform=ax.transAxes)
        ax.text(x + 0.05, y0 + 0.10 * p / 100 + 0.01, f"{p:.0f}", ha="center",
                fontsize=8, color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.30, "Probability of reaching each stage", ha="center",
            fontsize=10, color=MUTED, transform=ax.transAxes)
    from .predictions import opponents_for
    od = opponents_for(table, team)
    if od.get("Round of 16", {}).get("opponents"):
        opp_name = od["Round of 16"]["opponents"][0]["team"]
        ax.text(0.5, 0.18, f"Most likely round-of-16 opponent: {opp_name}",
                ha="center", fontsize=12, color=TEXT, transform=ax.transAxes)
    fig.savefig(out, facecolor=INK)
    plt.close(fig)
    return out


def share_cards(table, outdir, team=None):
    outdir = Path(outdir)
    champions_card(table, outdir / "share_title.png")
    spotlight_card(table, outdir / "share_spotlight.png", team)
    return outdir


# --------------------------------------------------------------------------- #
def comparison_page(snap_csv, pre_csv, out_html, top=14):
    snap = pd.read_csv(snap_csv).set_index("team")
    pre = pd.read_csv(pre_csv).set_index("team")
    order = snap.sort_values("p_champion", ascending=False).head(top).index

    rows = ""
    for t in order:
        s = snap.loc[t, "p_champion"] * 100
        p = pre.loc[t, "p_champion"] * 100 if t in pre.index else 0.0
        hl = " style='color:#ffd34d;font-weight:700'" if t == order[0] else ""
        rows += (f"<tr{hl}><td>{flag(t)} {t}</td><td>{s:.1f}%</td>"
                 f"<td>{p:.1f}%</td><td>{s-p:+.1f}</td></tr>")

    css = ("body{background:#0f1420;color:#e6e9ef;font-family:-apple-system,"
           "Segoe UI,Roboto,Arial,sans-serif;max-width:680px;margin:0 auto;"
           "padding:30px 18px}h1{font-size:24px}table{width:100%;border-collapse:"
           "collapse}td,th{padding:8px;border-bottom:1px solid #232c40;text-align:"
           "left;font-size:14px}th{color:#8a93a8}.sub{color:#8a93a8}")
    html = (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>World Cup 2026 — comparison</title><style>{css}</style></head><body>"
        f"<h1>🏆 World Cup 2026 — live vs. pre-tournament</h1>"
        f"<p class='sub'>Probability of winning the title: the <b>live</b> version (with "
        f"the results already known) vs. the <b>pre-tournament</b> prediction (blind). "
        f"The Δ column shows the effect of the matches already played.</p>"
        f"<table><tr><th>Team</th><th>Live</th><th>Pre-tournament</th><th>Δ</th></tr>"
        f"{rows}</table>"
        f"<p class='sub'>Open the full dashboards: "
        f"<a style='color:#00d68f' href='outputs/dashboard.html'>live</a> · "
        f"<a style='color:#00d68f' href='outputs_pretournament/dashboard.html'>"
        f"pre-tournament</a>.</p></body></html>"
    )
    Path(out_html).write_text(html, encoding="utf-8")
    return out_html


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    from . import data as D, model as M, tournament as T
    b = D.load_all()
    tr = M.train_full(b)
    table = T.simulate(b, tr, n_sims=20000)
    Path("outputs").mkdir(exist_ok=True)
    share_cards(table, "outputs")
    print("Cards: outputs/share_title.png, outputs/share_spotlight.png")
    import os
    if os.path.exists("outputs_pretournament/predictions.csv"):
        comparison_page("outputs/predictions.csv",
                        "outputs_pretournament/predictions.csv", "compare.html")
        print("Comparison: compare.html")
