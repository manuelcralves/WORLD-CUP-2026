"""Charts (matplotlib): the title-race ranking, a team's path, and the Elo bar
chart race (GIF). The highlight colour marks the favourite / selected team.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
# Note: the backend is not forced here (so the notebook can show inline).
# The run_pipeline.py script selects the "Agg" backend to run without a window.
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .tournament import OFFICIAL_GROUPS, STAGE_LABELS, STAGES

GOLD = "#ffd34d"      # highlight (favourite / selected team)
GREEN = "#00d68f"     # accent color
INK = "#11161f"
PANEL = "#1a2233"
TEXT = "#e6e9ef"
MUTED = "#8a93a8"

# Flags (emoji) of the 48 teams, with names exactly as in the dataset.
FLAGS = {
    "Mexico": "🇲🇽", "South Korea": "🇰🇷", "South Africa": "🇿🇦",
    "Czech Republic": "🇨🇿", "Canada": "🇨🇦", "Switzerland": "🇨🇭",
    "Qatar": "🇶🇦", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Morocco": "🇲🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Haiti": "🇭🇹",
    "United States": "🇺🇸", "Australia": "🇦🇺", "Paraguay": "🇵🇾",
    "Turkey": "🇹🇷", "Germany": "🇩🇪", "Ecuador": "🇪🇨", "Ivory Coast": "🇨🇮",
    "Curaçao": "🇨🇼", "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Tunisia": "🇹🇳",
    "Sweden": "🇸🇪", "Belgium": "🇧🇪", "Iran": "🇮🇷", "Egypt": "🇪🇬",
    "New Zealand": "🇳🇿", "Spain": "🇪🇸", "Uruguay": "🇺🇾",
    "Saudi Arabia": "🇸🇦", "Cape Verde": "🇨🇻", "France": "🇫🇷",
    "Senegal": "🇸🇳", "Norway": "🇳🇴", "Iraq": "🇮🇶", "Argentina": "🇦🇷",
    "Austria": "🇦🇹", "Algeria": "🇩🇿", "Jordan": "🇯🇴", "Portugal": "🇵🇹",
    "Colombia": "🇨🇴", "Uzbekistan": "🇺🇿", "DR Congo": "🇨🇩",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Panama": "🇵🇦", "Ghana": "🇬🇭",
}


def flag(team: str) -> str:
    return FLAGS.get(team, "🏳️")


# ISO 3166-1 alpha-2 codes (for crisp flag images via flagcdn.com).
CODES = {
    "Mexico": "mx", "South Korea": "kr", "South Africa": "za",
    "Czech Republic": "cz", "Canada": "ca", "Switzerland": "ch", "Qatar": "qa",
    "Bosnia and Herzegovina": "ba", "Brazil": "br", "Morocco": "ma",
    "Scotland": "gb-sct", "Haiti": "ht", "United States": "us", "Australia": "au",
    "Paraguay": "py", "Turkey": "tr", "Germany": "de", "Ecuador": "ec",
    "Ivory Coast": "ci", "Curaçao": "cw", "Netherlands": "nl", "Japan": "jp",
    "Tunisia": "tn", "Sweden": "se", "Belgium": "be", "Iran": "ir", "Egypt": "eg",
    "New Zealand": "nz", "Spain": "es", "Uruguay": "uy", "Saudi Arabia": "sa",
    "Cape Verde": "cv", "France": "fr", "Senegal": "sn", "Norway": "no",
    "Iraq": "iq", "Argentina": "ar", "Austria": "at", "Algeria": "dz",
    "Jordan": "jo", "Portugal": "pt", "Colombia": "co", "Uzbekistan": "uz",
    "DR Congo": "cd", "England": "gb-eng", "Croatia": "hr", "Panama": "pa",
    "Ghana": "gh",
}


def flag_code(team: str) -> str:
    return CODES.get(team, "")


# --------------------------------------------------------------------------- #
# matplotlib charts
# --------------------------------------------------------------------------- #
def _style(ax):
    ax.set_facecolor(INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def champion_chart(table, top: int = 15):
    d = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    fig.patch.set_facecolor(INK)
    lead = float(d["p_champion"].max())  # highlight the favourite (data-driven)
    colors = [GOLD if v == lead else GREEN for v in d["p_champion"]]
    # no emoji in the labels: the matplotlib font has no flags
    ax.barh(list(d["team"]), d["p_champion"] * 100, color=colors)
    for y, v in enumerate(d["p_champion"] * 100):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", color=TEXT, fontsize=9)
    ax.set_xlabel("Probability of winning the title (%)")
    ax.set_title("FIFA World Cup 2026 — title contenders", fontweight="bold")
    _style(ax)
    fig.tight_layout()
    return fig


def path_chart(table, team=None):
    if team is None:
        team = table.iloc[0]["team"]  # the favourite, by default
    row = table[table["team"] == team].iloc[0]
    vals = [row[s] * 100 for s in STAGES]
    labels = [STAGE_LABELS[s] for s in STAGES]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    fig.patch.set_facecolor(INK)
    ax.bar(labels, vals, color=GOLD)
    for x, v in enumerate(vals):
        ax.text(x, v + 1, f"{v:.0f}%", ha="center", color=TEXT, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"{team}'s path", fontweight="bold")
    _style(ax)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig


def save_charts(table, outdir: Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fig in [("champions", champion_chart(table)),
                      ("path", path_chart(table))]:
        p = outdir / f"{name}.png"
        fig.savefig(p, facecolor=INK, bbox_inches="tight")
        plt.close(fig)
        paths[name] = p
    return paths


def elo_race_gif(matches, out_path, start_year=1960, top=10, fps=5):
    """Bar chart race (GIF) of the top teams by Elo throughout history."""
    from .elo import compute_elo
    _, _, hist = compute_elo(matches, keep_history=True)
    hist = hist.sort_values("date")
    frames = []
    for y in range(start_year, 2027):
        snap = hist[hist["date"] <= pd.Timestamp(f"{y}-12-31")]
        if snap.empty:
            continue
        s = snap.groupby("team")["rating"].last().sort_values(ascending=False).head(top)
        frames.append((y, s.iloc[::-1]))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=100)

    def draw(i):
        ax.clear()
        y, s = frames[i]
        lead = float(s.max())  # highlight the current #1 each year
        colors = [GOLD if v == lead else GREEN for v in s.values]
        ax.barh(list(s.index), list(s.values), color=colors)
        for j, v in enumerate(s.values):
            ax.text(v - 6, j, f"{int(v)}", va="center", ha="right",
                    color=INK, fontsize=9, fontweight="bold")
        ax.set_title(f"Top teams by Elo — {y}", color=TEXT,
                     fontweight="bold", fontsize=14)
        ax.set_xlim(1400, max(2300, float(s.max()) * 1.03))
        fig.patch.set_facecolor(INK)
        _style(ax)

    anim = animation.FuncAnimation(fig, draw, frames=len(frames),
                                   interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps),
              savefig_kwargs={"facecolor": INK})
    plt.close(fig)
    return out_path
