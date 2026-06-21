"""Sharing: image cards for social media + a comparison page of the versions.

 - share_cards()      : square, branded PNGs (title race; favourite spotlight)
 - comparison_page()  : side-by-side HTML of the live version vs pre-tournament
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .tournament import STAGES, STAGE_LABELS
from .viz import CODES, FAVICON, GOLD, GREEN, INK, MUTED, SITE, TEXT, flag


def _flag_img(team: str, h: int = 13) -> str:
    """flagcdn <img> (emoji flags render as 'AR'/'ES' text on Windows)."""
    code = CODES.get(team, "")
    if not code:
        return "🏳️"
    return (f"<img src='https://flagcdn.com/w40/{code}.png' alt='' "
            f"style='width:{round(h*1.4)}px;height:{h}px;border-radius:2px;"
            f"object-fit:cover;vertical-align:middle;margin-right:7px'>")

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


# --------------------------------------------------------------------------- #
# Hero share card (1200x630) — a branded, homepage-style image for rich link
# previews (WhatsApp / Twitter / Facebook), rather than a bare bar chart.
# Drawn with Pillow; text uses the DejaVu Sans that matplotlib bundles, so it
# renders the same locally (Windows) and in the GitHub Action (Ubuntu).
# --------------------------------------------------------------------------- #
_HERO = (1200, 630)
_GOLD = (255, 211, 77)
_GREEN = (0, 224, 164)
_WHITE = (238, 241, 247)
_GREY = (139, 149, 171)
_DARK = (12, 16, 24)


def _ttf(size: int, bold: bool = True):
    fp = _fm.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return ImageFont.truetype(_fm.findfont(fp), size)


def _hero_bg(w: int, h: int) -> Image.Image:
    """Dark vertical gradient with soft radial glows (the homepage's backdrop)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (yy / h)[..., None]
    img = np.array([12, 16, 24], np.float32) * (1 - t) + np.array([7, 10, 17], np.float32) * t

    def glow(cx, cy, rx, ry, col, peak):
        dd = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        return (np.clip(1 - dd, 0, 1) ** 1.7 * peak)[..., None] * np.array(col, np.float32)

    img = img + glow(600, -40, 660, 380, _GREEN, 0.17)      # green, top-centre
    img = img + glow(1080, 20, 560, 400, _GOLD, 0.09)       # gold, top-right
    img = img + glow(130, 260, 560, 430, (91, 141, 239), 0.07)  # blue, left
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"), "RGB")


def _hero_trophy(d: ImageDraw.ImageDraw, left: float, top: float, px: float):
    """The gold globe-trophy (same drawing as the favicon / app icons)."""
    sc = px / 64.0

    def R(a, b, c, e):
        return [left + a * sc, top + b * sc, left + c * sc, top + e * sc]

    cx, cy, r = 32, 24, 13.5
    lw = max(1, round(1.7 * sc))
    d.ellipse(R(cx - r, cy - r, cx + r, cy + r), fill=_GOLD)
    d.line(R(cx - r, cy, cx + r, cy), fill=_DARK, width=lw)
    d.ellipse(R(cx - r, cy - r * 0.5, cx + r, cy + r * 0.5), outline=_DARK, width=lw)
    d.line(R(cx, cy - r, cx, cy + r), fill=_DARK, width=lw)
    d.ellipse(R(cx - r * 0.5, cy - r, cx + r * 0.5, cy + r), outline=_DARK, width=lw)
    d.rectangle(R(30.5, 37, 33.5, 46), fill=_GOLD)
    d.rounded_rectangle(R(23, 46, 41, 50), radius=2 * sc, fill=_GOLD)
    d.rounded_rectangle(R(18, 50, 46, 55), radius=2.5 * sc, fill=_GOLD)


def _hero_pill(d, cx, cy, text, font, textcol, bg, border, h=44, padx=20):
    w = d.textlength(text, font=font) + padx * 2
    x0, y0 = cx - w / 2, cy - h / 2
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=h / 2,
                        fill=bg, outline=border, width=2)
    d.text((cx, cy), text, font=font, fill=textcol, anchor="mm")
    return w


def hero_card(table, out):
    """A 1200x630 branded card (the homepage in a single image) for link previews."""
    W, H = _HERO
    img = _hero_bg(W, H)
    d = ImageDraw.Draw(img)

    _hero_trophy(d, 600 - 32 * (118 / 64), 28, 118)         # centred trophy
    _hero_pill(d, 600, 170, "MACHINE-LEARNING PREDICTION", _ttf(19),
               _GREEN, (14, 33, 30), (0, 92, 76), h=38)
    fh = _ttf(62)
    d.text((600, 238), "Who wins the", font=fh, fill=_WHITE, anchor="mm")
    d.text((600, 306), "World Cup 2026?", font=fh, fill=_GOLD, anchor="mm")
    d.text((600, 372), "A machine-learning model — Poisson + Elo, 150 years of football",
           font=_ttf(25, bold=False), fill=_GREY, anchor="mm")

    # top-3 contenders as pills (a hook, not a chart) — favourite in gold
    fp = _ttf(26)
    labels = [f"{r['team']}  {r['p_champion'] * 100:.0f}%"
              for _, r in table.head(3).iterrows()]
    widths = [d.textlength(s, font=fp) + 40 for s in labels]
    gap, x = 22, 600 - (sum(widths) + 22 * (len(widths) - 1)) / 2
    for i, (s, w) in enumerate(zip(labels, widths)):
        if i == 0:
            _hero_pill(d, x + w / 2, 448, s, fp, _GOLD, (40, 33, 13), (150, 120, 30), h=54)
        else:
            _hero_pill(d, x + w / 2, 448, s, fp, _WHITE, (22, 29, 43), (36, 48, 73), h=54)
        x += w + gap

    d.text((600, 556), "worldcup2026ml.pt",
           font=_ttf(24), fill=_GREEN, anchor="mm")
    img.save(out, "PNG")
    return out


def share_cards(table, outdir, team=None):
    outdir = Path(outdir)
    hero_card(table, outdir / "share_card.png")          # the rich-link preview
    champions_card(table, outdir / "share_title.png")    # standalone title chart
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
        rows += (f"<tr{hl}><td>{_flag_img(t)}{t}</td><td>{s:.1f}%</td>"
                 f"<td>{p:.1f}%</td><td>{s-p:+.1f}</td></tr>")

    css = ("body{background:#0f1420;color:#e6e9ef;font-family:-apple-system,"
           "Segoe UI,Roboto,Arial,sans-serif;max-width:680px;margin:0 auto;"
           "padding:64px 18px 30px}h1{font-size:24px}table{width:100%;border-collapse:"
           "collapse}td,th{padding:8px;border-bottom:1px solid #232c40;text-align:"
           "left;font-size:14px}th{color:#8a93a8}.sub{color:#8a93a8}"
           ".homebtn{position:fixed;top:14px;left:14px;display:inline-flex;align-items:"
           "center;gap:6px;background:#161d2b;border:1px solid #243049;color:#e6e9ef;"
           "text-decoration:none;font-size:13px;font-weight:600;padding:7px 13px;"
           "border-radius:999px}.homebtn:hover{border-color:#00e0a4;color:#00e0a4}"
           "@media(max-width:560px){body{padding:56px 12px 30px}h1{font-size:20px}"
           "table{display:block;overflow-x:auto;white-space:nowrap}"
           "td,th{font-size:13px;padding:7px}}")
    html = (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>World Cup 2026 — comparison</title>{FAVICON}"
        '<meta property="og:title" content="World Cup 2026 — live vs pre-tournament">'
        '<meta property="og:description" content="Compare the live and blind '
        'pre-tournament title odds for all 48 teams.">'
        f'<meta property="og:image" content="{SITE}/outputs/share_card.png">'
        '<meta property="og:image:width" content="1200">'
        '<meta property="og:image:height" content="630">'
        f'<meta property="og:url" content="{SITE}/compare.html">'
        '<meta property="og:type" content="website">'
        '<meta name="twitter:card" content="summary_large_image">'
        '<meta name="theme-color" content="#0c1018">'
        '<link rel="manifest" href="manifest.json">'
        '<link rel="apple-touch-icon" href="apple-touch-icon.png">'
        f"<style>{css}</style></head><body>"
        f"<a class='homebtn' href='index.html'>🏠 Home</a>"
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
