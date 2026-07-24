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
_GOLD = (255, 203, 92)
_GREEN = (46, 230, 166)
_WHITE = (240, 243, 249)
_GREY = (138, 149, 169)
_DARK = (10, 14, 20)


def _ttf(size: int, bold: bool = True):
    fp = _fm.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return ImageFont.truetype(_fm.findfont(fp), size)


def _hero_bg(w: int, h: int) -> Image.Image:
    """Dark vertical gradient with soft radial glows (the homepage's backdrop)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (yy / h)[..., None]
    img = np.array([10, 14, 20], np.float32) * (1 - t) + np.array([6, 9, 16], np.float32) * t

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
    """A 1200x630 branded card (the homepage in a single image) for link previews. Once the title
    is settled (favourite at 100%) it switches from 'who wins?' + odds to a champions card."""
    W, H = _HERO
    img = _hero_bg(W, H)
    d = ImageDraw.Draw(img)
    _hero_trophy(d, 600 - 32 * (118 / 64), 28, 118)         # centred trophy

    champ = table.iloc[0]
    decided = float(champ["p_champion"]) >= 0.999
    fh = _ttf(62)
    if decided:
        _hero_pill(d, 600, 170, "WORLD CUP 2026  ·  FULL TIME", _ttf(19),
                   _GOLD, (40, 33, 13), (150, 120, 30), h=38)
        d.text((600, 238), f"{champ['team']} are", font=fh, fill=_WHITE, anchor="mm")
        d.text((600, 306), "World Champions", font=fh, fill=_GOLD, anchor="mm")
        d.text((600, 372), "Blind, it nailed both finalists and all four semi-finalists",
               font=_ttf(23, bold=False), fill=_GREY, anchor="mm")
        labels, golds = ["Both finalists", "All four semis", "Champion in its top 2"], {0}
    else:
        _hero_pill(d, 600, 170, "MACHINE-LEARNING PREDICTION", _ttf(19),
                   _GREEN, (14, 33, 30), (0, 92, 76), h=38)
        d.text((600, 238), "Who wins the", font=fh, fill=_WHITE, anchor="mm")
        d.text((600, 306), "World Cup 2026?", font=fh, fill=_GOLD, anchor="mm")
        d.text((600, 372), "A machine-learning model — Poisson + Elo, 150 years of football",
               font=_ttf(25, bold=False), fill=_GREY, anchor="mm")
        labels = [f"{r['team']}  {r['p_champion'] * 100:.0f}%" for _, r in table.head(3).iterrows()]
        golds = {0}

    # pills row (odds contenders, or the 'what it got right' badges once decided) — highlighted in gold
    fp = _ttf(25)
    widths = [d.textlength(s, font=fp) + 40 for s in labels]
    gap, x = 22, 600 - (sum(widths) + 22 * (len(widths) - 1)) / 2
    for i, (s, w) in enumerate(zip(labels, widths)):
        if i in golds:
            _hero_pill(d, x + w / 2, 448, s, fp, _GOLD, (40, 33, 13), (150, 120, 30), h=54)
        else:
            _hero_pill(d, x + w / 2, 448, s, fp, _WHITE, (22, 29, 43), (36, 48, 73), h=54)
        x += w + gap

    d.text((600, 556), "worldcup2026ml.pt", font=_ttf(24), fill=_GREEN, anchor="mm")
    img.save(out, "PNG")
    return out


def share_cards(table, outdir, team=None):
    outdir = Path(outdir)
    hero_card(table, outdir / "share_card.png")          # the rich-link preview
    champions_card(table, outdir / "share_title.png")    # standalone title chart
    spotlight_card(table, outdir / "share_spotlight.png", team)
    return outdir


# --------------------------------------------------------------------------- #
# before-vs-after: the blind pre-tournament call vs where each team actually finished
# --------------------------------------------------------------------------- #
_STAGE_RANK = {"Group Stage": 0, "Round of 32": 1, "Round of 16": 2,
               "Quarter-finals": 3, "Semi-finals": 4, "Final": 5}
_STAGE_LABEL = {0: "Groups", 1: "Round of 32", 2: "Round of 16", 3: "Quarter-finals",
                4: "Semi-finals", 5: "Final"}


def _actual_stages() -> dict:
    """Deepest stage each team reached (martj42 names). The 3rd-place play-off counts as the
    semi-finals — its teams lost there, they didn't reach a deeper round. {} if cache absent."""
    import csv as _csv
    mf = Path(__file__).resolve().parent.parent / "api_cache" / "wc_matches.csv"
    if not mf.exists():
        return {}
    try:
        from .richdata import HL_TO_MARTJ42 as HL
    except Exception:
        HL = {}
    reached: dict = {}
    with mf.open(encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rd = r["round"]
            base = ("Group Stage" if rd.startswith("Group")
                    else "Semi-finals" if rd == "3rd Place Final" else rd)
            if base in _STAGE_RANK:
                for t in (HL.get(r["home"], r["home"]), HL.get(r["away"], r["away"])):
                    reached[t] = max(reached.get(t, 0), _STAGE_RANK[base])
    return reached


def comparison_page(snap_csv, pre_csv, out_html, top=14):
    from . import retro_common as RC
    snap = pd.read_csv(snap_csv).set_index("team")
    pre = pd.read_csv(pre_csv).set_index("team")
    reached = _actual_stages()

    def _exp(t):                     # expected KO rounds reached (sum of pre-tournament reach probs)
        return float(sum(pre.loc[t, c] for c in ("p_ko", "p_r16", "p_qf", "p_sf", "p_final")
                         if t in pre.index and c in pre.columns))

    # 1) favourites' report card — the top-8 pre-tournament picks vs where they finished
    fav = list(pre.sort_values("p_champion", ascending=False).index[:8])
    fav_rows = ""
    for t in fav:
        st = reached.get(t)
        met = st is not None and st >= round(_exp(t))
        stage = _STAGE_LABEL.get(st, "—") if st is not None else "—"
        fav_rows += (f'<tr><td>{_flag_img(t)}{t}</td>'
                     f'<td class="n">{pre.loc[t, "p_champion"] * 100:.1f}%</td>'
                     f'<td><span class="tag {"up" if met else "down"}">{stage}</span></td></tr>')

    # 2) surprises — biggest over/under-performers vs pre-tournament expected depth. The top-4
    #    seeds are excluded from "beat the odds" (they were meant to go deep — that's the funnel).
    surp = [(t, st, st - _exp(t)) for t, st in reached.items() if t in pre.index]
    top4 = set(fav[:4])
    over = [x for x in sorted(surp, key=lambda x: -x[2]) if x[0] not in top4][:5]
    under = sorted(surp, key=lambda x: x[2])[:5]

    def _slist(items, up):
        return "".join(
            f'<div class="srow"><span class="stm">{_flag_img(t)}{t}</span>'
            f'<span class="sr {"up" if up else "down"}">{_STAGE_LABEL[st]}'
            f'<em> · {pre.loc[t, "p_champion"] * 100:.1f}% to win</em></span></div>'
            for t, st, _ in items)

    # 3) title race — the two finalists' odds, pre-tournament -> live. Use p_final (both actual
    #    finalists sit at ~100% reach-the-final), champion first: sorting by p_champion breaks
    #    once the title is settled, when every non-champion is tied at 0% (an arbitrary 0% team
    #    would sneak in as "finalist").
    finals = [t for t in snap.sort_values(["p_final", "p_champion"], ascending=False).index[:2]
              if t in pre.index]
    race = "".join(
        f'<div class="stat"><span>{_flag_img(t)}{t}</span>'
        f'<b>{pre.loc[t, "p_champion"] * 100:.1f}% <span class="arrow">→</span> '
        f'{snap.loc[t, "p_champion"] * 100:.1f}%</b></div>' for t in finals)

    css = """
*{box-sizing:border-box}
:root{--bg:#0a0e14;--panel:#131a26;--panel2:#1a2434;--line:#263143;--line2:#1b2431;
--text:#f0f3f9;--text2:#cbd3e1;--muted:#8a95a9;--faint:#5d6a85;--green:#2ee6a6;--green2:#12c98d;
--gold:#ffcb5c;--red:#ff6b6b}
body{margin:0;color:var(--text);font-family:Inter,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
line-height:1.55;-webkit-font-smoothing:antialiased;
background:radial-gradient(1100px 520px at 50% -150px,rgba(46,230,166,.10),transparent 70%),
linear-gradient(180deg,#0a0e14,#060910 80%);background-attachment:fixed;
max-width:900px;margin:0 auto;padding:52px 20px 80px}
.homebtn{position:fixed;top:14px;left:14px;display:inline-flex;align-items:center;gap:6px;
background:rgba(19,26,38,.82);backdrop-filter:blur(8px);border:1px solid var(--line);color:var(--text2);
text-decoration:none;font-size:13px;font-weight:600;padding:7px 13px;border-radius:999px}
.homebtn:hover{border-color:var(--green);color:var(--green)}
.eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:11.5px;font-weight:700;letter-spacing:1.4px;
text-transform:uppercase;color:var(--green);background:rgba(46,230,166,.12);border:1px solid rgba(46,230,166,.28);
padding:6px 14px;border-radius:999px}
.eyebrow::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green)}
h1{font-family:Outfit,sans-serif;font-size:clamp(28px,5vw,42px);font-weight:800;letter-spacing:-1px;margin:16px 0 8px}
.sub{color:var(--muted);font-size:15px;max-width:660px;margin:0 0 8px}.sub b{color:var(--text2)}
h2{font-family:Outfit,sans-serif;font-size:20px;font-weight:700;margin:40px 0 14px;display:flex;align-items:center;gap:11px}
h2::before{content:"";width:4px;height:19px;border-radius:3px;background:linear-gradient(var(--green),var(--green2))}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:680px){.two{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:14px}
td,th{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line2)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
tbody tr:last-child td,table tr:last-child td{border-bottom:0}
img.fl{width:19px;height:13px;border-radius:2px;object-fit:cover;vertical-align:middle;margin-right:8px}
.tag{font-size:11.5px;font-weight:700;padding:3px 10px;border-radius:999px;white-space:nowrap}
.tag.up{color:var(--green);background:rgba(46,230,166,.12)}
.tag.down{color:var(--red);background:rgba(255,107,107,.13)}
.srow{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--line2);font-size:14px}
.srow:last-child{border-bottom:0}
.sr{font-weight:700;font-size:12.5px;white-space:nowrap}.sr em{font-style:normal;font-weight:400;color:var(--faint);font-size:11.5px}
.sr.up{color:var(--green)}.sr.down{color:var(--red)}
.stat{display:flex;justify-content:space-between;align-items:center;padding:11px 0;border-bottom:1px solid var(--line2);font-size:15px}
.stat:last-child{border-bottom:0}.stat b{font-family:Outfit,sans-serif;font-variant-numeric:tabular-nums}
.arrow{color:var(--faint);margin:0 4px}
.note{color:var(--faint);font-size:12.5px;margin-top:12px}
.links{margin-top:30px;color:var(--faint);font-size:13px}.links a{color:var(--green);text-decoration:none;font-weight:600}
"""
    fin_names = " &amp; ".join(finals) if finals else ""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>World Cup 2026 — predicted vs reality</title>{FAVICON}
<meta property="og:title" content="World Cup 2026 — predicted vs reality">
<meta property="og:description" content="The blind pre-tournament model's picks vs how the World Cup actually played out — who delivered, who beat the odds, who fell short.">
<meta property="og:image" content="{SITE}/outputs/share_card.png">
<meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">
<meta property="og:url" content="{SITE}/compare.html"><meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image"><meta name="theme-color" content="#0a0e14">
<link rel="manifest" href="manifest.json"><link rel="apple-touch-icon" href="apple-touch-icon.png">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
<style>{css}{RC.NAV_CSS}</style></head><body>
{RC.nav('compare')}
<span class="eyebrow">Before vs after</span>
<h1>Predicted vs reality</h1>
<p class="sub">The model ranked all 48 teams <b>before a ball was kicked</b>. Here is how that blind call
held up against the real bracket — who delivered, who beat the odds, and who fell short.</p>

<h2>Did the favourites deliver?</h2>
<div class="panel"><table><thead><tr><th>Pre-tournament pick</th><th class="n">Title odds</th><th>Finished</th></tr></thead>
<tbody>{fav_rows}</tbody></table>
<p class="note">Green = reached at least its expected depth · red = fell short.</p></div>

<h2>Surprises of the tournament</h2>
<div class="two">
<div class="panel"><h3 style="margin:0 0 8px;font-size:14px;color:var(--text2)">📈 Beat the odds</h3>{_slist(over, True)}</div>
<div class="panel"><h3 style="margin:0 0 8px;font-size:14px;color:var(--text2)">📉 Fell short</h3>{_slist(under, False)}</div>
</div>

<h2>The title race</h2>
<div class="panel">{race}
<p class="note">The two finalists — <b>{fin_names}</b> — were the model's <b>top two</b> pre-tournament. Their odds soared as the field cleared.</p></div>

<p class="links">Open the full dashboards: <a href="outputs/dashboard.html">live</a> · <a href="outputs_pretournament/dashboard.html">pre-tournament (blind)</a></p>
</body></html>"""
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
