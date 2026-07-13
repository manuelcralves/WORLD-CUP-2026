"""Model report card — how well the BLIND pre-tournament model called the 2026 World
Cup. Reads outputs/played_review.csv (the blind model's W/D/L probabilities vs the actual
results) and writes a self-contained outputs/report.html with skill metrics, a calibration
(reliability) diagram and the sharpest calls / biggest shocks.

Standalone + UNLISTED: not linked from the site or the dashboard nav. Re-run any time
(auto-improves as more matches are played): `python report.py`.
"""
from __future__ import annotations

import html as _html
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
SITE = "https://worldcup2026ml.pt"
OUTCOMES = [("p_home", "home"), ("p_draw", "draw"), ("p_away", "away")]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _rps(p3, o3):
    """Ranked probability score for one game (ordered home-draw-away). Lower = better."""
    cp, co, s = 0.0, 0.0, 0.0
    for i in range(2):                    # last cumulative term is always 0
        cp += p3[i]; co += o3[i]
        s += (cp - co) ** 2
    return 0.5 * s


def compute(pr: pd.DataFrame) -> dict:
    P = pr[["p_home", "p_draw", "p_away"]].to_numpy(float)
    onehot = np.array([[1.0 if pr.iloc[i]["actual"] == o else 0.0 for _, o in OUTCOMES]
                       for i in range(len(pr))])
    base = onehot.mean(axis=0)            # climatology baseline (overall H/D/A base rates)

    rps = np.mean([_rps(P[i], onehot[i]) for i in range(len(pr))])
    rps_naive = np.mean([_rps(base, onehot[i]) for i in range(len(pr))])
    brier = np.mean(((P - onehot) ** 2).sum(axis=1))
    p_act = pr["p_actual"].to_numpy(float).clip(1e-6, 1)
    logloss = float(-np.log(p_act).mean())
    acc = float(pr["hit"].mean())

    # calibration: every (predicted prob, did-it-happen) pair across H/D/A
    probs = P.flatten()
    happened = onehot.flatten()
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, 9)
    bins = []
    for b in range(10):
        m = idx == b
        if m.sum() >= 4:
            bins.append({"pred": float(probs[m].mean()), "obs": float(happened[m].mean()),
                         "n": int(m.sum()), "lo": int(edges[b] * 100), "hi": int(edges[b + 1] * 100)})

    by_stage = {}
    if "stage" in pr.columns:
        for st, g in pr.groupby("stage"):
            by_stage[st] = {"n": len(g), "acc": float(g["hit"].mean())}

    return {"n": len(pr), "acc": acc, "rps": float(rps), "rps_naive": float(rps_naive),
            "brier": float(brier), "logloss": logloss, "bins": bins, "by_stage": by_stage,
            "base": base.tolist()}


# --------------------------------------------------------------------------- #
# reliability diagram (self-contained inline SVG)
# --------------------------------------------------------------------------- #
def reliability_svg(bins) -> str:
    W = H = 300; PAD = 40
    x = lambda p: PAD + p * W
    y = lambda p: PAD + (1 - p) * H
    g = [f'<svg viewBox="0 0 {W + PAD * 2} {H + PAD * 2}" xmlns="http://www.w3.org/2000/svg" '
         f'style="max-width:460px;width:100%">']
    # grid + ticks
    for t in range(0, 11, 2):
        p = t / 10
        g.append(f'<line x1="{x(p):.0f}" y1="{y(0):.0f}" x2="{x(p):.0f}" y2="{y(1):.0f}" stroke="#1b2431"/>')
        g.append(f'<line x1="{x(0):.0f}" y1="{y(p):.0f}" x2="{x(1):.0f}" y2="{y(p):.0f}" stroke="#1b2431"/>')
        g.append(f'<text x="{x(p):.0f}" y="{y(0) + 20:.0f}" fill="#8a95a9" font-size="10" text-anchor="middle">{t * 10}%</text>')
        g.append(f'<text x="{x(0) - 8:.0f}" y="{y(p) + 3:.0f}" fill="#8a95a9" font-size="10" text-anchor="end">{t * 10}%</text>')
    # perfect-calibration diagonal
    g.append(f'<line x1="{x(0):.0f}" y1="{y(0):.0f}" x2="{x(1):.0f}" y2="{y(1):.0f}" '
             f'stroke="#5d6a85" stroke-dasharray="5 4" stroke-width="1.5"/>')
    g.append(f'<text x="{x(.72):.0f}" y="{y(.80):.0f}" fill="#5d6a85" font-size="10" '
             f'transform="rotate(-45 {x(.72):.0f} {y(.80):.0f})">perfectly calibrated</text>')
    # the model's calibration curve + points (radius ~ count)
    pts = " ".join(f"{x(b['pred']):.1f},{y(b['obs']):.1f}" for b in bins)
    g.append(f'<polyline points="{pts}" fill="none" stroke="#2ee6a6" stroke-width="2.5" '
             f'stroke-linejoin="round" opacity=".9"/>')
    nmax = max((b["n"] for b in bins), default=1)
    for b in bins:
        r = 4 + 7 * (b["n"] / nmax) ** .5
        g.append(f'<circle cx="{x(b["pred"]):.1f}" cy="{y(b["obs"]):.1f}" r="{r:.1f}" '
                 f'fill="#2ee6a6" stroke="#0a0e14" stroke-width="1.5"/>')
    # axis titles
    g.append(f'<text x="{PAD + W / 2:.0f}" y="{H + PAD * 2 - 4:.0f}" fill="#cbd3e1" font-size="12" '
             f'font-weight="600" text-anchor="middle">Model said this likely →</text>')
    g.append(f'<text x="14" y="{PAD + H / 2:.0f}" fill="#cbd3e1" font-size="12" font-weight="600" '
             f'text-anchor="middle" transform="rotate(-90 14 {PAD + H / 2:.0f})">…and it happened this often</text>')
    g.append("</svg>")
    return "".join(g)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
def _fl(r, side):
    return f"{r['home']} {int(r['hs'])}-{int(r['as'])} {r['away']}" if side else ""


def build(pr: pd.DataFrame, m: dict) -> str:
    def kpi(big, lbl, foot=""):
        return (f'<div class="kpi"><div class="big">{big}</div>'
                f'<div class="lbl">{lbl}</div>{f"<div class=foot>{foot}</div>" if foot else ""}</div>')

    edge = (m["rps_naive"] - m["rps"]) / m["rps_naive"] * 100
    kpis = "".join([
        kpi(f'{m["acc"] * 100:.0f}%', "Winners called", f'{int(round(m["acc"] * m["n"]))}/{m["n"]} matches'),
        kpi(f'{m["rps"]:.3f}', "Ranked prob. score", f'naive {m["rps_naive"]:.3f} · {edge:.0f}% sharper'),
        kpi(f'{m["brier"]:.3f}', "Brier score", "multiclass W/D/L"),
        kpi(f'{m["logloss"]:.3f}', "Log loss", "lower = better"),
    ])

    # calibration table
    crows = "".join(
        f'<tr><td>{b["lo"]}–{b["hi"]}%</td><td class="n">{b["pred"] * 100:.0f}%</td>'
        f'<td class="n">{b["obs"] * 100:.0f}%</td><td class="n">{b["n"]}</td></tr>'
        for b in m["bins"])

    # sharpest calls (confident + right) and biggest shocks (low prob, happened)
    hits = pr[pr["hit"]].sort_values("p_actual", ascending=False).head(6)
    shocks = pr.sort_values("p_actual").head(6)

    def call_rows(df, shock=False):
        out = ""
        for _, r in df.iterrows():
            res = f'{_html.escape(str(r["home"]))} <b>{int(r["hs"])}-{int(r["as"])}</b> {_html.escape(str(r["away"]))}'
            pc = f'{r["p_actual"] * 100:.0f}%'
            tag = (f'<span class="pill r">gave it {pc}</span>' if shock
                   else f'<span class="pill g">{pc} · nailed</span>')
            out += f'<div class="crow"><span>{res}</span>{tag}</div>'
        return out

    stage = " · ".join(f'{st}: {v["acc"] * 100:.0f}% ({v["n"]})' for st, v in m["by_stage"].items())

    css = """
*{box-sizing:border-box}
:root{--bg:#0a0e14;--panel:#131a26;--panel2:#1a2434;--line:#263143;--line2:#1b2431;
--text:#f0f3f9;--text2:#cbd3e1;--muted:#8a95a9;--faint:#5d6a85;--green:#2ee6a6;--green2:#12c98d;
--gold:#ffcb5c;--red:#ff6b6b;--ink:#052018}
body{margin:0;color:var(--text);font-family:Inter,-apple-system,Segoe UI,Roboto,Arial,"Segoe UI Emoji",sans-serif;
line-height:1.55;-webkit-font-smoothing:antialiased;
background:radial-gradient(1100px 520px at 50% -150px,rgba(46,230,166,.10),transparent 70%),
linear-gradient(180deg,#0a0e14,#060910 80%);background-attachment:fixed;
max-width:920px;margin:0 auto;padding:52px 20px 80px}
.eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:11.5px;font-weight:700;letter-spacing:1.4px;
text-transform:uppercase;color:var(--green);background:rgba(46,230,166,.12);border:1px solid rgba(46,230,166,.28);
padding:6px 14px;border-radius:999px}
.eyebrow::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green)}
h1{font-family:Outfit,sans-serif;font-size:clamp(28px,5vw,42px);font-weight:800;letter-spacing:-1px;margin:16px 0 8px}
.sub{color:var(--muted);font-size:15px;max-width:640px;margin:0 0 8px}
.sub b{color:var(--text2)}
h2{font-family:Outfit,sans-serif;font-size:20px;font-weight:700;margin:40px 0 14px;display:flex;align-items:center;gap:11px}
h2::before{content:"";width:4px;height:19px;border-radius:3px;background:linear-gradient(var(--green),var(--green2))}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-top:26px}
.kpi{background:linear-gradient(158deg,var(--panel),var(--panel2));border:1px solid var(--line);
border-radius:14px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.kpi .big{font-family:Outfit,sans-serif;font-size:30px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.kpi .lbl{color:var(--muted);font-size:11px;margin-top:8px;text-transform:uppercase;letter-spacing:.6px;font-weight:600}
.kpi .foot{color:var(--faint);font-size:12px;margin-top:4px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.calib{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:center}
@media(max-width:680px){.calib{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line2)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
tbody tr:last-child td{border-bottom:0}
.note{color:var(--faint);font-size:12.5px;margin-top:10px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:680px){.two{grid-template-columns:1fr}}
.crow{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--line2);font-size:13.5px}
.crow:last-child{border-bottom:0}
.pill{font-size:11px;font-weight:700;padding:3px 10px;border-radius:999px;white-space:nowrap}
.pill.g{color:var(--green);background:rgba(46,230,166,.12)}
.pill.r{color:var(--gold);background:rgba(255,203,92,.12)}
.foota{margin-top:44px;color:var(--faint);font-size:12px;text-align:center}
.foota a{color:var(--green);text-decoration:none}
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Model report card · World Cup 2026</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
<style>{css}</style></head><body>
<span class="eyebrow">Model report card</span>
<h1>How good were the blind predictions?</h1>
<p class="sub">The pre-tournament model made a call on every match with <b>zero knowledge of any 2026 result</b>.
Here is how those blind calls held up against what actually happened — accuracy, sharpness, and how
well its confidence matched reality. <b>{m['n']}</b> matches so far; this updates as the tournament plays out.</p>

<div class="cards">{kpis}</div>

<h2>Is it well calibrated?</h2>
<div class="panel"><div class="calib">
<div>{reliability_svg(m['bins'])}</div>
<div>
<p class="sub" style="margin:0 0 12px">Every dot bundles all the calls at one confidence level. If the model
is honest, when it says <b>60%</b> the thing happens <b>~60%</b> of the time — so the dots hug the dashed line.
Dot size = how many calls sit in that bucket.</p>
<table><thead><tr><th>Model said</th><th class="n">avg</th><th class="n">happened</th><th class="n">calls</th></tr></thead>
<tbody>{crows}</tbody></table>
</div>
</div>
<p class="note">Winners called by stage — {stage}.</p></div>

<h2>Sharpest calls &amp; biggest shocks</h2>
<div class="two">
<div class="panel"><h3 style="margin:0 0 6px;font-size:14px;color:var(--text2)">🎯 Nailed with conviction</h3>
{call_rows(hits)}</div>
<div class="panel"><h3 style="margin:0 0 6px;font-size:14px;color:var(--text2)">😱 Never saw it coming</h3>
{call_rows(shocks, shock=True)}</div>
</div>

<p class="foota">Poisson (Dixon-Coles) + Elo · blind pre-tournament model · <a href="{SITE}">{SITE.replace('https://','')}</a></p>
</body></html>"""


def main():
    p = OUT / "played_review.csv"
    if not p.exists():
        print("no played_review.csv — run the pipeline first")
        return
    pr = pd.read_csv(p)
    pr = pr[pr["p_actual"].notna()].reset_index(drop=True)
    if pr.empty:
        print("no reviewable matches yet")
        return
    m = compute(pr)
    (OUT / "report.html").write_text(build(pr, m), encoding="utf-8")
    print(f"wrote outputs/report.html — {m['n']} matches, acc {m['acc']*100:.0f}%, "
          f"RPS {m['rps']:.3f} (naive {m['rps_naive']:.3f})")


if __name__ == "__main__":
    main()
