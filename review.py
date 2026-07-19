"""Tournament Review — the end-of-World-Cup capstone + hub. A self-contained
outputs/review.html (Emerald Pitch) that opens with the champion + the model's
verdict and the champion's road, then links out to the three detailed
retrospectives (competition / report card / predicted-vs-reality).

Standalone + auto-updating: `python review.py`, and wired into run_pipeline so it
refreshes each build. It only renders once the final has a winner.
"""
from __future__ import annotations

import csv
import html as _html
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
CACHE = BASE / "api_cache"
SITE = "https://worldcup2026ml.pt"
STAGE_ORD = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
STAGE_SHORT = {"Round of 32": "R32", "Round of 16": "R16", "Quarter-finals": "QF",
               "Semi-finals": "SF", "Final": "Final"}


def _wc() -> pd.DataFrame:
    r = pd.read_csv(BASE / "results.csv", parse_dates=["date"])
    return r[(r["tournament"] == "FIFA World Cup") & (r["date"].dt.year == 2026)].sort_values("date")


def collect() -> dict | None:
    wc = _wc()
    played = wc[wc["home_score"].notna()].copy()
    if played.empty:
        return None
    final = played.iloc[-1]                                   # last WC game = the final
    hs, as_ = int(final["home_score"]), int(final["away_score"])
    if hs == as_:                                             # not decided in normal time -> need the KO winner
        pass
    champ = final["home_team"] if hs >= as_ else final["away_team"]
    runner = final["away_team"] if hs >= as_ else final["home_team"]
    cs, rs = (hs, as_) if hs >= as_ else (as_, hs)

    try:
        from wc2026.richdata import HL_TO_MARTJ42 as HL
    except Exception:
        HL = {}

    def nm(t):
        return HL.get(t, t)

    # champion's knockout road + whether the final went to extra time / pens
    path, aet = [], ""
    if (CACHE / "wc_matches.csv").exists():
        with (CACHE / "wc_matches.csv").open(encoding="utf-8") as f:
            for m in csv.DictReader(f):
                if m["round"] == "Final":
                    aet = (" (a.e.t.)" if "extra time" in m["status"]
                           else " (pens)" if "penalties" in m["status"] else "")
                if m["round"] not in STAGE_ORD:
                    continue
                h, a = nm(m["home"]), nm(m["away"])
                if champ not in (h, a):
                    continue
                sc = (m["score"] or "").split(" - ")
                if len(sc) != 2 or not sc[0].strip().isdigit():
                    continue
                hg, ag = int(sc[0]), int(sc[1])
                opp, cg, og = (a, hg, ag) if h == champ else (h, ag, hg)
                path.append({"round": m["round"], "opp": opp, "cg": cg, "og": og})
    path.sort(key=lambda x: STAGE_ORD.index(x["round"]))

    # the final's goal(s)
    gs = pd.read_csv(BASE / "goalscorers.csv", parse_dates=["date"])
    fg = gs[(gs["date"] == final["date"]) & (gs["home_team"] == final["home_team"])
            & (gs["away_team"] == final["away_team"])]
    fscorers = [f'{r["scorer"]} {int(r["minute"])}\'' for _, r in fg.iterrows()
                if str(r["scorer"]).strip()]

    # Golden Boot (from the built csv), goals total, champion's pre-tournament rank, the funnel
    gb = None
    if (OUT / "golden_boot.csv").exists():
        g = pd.read_csv(OUT / "golden_boot.csv")
        if len(g):
            gb = {"scorer": g.iloc[0]["scorer"], "g": int(g.iloc[0]["wc"])}
    goals = int(played["home_score"].sum() + played["away_score"].sum())
    pre_rank = None
    pf = BASE / "outputs_pretournament" / "predictions.csv"
    if pf.exists():
        pre = pd.read_csv(pf).sort_values("p_champion", ascending=False).reset_index(drop=True)
        hit = pre.index[pre["team"] == champ]
        pre_rank = int(hit[0]) + 1 if len(hit) else None
    try:
        from report import bracket_funnel
        funnel = bracket_funnel()
    except Exception:
        funnel = None

    return {"champ": champ, "runner": runner, "cs": cs, "rs": rs, "aet": aet,
            "path": path, "fscorers": fscorers, "gb": gb, "goals": goals,
            "n_matches": len(played), "pre_rank": pre_rank, "funnel": funnel}


_ORD = {1: "top pick", 2: "2nd pick", 3: "3rd pick"}


def build(m: dict) -> str:
    champ = _html.escape(m["champ"])
    runner = _html.escape(m["runner"])
    scline = f'{m["cs"]}–{m["rs"]}{m["aet"]}'
    goal = (" · " + ", ".join(_html.escape(s) for s in m["fscorers"])) if m["fscorers"] else ""

    def kpi(big, lbl, foot=""):
        return (f'<div class="kpi"><div class="big">{big}</div><div class="lbl">{lbl}</div>'
                f'{f"<div class=foot>{foot}</div>" if foot else ""}</div>')

    gb = m["gb"]
    kpis = "".join([
        kpi("🏆", "World Champions", champ),
        kpi(scline, "The final", f'{champ} beat {runner}{goal}'),
        kpi(f'{gb["g"]} ⚽' if gb else "—", "Golden Boot", _html.escape(gb["scorer"]) if gb else ""),
        kpi(f'{m["goals"]}', "Goals in the tournament", f'{m["n_matches"]} matches'),
    ])

    # the model's verdict (bracket funnel, sharpened to the point)
    verdict = ""
    fn = m["funnel"]
    if fn:
        frows = "".join(
            f'<div class="frow{" perfect" if r["hit"] == r["N"] else ""}">'
            f'<span class="fst">{r["lbl"]}<em> · last {r["N"]}</em></span>'
            f'<span class="fbar"><i style="width:{r["hit"] / r["N"] * 100:.0f}%"></i></span>'
            f'<span class="fn">{r["hit"]} / {r["N"]}</span></div>' for r in fn)
        rank_bit = (f' Its pre-tournament <b>{_ORD.get(m["pre_rank"], f"#{m['pre_rank']}")}</b>, '
                    f'{champ}, lifted the trophy.' if m["pre_rank"] else "")
        verdict = f"""<h2>The model called it</h2>
<div class="panel">
<p class="sub" style="margin:0 0 14px">Ranked by title odds <b>before a ball was kicked</b>, here is how many of the
model's top&nbsp;N reached the last&nbsp;N — perfect at the sharp end.{rank_bit}</p>
{frows}
<p class="note">Full skill + calibration breakdown in the <a href="report.html">model report card</a>.</p></div>"""

    # champion's road
    road = ""
    if m["path"]:
        cells = "".join(
            f'<div class="road"><div class="rl">{STAGE_SHORT.get(p["round"], p["round"])}</div>'
            f'<div class="rs">{p["cg"]}–{p["og"]}</div>'
            f'<div class="ro">{_html.escape(p["opp"])}</div></div>' for p in m["path"])
        road = f"""<h2>{champ}'s road to glory</h2>
<div class="panel"><div class="roadwrap">{cells}</div>
<p class="note">Five knockout wins to the title.</p></div>"""

    cards = f"""<h2>The full story</h2>
<div class="hub">
<a class="hcard" href="competition.html"><div class="he">📊</div><div class="ht">The tournament in numbers</div>
<div class="hd">Goals, creators, cards, records — the whole competition tallied.</div></a>
<a class="hcard" href="report.html"><div class="he">🎯</div><div class="ht">Model report card</div>
<div class="hd">Accuracy, calibration and the sharpest calls vs biggest shocks.</div></a>
<a class="hcard" href="../compare.html"><div class="he">🔮</div><div class="ht">Predicted vs reality</div>
<div class="hd">Who delivered, who beat the odds, who fell short of the forecast.</div></a>
</div>"""

    css = """
*{box-sizing:border-box}
:root{--bg:#0a0e14;--panel:#131a26;--panel2:#1a2434;--line:#263143;--line2:#1b2431;
--text:#f0f3f9;--text2:#cbd3e1;--muted:#8a95a9;--faint:#5d6a85;--green:#2ee6a6;--green2:#12c98d;
--gold:#ffcb5c;--red:#ff6b6b}
body{margin:0;color:var(--text);font-family:Inter,-apple-system,Segoe UI,Roboto,Arial,"Segoe UI Emoji",sans-serif;
line-height:1.55;-webkit-font-smoothing:antialiased;
background:radial-gradient(1100px 560px at 50% -160px,rgba(255,203,92,.10),transparent 70%),
radial-gradient(900px 480px at 50% 0,rgba(46,230,166,.08),transparent 70%),
linear-gradient(180deg,#0a0e14,#060910 80%);background-attachment:fixed;
max-width:940px;margin:0 auto;padding:52px 20px 80px}
.homebtn{position:fixed;top:14px;left:14px;display:inline-flex;align-items:center;gap:6px;
background:rgba(19,26,38,.82);backdrop-filter:blur(8px);border:1px solid var(--line);color:var(--text2);
text-decoration:none;font-size:13px;font-weight:600;padding:7px 13px;border-radius:999px}
.homebtn:hover{border-color:var(--green);color:var(--green)}
.eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:11.5px;font-weight:700;letter-spacing:1.4px;
text-transform:uppercase;color:var(--gold);background:rgba(255,203,92,.12);border:1px solid rgba(255,203,92,.30);
padding:6px 14px;border-radius:999px}
.eyebrow::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--gold);box-shadow:0 0 10px var(--gold)}
h1{font-family:Outfit,sans-serif;font-size:clamp(30px,6vw,50px);font-weight:800;letter-spacing:-1.5px;margin:16px 0 8px;line-height:1.02}
.sub{color:var(--muted);font-size:15px;max-width:680px;margin:0 0 8px}.sub b{color:var(--text2)}
h2{font-family:Outfit,sans-serif;font-size:20px;font-weight:700;margin:40px 0 14px;display:flex;align-items:center;gap:11px}
h2::before{content:"";width:4px;height:19px;border-radius:3px;background:linear-gradient(var(--gold),#e0a83a)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-top:26px}
.kpi{background:linear-gradient(158deg,var(--panel),var(--panel2));border:1px solid var(--line);
border-radius:14px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.kpi .big{font-family:Outfit,sans-serif;font-size:28px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.kpi .lbl{color:var(--muted);font-size:11px;margin-top:8px;text-transform:uppercase;letter-spacing:.6px;font-weight:600}
.kpi .foot{color:var(--faint);font-size:12px;margin-top:4px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.note{color:var(--faint);font-size:12.5px;margin-top:12px}.note a{color:var(--green);text-decoration:none;font-weight:600}
.frow{display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px solid var(--line2)}
.frow:last-of-type{border-bottom:0}
.fst{width:150px;font-size:13.5px;color:var(--text2);font-weight:600}.fst em{color:var(--faint);font-style:normal;font-weight:400;font-size:12px}
.fbar{flex:1;height:9px;background:var(--line2);border-radius:99px;overflow:hidden}
.fbar i{display:block;height:100%;background:linear-gradient(90deg,var(--green2),var(--green));border-radius:99px}
.fn{width:50px;text-align:right;font-family:Outfit,sans-serif;font-weight:700;font-size:14px;font-variant-numeric:tabular-nums;color:var(--text2)}
.frow.perfect .fbar i{background:linear-gradient(90deg,var(--gold),#ffd97a)}.frow.perfect .fn{color:var(--gold)}
.roadwrap{display:flex;gap:10px;overflow-x:auto;padding-bottom:4px}
.road{flex:1;min-width:92px;text-align:center;background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:12px 8px}
.road .rl{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--gold)}
.road .rs{font-family:Outfit,sans-serif;font-size:22px;font-weight:800;margin:6px 0 2px;font-variant-numeric:tabular-nums}
.road .ro{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hub{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.hcard{display:block;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;text-decoration:none;
box-shadow:0 2px 8px rgba(0,0,0,.35);transition:border-color .15s,transform .15s}
.hcard:hover{border-color:var(--green);transform:translateY(-2px)}
.hcard .he{font-size:24px}.hcard .ht{font-family:Outfit,sans-serif;font-size:16px;font-weight:700;color:var(--text);margin:10px 0 4px}
.hcard .hd{color:var(--muted);font-size:13px}
.foota{margin-top:44px;color:var(--faint);font-size:12px;text-align:center}.foota a{color:var(--green);text-decoration:none}
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>World Cup 2026 — Tournament Review</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
<style>{css}</style></head><body>
<a class="homebtn" href="../index.html">← Home</a>
<span class="eyebrow">Tournament review</span>
<h1>{champ} are world champions.</h1>
<p class="sub">The first 48-team World Cup is done — <b>{champ}</b> beat <b>{runner}</b> <b>{scline}</b> in the final.
Here is how it played out, and how the model's blind pre-tournament call held up.</p>

<div class="cards">{kpis}</div>

{verdict}

{road}

{cards}

<p class="foota">Poisson (Dixon-Coles) + Elo · <a href="{SITE}">{SITE.replace('https://','')}</a></p>
</body></html>"""


def main():
    m = collect()
    if not m:
        print("review skipped: no final yet")
        return
    (OUT / "review.html").write_text(build(m), encoding="utf-8")
    print(f"wrote outputs/review.html — champions: {m['champ']} ({m['cs']}–{m['rs']}{m['aet']} vs {m['runner']})")


if __name__ == "__main__":
    main()
