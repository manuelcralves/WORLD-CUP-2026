"""The tournament in numbers — a self-contained retrospective of the 2026 World Cup
built from the martj42 results/goalscorers and the Highlightly rich data (assists,
cards, match stats). Writes outputs/competition.html (Emerald Pitch, noindex, unlisted).

Standalone + auto-updating: re-run any time (`python competition.py`), and it is wired
into run_pipeline so it refreshes each build — the numbers complete once the final is in.
"""
from __future__ import annotations

import csv
import html as _html
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from wc2026 import richdata as RICH

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
CACHE = BASE / "api_cache"
SITE = "https://worldcup2026ml.pt"

_BANDS = [(0, 15, "1-15"), (16, 30, "16-30"), (31, 45, "31-45"),
          (46, 60, "46-60"), (61, 75, "61-75"), (76, 120, "76-90+")]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _wc_played() -> pd.DataFrame:
    r = pd.read_csv(BASE / "results.csv", parse_dates=["date"])
    wc = r[(r["tournament"] == "FIFA World Cup") & (r["date"].dt.year == 2026)].copy()
    p = wc[wc["home_score"].notna()].copy()
    p["tot"] = p["home_score"] + p["away_score"]
    p["marg"] = (p["home_score"] - p["away_score"]).abs()
    return p


def _wc_goals(played: pd.DataFrame) -> pd.DataFrame:
    """Goalscorer rows restricted to World Cup matches (goalscorers.csv also holds 2026
    friendlies / qualifiers, which must not pollute the totals)."""
    g = pd.read_csv(BASE / "goalscorers.csv", parse_dates=["date"])
    g = g[g["date"].dt.year == 2026].copy()
    g["k"] = g["date"].dt.strftime("%Y-%m-%d") + "|" + g["home_team"] + "|" + g["away_team"]
    keys = set(played["date"].dt.strftime("%Y-%m-%d") + "|" + played["home_team"] + "|" + played["away_team"])
    g = g[g["k"].isin(keys)].copy()
    g["og"] = g["own_goal"].astype(str).str.upper().eq("TRUE")
    g["pen"] = g["penalty"].astype(str).str.upper().eq("TRUE")
    g["min"] = pd.to_numeric(g["minute"], errors="coerce")
    return g


def _name_map(goals: pd.DataFrame) -> dict:
    """assist_key -> full, accented name. Built from the Highlightly line-ups (everyone who
    played, spelled in full — 'Brahim Díaz', 'Martin Ødegaard'), with martj42 scorer spellings
    taking priority so an assister who also scored matches the scorers table exactly. Fixes the
    abbreviated / accent-dropped assist names Highlightly puts in goal events ('M. Odegaard')."""
    m: dict = defaultdict(Counter)
    lf = CACHE / "wc_lineups.csv"
    if lf.exists():
        with lf.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                nm = (r.get("player") or "").strip()
                if nm:
                    m[RICH.assist_key(nm)][nm] += 1
    out = {k: max(v, key=lambda s: (len(s), v[s])) for k, v in m.items()}
    for nm in goals["scorer"].dropna():          # martj42 scorer spelling wins
        out[RICH.assist_key(str(nm))] = str(nm)
    return out


def _top_assisters(name_map: dict, n=8) -> list:
    """Top creators by assist_key (same Goal-only, accent-stripped logic the Golden Boot
    uses), shown with the proper martj42 name where the assister also scored."""
    cnt: Counter = Counter()
    names: dict = defaultdict(Counter)
    ef = CACHE / "wc_events.csv"
    if not ef.exists():
        return []
    with ef.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("type") != "Goal":
                continue
            a = (r.get("assist") or "").strip()
            if not a:
                continue
            k = RICH.assist_key(a)
            cnt[k] += 1
            names[k][a] += 1
    out = []
    for k, c in cnt.most_common(n):
        disp = name_map.get(k) or max(names[k], key=lambda nm: (len(nm), names[k][nm]))
        out.append({"name": disp, "n": c})
    return out


def _cards(name_map: dict) -> dict:
    yel = red = 0
    team: Counter = Counter()
    player: dict = defaultdict(lambda: {"team": "", "Y": 0, "R": 0})
    ef = CACHE / "wc_events.csv"
    if ef.exists():
        with ef.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                t = r["type"]
                k = "Y" if t == "Yellow Card" else "R" if t == "Red Card" else None
                if not k:
                    continue
                if k == "Y":
                    yel += 1
                else:
                    red += 1
                team[r["team"]] += (1 if k == "Y" else 1)
                if r.get("player"):
                    player[r["player"]]["team"] = r["team"]
                    player[r["player"]][k] += 1
    worst = sorted(player.items(), key=lambda kv: (-(kv[1]["Y"] + kv[1]["R"]), -kv[1]["R"], kv[0]))[:1]
    wp = None
    if worst:
        raw = worst[0][0]
        wp = {"player": name_map.get(RICH.assist_key(raw), raw), **worst[0][1]}
    return {"yellow": yel, "red": red, "teams": team.most_common(5), "worst": wp}


def _stats_totals() -> dict:
    corners = shots = fouls = 0
    have = set()
    sf = CACHE / "wc_stats.csv"
    if sf.exists():
        with sf.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                have.add(r["match_id"])
                v = r["value"]
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if r["stat"] == "Corners":
                    corners += iv
                elif r["stat"] in ("Shots on target", "Shots off target"):
                    shots += iv
                elif r["stat"] == "Fouls":
                    fouls += iv
    return {"corners": corners, "shots": shots, "fouls": fouls, "n": len(have)}


def _knockout_drama() -> dict:
    sh, et = [], 0
    mf = CACHE / "wc_matches.csv"
    if mf.exists():
        with mf.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "penalties" in r["status"]:
                    sh.append(f'{r["home"]} v {r["away"]}')
                if "extra time" in r["status"]:
                    et += 1
    return {"shootouts": sh, "extra_time": et}


def compute() -> dict:
    played = _wc_played()
    goals = _wc_goals(played)
    name_map = _name_map(goals)
    real = goals[~goals["og"]]
    scorers = real.groupby("scorer").size().sort_values(ascending=False)
    ast = RICH.assist_counts(str(CACHE))
    top_sc = []
    for name, c in scorers.head(10).items():
        top_sc.append({"name": name, "g": int(c), "a": ast.get(RICH.assist_key(str(name)), 0)})

    bands = []
    mins = goals["min"].dropna()
    for lo, hi, lbl in _BANDS:
        bands.append({"lbl": lbl, "n": int(((mins >= lo) & (mins <= hi)).sum())})

    bw = played.sort_values("marg", ascending=False).iloc[0]
    hs = played.sort_values("tot", ascending=False).iloc[0]
    home = int((played["home_score"] > played["away_score"]).sum())
    draw = int((played["home_score"] == played["away_score"]).sum())
    away = int((played["home_score"] < played["away_score"]).sum())

    return {
        "n_matches": len(played), "goals": int(played["tot"].sum()),
        "gpg": float(played["tot"].mean()),
        "pens": int(goals["pen"].sum()), "ogs": int(goals["og"].sum()),
        "scorers": top_sc, "bands": bands,
        "biggest": {"s": f'{bw.home_team} {int(bw.home_score)}-{int(bw.away_score)} {bw.away_team}'},
        "highest": {"s": f'{hs.home_team} {int(hs.home_score)}-{int(hs.away_score)} {hs.away_team}', "t": int(hs.tot)},
        "hda": {"home": home, "draw": draw, "away": away},
        "assists": _top_assisters(name_map), "cards": _cards(name_map),
        "totals": _stats_totals(), "drama": _knockout_drama(),
    }


# --------------------------------------------------------------------------- #
# little inline-SVG bar chart: goals by minute band
# --------------------------------------------------------------------------- #
def bands_svg(bands) -> str:
    W, H, PAD, GAP = 460, 200, 34, 14
    n = len(bands)
    bw = (W - PAD * 2 - GAP * (n - 1)) / n
    mx = max((b["n"] for b in bands), default=1)
    g = [f'<svg viewBox="0 0 {W} {H + 34}" xmlns="http://www.w3.org/2000/svg" style="width:100%">']
    for i, b in enumerate(bands):
        x = PAD + i * (bw + GAP)
        bh = (b["n"] / mx) * (H - PAD)
        y = H - bh
        peak = b["n"] == mx
        col = "var(--gold)" if peak else "var(--green)"
        g.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="5" fill="{col}" opacity="{0.95 if peak else 0.8}"/>')
        g.append(f'<text x="{x + bw / 2:.1f}" y="{y - 7:.1f}" fill="var(--text)" font-size="13" font-weight="700" text-anchor="middle" style="font-variant-numeric:tabular-nums">{b["n"]}</text>')
        g.append(f'<text x="{x + bw / 2:.1f}" y="{H + 18:.1f}" fill="var(--muted)" font-size="11" text-anchor="middle">{b["lbl"]}</text>')
    g.append(f'<text x="{W / 2:.0f}" y="{H + 32:.0f}" fill="var(--faint)" font-size="10.5" text-anchor="middle">minute of the match</text>')
    g.append("</svg>")
    return "".join(g)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
def build(m: dict) -> str:
    def kpi(big, lbl, foot=""):
        return (f'<div class="kpi"><div class="big">{big}</div>'
                f'<div class="lbl">{lbl}</div>{f"<div class=foot>{foot}</div>" if foot else ""}</div>')

    kpis = "".join([
        kpi(f'{m["goals"]}', "Goals scored", f'{m["n_matches"]} of 104 matches'),
        kpi(f'{m["gpg"]:.2f}', "Goals per game", "across the tournament"),
        kpi(f'{m["pens"]}', "Penalties scored", f'+ {m["ogs"]} own goals'),
        kpi(f'{len(m["drama"]["shootouts"])}', "Shootouts", f'{m["drama"]["extra_time"]} decided in extra time'),
        kpi(f'{m["cards"]["red"]}', "Red cards", f'{m["cards"]["yellow"]} yellows'),
    ])

    # top scorers
    lead = m["scorers"][0] if m["scorers"] else None
    sc_rows = "".join(
        f'<tr><td>{i + 1}</td><td>{_html.escape(s["name"])}</td>'
        f'<td class="n"><b>{s["g"]}</b></td><td class="n">{s["a"] or "—"}</td></tr>'
        for i, s in enumerate(m["scorers"]))

    # assists
    as_rows = "".join(
        f'<tr><td>{i + 1}</td><td>{_html.escape(a["name"])}</td><td class="n"><b>{a["n"]}</b></td></tr>'
        for i, a in enumerate(m["assists"]))

    # discipline bars (yellow-equivalent points, Y + 4R weighting for order already)
    cmax = max((c for _, c in m["cards"]["teams"]), default=1)
    disc = "".join(
        f'<div class="drow"><span class="dt">{_html.escape(t)}</span>'
        f'<span class="dbar"><i style="width:{c / cmax * 100:.0f}%"></i></span>'
        f'<span class="dn">{c}</span></div>'
        for t, c in m["cards"]["teams"])
    worst = m["cards"]["worst"]
    worst_txt = (f'Most-booked player: <b>{_html.escape(worst["player"])}</b> '
                 f'({worst["Y"]}Y{f", {worst['R']}R" if worst["R"] else ""})' if worst else "")

    # shootouts list
    sh = m["drama"]["shootouts"]
    sh_txt = " · ".join(_html.escape(s) for s in sh) if sh else "none"

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
.sub{color:var(--muted);font-size:15px;max-width:660px;margin:0 0 8px}
.sub b{color:var(--text2)}
h2{font-family:Outfit,sans-serif;font-size:20px;font-weight:700;margin:40px 0 14px;display:flex;align-items:center;gap:11px}
h2::before{content:"";width:4px;height:19px;border-radius:3px;background:linear-gradient(var(--green),var(--green2))}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-top:26px}
.kpi{background:linear-gradient(158deg,var(--panel),var(--panel2));border:1px solid var(--line);
border-radius:14px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.kpi .big{font-family:Outfit,sans-serif;font-size:30px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.kpi .lbl{color:var(--muted);font-size:11px;margin-top:8px;text-transform:uppercase;letter-spacing:.6px;font-weight:600}
.kpi .foot{color:var(--faint);font-size:12px;margin-top:4px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.split{display:grid;grid-template-columns:1.15fr 1fr;gap:22px;align-items:center}
@media(max-width:680px){.two,.split{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line2)}
td:first-child,th:first-child{color:var(--faint);width:26px;text-align:right;padding-right:6px;font-variant-numeric:tabular-nums}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
tbody tr:last-child td{border-bottom:0}
.hl{color:var(--gold)}
.note{color:var(--faint);font-size:12.5px;margin-top:12px}
.big2{font-family:Outfit,sans-serif;font-size:17px;font-weight:700}
.lead{display:flex;align-items:baseline;gap:10px;margin-bottom:4px}
.lead .g{font-family:Outfit,sans-serif;font-size:34px;font-weight:800;color:var(--gold);font-variant-numeric:tabular-nums}
.drow{display:flex;align-items:center;gap:10px;padding:6px 0;font-size:13px}
.dt{width:92px;color:var(--text2)}
.dbar{flex:1;height:8px;background:var(--line2);border-radius:99px;overflow:hidden}
.dbar i{display:block;height:100%;background:linear-gradient(90deg,var(--gold),var(--red));border-radius:99px}
.dn{width:24px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
.stat{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--line2);font-size:14px}
.stat:last-child{border-bottom:0}.stat b{font-variant-numeric:tabular-nums}
.foota{margin-top:44px;color:var(--faint);font-size:12px;text-align:center}
.foota a{color:var(--green);text-decoration:none}
"""
    pending = m["n_matches"] < 104
    tail = (" One match to go — the final completes these numbers." if pending else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>World Cup 2026 by the numbers</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
<style>{css}</style></head><body>
<span class="eyebrow">The tournament in numbers</span>
<h1>World Cup 2026, by the numbers</h1>
<p class="sub">Everything the first 48-team World Cup produced across <b>{m['n_matches']}</b> matches —
goals, creators, cards and knockout drama, from the match data.{tail}</p>

<div class="cards">{kpis}</div>

<h2>The goals</h2>
<div class="split">
<div class="panel">
<div class="lead"><span class="g">{lead['g'] if lead else ''}</span>
<span class="big2">{_html.escape(lead['name']) if lead else ''}</span></div>
<p class="note" style="margin:0 0 10px">Golden Boot lead. Goals, then assists breaks the tie.</p>
<table><thead><tr><th></th><th>Player</th><th class="n">G</th><th class="n">A</th></tr></thead>
<tbody>{sc_rows}</tbody></table>
</div>
<div class="panel">
<h3 style="margin:0 0 4px;font-size:14px;color:var(--text2)">When goals happen</h3>
<p class="note" style="margin:0 0 6px">Goals by 15-minute band — the game bursts open late.</p>
{bands_svg(m['bands'])}
</div>
</div>

<h2>Creators &amp; chaos</h2>
<div class="two">
<div class="panel"><h3 style="margin:0 0 8px;font-size:14px;color:var(--text2)">🅰️ Most assists</h3>
<table><tbody>{as_rows}</tbody></table></div>
<div class="panel"><h3 style="margin:0 0 10px;font-size:14px;color:var(--text2)">🟨 Most-carded teams</h3>
{disc}
<p class="note">{m['cards']['yellow']} yellows &amp; {m['cards']['red']} reds in total. {worst_txt}</p></div>
</div>

<h2>Results &amp; records</h2>
<div class="panel">
<div class="stat"><span>Biggest win</span><b>{_html.escape(m['biggest']['s'])}</b></div>
<div class="stat"><span>Highest-scoring match</span><b>{_html.escape(m['highest']['s'])} <span class="hl">({m['highest']['t']} goals)</span></b></div>
<div class="stat"><span>Result split (H / D / A)</span><b>{m['hda']['home']} / {m['hda']['draw']} / {m['hda']['away']}</b></div>
<div class="stat"><span>Total corners</span><b>{m['totals']['corners']:,}</b></div>
<div class="stat"><span>Total shots</span><b>{m['totals']['shots']:,}</b></div>
<div class="stat"><span>Penalty shootouts</span><b>{sh_txt}</b></div>
</div>

<p class="foota">Data: martj42 results + Highlightly · <a href="{SITE}">{SITE.replace('https://','')}</a></p>
</body></html>"""


def main():
    try:
        m = compute()
    except FileNotFoundError as e:
        print("competition stats skipped:", e)
        return
    (OUT / "competition.html").write_text(build(m), encoding="utf-8")
    print(f"wrote outputs/competition.html — {m['goals']} goals in {m['n_matches']} matches, "
          f"lead {m['scorers'][0]['name'] if m['scorers'] else '-'}")


if __name__ == "__main__":
    main()
