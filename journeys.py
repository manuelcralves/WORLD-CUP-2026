"""Team journeys — a per-team World Cup recap (predicted vs reality, full path, scorers) for all
48 teams. Self-contained outputs/journeys.html (Emerald Pitch) with a searchable team picker and
a detail panel, part of the retrospective section. `python journeys.py`; wired into run_pipeline.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from wc2026 import retro_common as RC
from wc2026.viz import CODES

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
CACHE = BASE / "api_cache"

_RANK = {"Round of 32": 1, "Round of 16": 2, "Quarter-finals": 3, "Semi-finals": 4, "Final": 5}
_SHORT = {"Round of 32": "R32", "Round of 16": "R16", "Quarter-finals": "QF",
          "Semi-finals": "SF", "3rd Place Final": "3rd", "Final": "Final"}


def _short(rnd: str) -> str:
    return "Group" if rnd.startswith("Group") else _SHORT.get(rnd, rnd)


def collect() -> list:
    from wc2026.richdata import HL_TO_MARTJ42 as HL

    def nm(t):
        return HL.get(t, t)

    # every match from the Highlightly cache (round + status + score), martj42-normalised
    rows = []
    with (CACHE / "wc_matches.csv").open(encoding="utf-8") as f:
        for m in csv.DictReader(f):
            sc = (m["score"] or "").split(" - ")
            if len(sc) != 2 or not sc[0].strip().isdigit() or not sc[1].strip().isdigit():
                continue
            rows.append({"date": m["date"], "round": m["round"], "status": m["status"],
                         "home": nm(m["home"]), "away": nm(m["away"]),
                         "hg": int(sc[0]), "ag": int(sc[1])})
    rows.sort(key=lambda r: r["date"])

    # placements from the final + 3rd-place play-off
    champ = runner = third = fourth = None
    for r in rows:
        if r["round"] == "Final":
            champ, runner = (r["home"], r["away"]) if r["hg"] >= r["ag"] else (r["away"], r["home"])
        elif r["round"] == "3rd Place Final":
            third, fourth = (r["home"], r["away"]) if r["hg"] >= r["ag"] else (r["away"], r["home"])

    # deepest stage each team reached (3rd-place play-off counts as the semis)
    reached: dict = {}
    matches: dict = defaultdict(list)
    for r in rows:
        rk = _RANK.get(r["round"], 4 if r["round"] == "3rd Place Final" else 0)
        tag = (" (pens)" if "penalties" in r["status"]
               else " (a.e.t.)" if "extra time" in r["status"] else "")
        for t, opp, gf, ga in ((r["home"], r["away"], r["hg"], r["ag"]),
                               (r["away"], r["home"], r["ag"], r["hg"])):
            reached[t] = max(reached.get(t, 0), rk)
            matches[t].append({"stage": _short(r["round"]), "opp": opp, "code": CODES.get(opp, ""),
                               "gf": gf, "ga": ga, "tag": tag,
                               "res": "W" if gf > ga else "L" if gf < ga else "D"})

    # pre-tournament odds + expected knockout depth
    pre = (pd.read_csv(BASE / "outputs_pretournament" / "predictions.csv")
           .sort_values("p_champion", ascending=False).reset_index(drop=True))
    pre_rank = {t: i + 1 for i, t in enumerate(pre["team"])}
    pre_champ = dict(zip(pre["team"], pre["p_champion"]))
    exp_depth = {r.team: float(r.p_ko + r.p_r16 + r.p_qf + r.p_sf + r.p_final)
                 for r in pre.itertuples(index=False)}

    # scorers per team (WC-only goalscorers, keyed to the played WC games)
    res = pd.read_csv(BASE / "results.csv", parse_dates=["date"])
    wc = res[(res["tournament"] == "FIFA World Cup") & (res["date"].dt.year == 2026)]
    played = wc[wc["home_score"].notna()]
    keys = set(played["date"].dt.strftime("%Y-%m-%d") + "|" + played["home_team"] + "|" + played["away_team"])
    g = pd.read_csv(BASE / "goalscorers.csv", parse_dates=["date"])
    g = g[g["date"].dt.year == 2026].copy()
    g["k"] = g["date"].dt.strftime("%Y-%m-%d") + "|" + g["home_team"] + "|" + g["away_team"]
    g = g[g["k"].isin(keys) & ~g["own_goal"].astype(str).str.upper().eq("TRUE")]
    scorers: dict = {}
    for team, grp in g.groupby("team"):
        s = grp.groupby("scorer").size().sort_values(ascending=False)
        scorers[team] = [{"name": n, "g": int(c)} for n, c in s.items()]

    def finish(t):
        if t == champ:
            return "Champions 🏆", 7
        if t == runner:
            return "Runners-up", 6
        if t == third:
            return "3rd place 🥉", 5
        if t == fourth:
            return "4th place", 4
        return {3: ("Quarter-finals", 3), 2: ("Round of 16", 2),
                1: ("Round of 32", 1)}.get(reached.get(t, 0), ("Group stage", 0))

    def verdict(t):
        if t not in exp_depth:
            return "met", "as predicted"
        s = reached.get(t, 0) - exp_depth[t]
        return ("over", "overachieved") if s >= 0.75 else ("under", "fell short") if s <= -0.75 else ("met", "as predicted")

    teams = []
    for t in sorted(matches):
        flabel, frank = finish(t)
        vkey, vlabel = verdict(t)
        ms = matches[t]
        wins = [m for m in ms if m["res"] == "W"]
        big = max(wins, key=lambda m: m["gf"] - m["ga"], default=None)
        teams.append({
            "team": t, "code": CODES.get(t, ""),
            "pre_champ": round(pre_champ.get(t, 0) * 100, 1), "pre_rank": pre_rank.get(t),
            "finish": flabel, "frank": frank, "verdict": vkey, "vlabel": vlabel,
            "matches": ms, "scorers": scorers.get(t, []),
            "biggest": (f'{big["gf"]}–{big["ga"]} v {big["opp"]}' if big else None),
        })
    teams.sort(key=lambda x: (-x["frank"], -(x["pre_champ"] or 0), x["team"]))
    return teams


def build(teams: list) -> str:
    data = json.dumps(teams, ensure_ascii=False)
    css = """
*{box-sizing:border-box}
:root{--bg:#0a0e14;--panel:#131a26;--panel2:#1a2434;--panel3:#212c3e;--line:#263143;--line2:#1b2431;
--text:#f0f3f9;--text2:#cbd3e1;--muted:#8a95a9;--faint:#5d6a85;--green:#2ee6a6;--green2:#12c98d;
--gold:#ffcb5c;--red:#ff6b6b}
body{margin:0;color:var(--text);font-family:Inter,-apple-system,Segoe UI,Roboto,Arial,"Segoe UI Emoji",sans-serif;
line-height:1.55;-webkit-font-smoothing:antialiased;
background:radial-gradient(1100px 520px at 50% -150px,rgba(46,230,166,.10),transparent 70%),
linear-gradient(180deg,#0a0e14,#060910 80%);background-attachment:fixed;
max-width:960px;margin:0 auto;padding:52px 20px 80px}
.eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:11.5px;font-weight:700;letter-spacing:1.4px;
text-transform:uppercase;color:var(--green);background:rgba(46,230,166,.12);border:1px solid rgba(46,230,166,.28);
padding:6px 14px;border-radius:999px}
.eyebrow::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green)}
h1{font-family:Outfit,sans-serif;font-size:clamp(28px,5vw,42px);font-weight:800;letter-spacing:-1px;margin:16px 0 8px}
.sub{color:var(--muted);font-size:15px;max-width:640px;margin:0 0 20px}
.fl{width:20px;height:14px;border-radius:2px;object-fit:cover;vertical-align:middle;margin-right:7px}
.layout{display:grid;grid-template-columns:300px 1fr;gap:18px;align-items:start}
@media(max-width:720px){.layout{grid-template-columns:1fr}}
.q{width:100%;padding:10px 13px;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--text);font-size:14px;margin-bottom:10px}
.q::placeholder{color:var(--faint)}
#grid{display:flex;flex-direction:column;gap:5px;max-height:70vh;overflow-y:auto;padding-right:4px}
@media(max-width:720px){#grid{max-height:none}}
.chip{display:flex;align-items:center;gap:2px;width:100%;text-align:left;cursor:pointer;
background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line);border-radius:10px;padding:9px 11px;color:var(--text);font-size:13.5px}
.chip:hover{background:var(--panel2)}
.chip.sel{background:var(--panel2);border-color:var(--green)}
.chip.v-over{border-left-color:var(--green)}.chip.v-under{border-left-color:var(--red)}.chip.v-met{border-left-color:var(--gold)}
.chip .cn{font-weight:600;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chip .cf{font-size:10.5px;color:var(--muted);white-space:nowrap}
.detail{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 2px 8px rgba(0,0,0,.35);position:sticky;top:16px}
.dh{font-family:Outfit,sans-serif;font-size:24px;font-weight:800;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.dh .fl{width:30px;height:21px;margin-right:4px}
.dfin{font-size:12px;font-weight:700;padding:4px 11px;border-radius:999px;margin-left:auto}
.dfin.v-over{color:var(--green);background:rgba(46,230,166,.13)}.dfin.v-under{color:var(--red);background:rgba(255,107,107,.13)}.dfin.v-met{color:var(--gold);background:rgba(255,203,92,.13)}
.dv{color:var(--text2);font-size:14px;margin:12px 0 4px}.dv b{color:var(--text)}
.dsec{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin:20px 0 8px}
.path{display:flex;flex-direction:column;gap:4px}
.mrow{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:9px;background:var(--panel2);font-size:13.5px;border-left:3px solid var(--line)}
.mrow.r-W{border-left-color:var(--green)}.mrow.r-D{border-left-color:var(--gold)}.mrow.r-L{border-left-color:var(--red)}
.mst{width:42px;flex:none;font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase}
.mo{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.msc{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}.msc em{color:var(--faint);font-style:normal;font-size:11px}
.mr{width:16px;flex:none;text-align:center;font-weight:800;font-size:12px}
.mrow.r-W .mr{color:var(--green)}.mrow.r-D .mr{color:var(--gold)}.mrow.r-L .mr{color:var(--red)}
.dsc{color:var(--text2);font-size:14px;line-height:1.7}
.note{color:var(--faint)}
.foota{margin-top:44px;color:var(--faint);font-size:12px;text-align:center}.foota a{color:var(--green);text-decoration:none}
"""
    js = "const T=" + data + ";" + """
const grid=document.getElementById('grid'),detail=document.getElementById('detail'),q=document.getElementById('q');
const esc=s=>String(s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
const flag=c=>c?`<img class="fl" src="https://flagcdn.com/w40/${c}.png" alt="">`:'🏳️';
grid.innerHTML=T.map((t,i)=>`<button class="chip v-${t.verdict}" data-i="${i}">${flag(t.code)}<span class="cn">${esc(t.team)}</span><span class="cf">${esc(t.finish)}</span></button>`).join('');
function render(i){const t=T[i];
  const path=t.matches.map(m=>`<div class="mrow r-${m.res}"><span class="mst">${m.stage}</span><span class="mo">${flag(m.code)} ${esc(m.opp)}</span><span class="msc">${m.gf}–${m.ga}<em>${m.tag}</em></span><span class="mr">${m.res}</span></div>`).join('');
  const sc=t.scorers.length?t.scorers.map(s=>`${esc(s.name)}${s.g>1?' <b>('+s.g+')</b>':''}`).join(' · '):'<span class="note">no goals scored</span>';
  const rank=t.pre_rank?`pre-tournament <b>#${t.pre_rank}</b> (${t.pre_champ}% to win the title)`:'unrated pre-tournament';
  detail.innerHTML=`<div class="dh">${flag(t.code)} <span>${esc(t.team)}</span><span class="dfin v-${t.verdict}">${esc(t.finish)}</span></div>`
    +`<div class="dv">Model: ${rank} → <b>${esc(t.vlabel)}</b></div>`
    +`<div class="dsec">The path — ${t.matches.length} games</div><div class="path">${path}</div>`
    +`<div class="dsec">Goalscorers</div><div class="dsc">${sc}</div>`
    +(t.biggest?`<div class="dsec">Biggest win</div><div class="dsc">${esc(t.biggest)}</div>`:'');
  grid.querySelectorAll('.chip').forEach(c=>c.classList.toggle('sel',+c.dataset.i===i));}
grid.querySelectorAll('.chip').forEach(c=>c.onclick=()=>render(+c.dataset.i));
q.oninput=()=>{const s=q.value.toLowerCase();grid.querySelectorAll('.chip').forEach(c=>{c.style.display=T[+c.dataset.i].team.toLowerCase().includes(s)?'':'none';});};
render(0);
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>World Cup 2026 — team journeys</title>
{RC.og("World Cup 2026 — team journeys", "How every one of the 48 teams' World Cup went — predicted vs reality, the full path and the scorers.", "/outputs/journeys.html")}
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
<style>{css}{RC.NAV_CSS}</style></head><body>
{RC.nav('journeys')}
<span class="eyebrow">Team journeys</span>
<h1>How every team's World Cup went</h1>
<p class="sub">All 48 teams — where the model had them <b>before a ball was kicked</b>, how far they actually
went, their full path and their scorers. Pick a team; the colour marks over- / under-achievers.</p>
<div class="layout">
<div><input class="q" id="q" placeholder="🔎 Find a team…" autocomplete="off"><div id="grid"></div></div>
<div class="detail" id="detail"></div>
</div>
<p class="foota">Data: martj42 results + Highlightly · <a href="https://worldcup2026ml.pt">worldcup2026ml.pt</a></p>
<script>{js}</script>
</body></html>"""


def main():
    try:
        teams = collect()
    except FileNotFoundError as e:
        print("team journeys skipped:", e)
        return
    (OUT / "journeys.html").write_text(build(teams), encoding="utf-8")
    print(f"wrote outputs/journeys.html — {len(teams)} teams, champions: {teams[0]['team']}")


if __name__ == "__main__":
    main()
