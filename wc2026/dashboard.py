"""Interactive dashboard v2 (self-contained HTML + JS, no external JS libraries).

Beautiful, interactive single page: real flag images, a sortable/searchable
ranking, a click-through team profile (path + opponents + history card), an
in-browser **Match Lab** (head-to-head predictor, computed client-side from the
exported Poisson model), top-3 scorelines per match, the drawn bracket, extra
analyses (group of death, dark horses, likely finals) and the backtesting.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import analysis as AN
from . import facts as FACTS
from . import goldenboot as GB
from . import predictions as PR
from .viz import CODES, FLAGS


def _num(x, default=1.3):
    x = float(x)
    return x if x == x else default  # NaN guard


def collect(bundle, trained, table, val=None, backtests=None,
            gb_before=None, mode_label=None, evolution=None) -> dict:
    """Gathers all the data (JSON-serialisable) that the dashboard shows."""
    teams = [{
        "team": r["team"], "group": r["group"], "elo": int(r["elo"]),
        "flag": FLAGS.get(r["team"], "🏳️"), "code": CODES.get(r["team"], ""),
        "p_win_group": float(r["p_win_group"]), "p_1st": float(r["p_1st"]),
        "p_2nd": float(r["p_2nd"]), "p_3rd": float(r["p_3rd"]),
        "p_4th": float(r["p_4th"]), "exp_points": float(r["exp_points"]),
        "p_ko": float(r["p_ko"]), "p_r16": float(r["p_r16"]),
        "p_qf": float(r["p_qf"]), "p_sf": float(r["p_sf"]),
        "p_final": float(r["p_final"]), "p_champion": float(r["p_champion"]),
        "opponents": PR.opponents_for(table, r["team"]),
    } for _, r in table.iterrows()]

    matches = PR.match_predictions(bundle, trained, topn=3)

    bracket = [{"label": label,
                "matches": [{"home": m["home"], "away": m["away"],
                             "adv": m["advances"], "p": m["p"]} for m in ms]}
               for label, ms in PR.most_likely_bracket(table, trained)]

    gb = GB.predict(bundle, table, before=gb_before, topn=15)
    golden = [{"scorer": r["scorer"], "team": r["team"],
               "flag": FLAGS.get(r["team"], "🏳️"),
               "goals": int(r["recent_goals"]), "proj": float(r["proj_goals"])}
              for _, r in gb.iterrows()]

    fav, second = table.iloc[0], table.iloc[1]
    finalists = table.sort_values("p_final", ascending=False).head(2)

    validation = None
    if val is not None:
        validation = {k: {"acc": float(val.loc[k, "accuracy"]),
                          "logloss": float(val.loc[k, "log_loss"]),
                          "rps": float(val.loc[k, "RPS"])} for k in val.index}

    # export the Poisson model + per-team state so the browser can run the H2H lab
    mdl = trained["model"]
    model = {"beta": [float(x) for x in mdl.beta], "features": list(mdl.features),
             "continuous": list(mdl.continuous), "rho": float(mdl.rho),
             "mu": {c: float(mdl.mu[c]) for c in mdl.continuous},
             "sd": {c: float(mdl.sd[c]) for c in mdl.continuous}}
    names = set(table["team"])
    state = {t: {"elo": float(s["elo"]), "gf": _num(s["gf_form"]),
                 "ga": _num(s["ga_form"])}
             for t, s in trained["state"].items() if t in names}

    return {
        "n_sims": int(table.attrs.get("n_sims", 0)),
        "teams": teams, "matches": matches, "bracket": bracket,
        "favorite": {"team": fav["team"], "flag": FLAGS.get(fav["team"], "🏳️"),
                     "code": CODES.get(fav["team"], ""), "p": float(fav["p_champion"])},
        "second": {"team": second["team"], "flag": FLAGS.get(second["team"], "🏳️"),
                   "code": CODES.get(second["team"], ""), "p": float(second["p_champion"])},
        "finalists": [{"team": t, "flag": FLAGS.get(t, "🏳️"), "code": CODES.get(t, "")}
                      for t in finalists["team"]],
        "validation": validation, "backtests": backtests or [],
        "golden_boot": golden, "params": trained.get("params", {}),
        "mode_label": mode_label, "evolution": evolution,
        "facts": (FACTS.fun_facts(trained["df_elo"], team=fav["team"])
                  if trained.get("df_elo") is not None else None),
        "model": model, "state": state,
        "analysis": {
            "group_of_death": AN.group_of_death(table),
            "dark_horses": AN.dark_horses(table),
            "likely_finals": AN.likely_finals(table),
            "champion_ci": AN.champion_ci(table),
        },
    }


_CSS = """
*{box-sizing:border-box}
:root{--bg:#0c1018;--panel:#161d2b;--panel2:#1c2536;--line:#243049;
--text:#eef1f7;--muted:#8b95ab;--green:#00e0a4;--gold:#ffd34d;--red:#ff6b6b}
body{margin:0;background:linear-gradient(180deg,#0c1018,#0b0f17 60%);color:var(--text);
font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:0 18px 80px}
.hero{padding:46px 0 24px;text-align:center;
background:radial-gradient(900px 320px at 50% -40px,rgba(0,224,164,.16),transparent)}
.hero h1{font-size:clamp(26px,4vw,40px);margin:0 0 8px;font-weight:800;letter-spacing:-.5px}
.hero .sub{color:var(--muted);margin:0;font-size:15px}
.badge{display:inline-block;margin-bottom:14px;padding:5px 14px;border-radius:999px;
background:rgba(0,224,164,.12);color:var(--green);font-size:12px;font-weight:700;
border:1px solid rgba(0,224,164,.3);letter-spacing:.5px}
h2{font-size:21px;margin:42px 0 16px;font-weight:700;display:flex;align-items:center;gap:8px}
h2::before{content:"";width:4px;height:22px;background:linear-gradient(var(--green),#00b083);
border-radius:3px}
.tag{font-size:11px;color:var(--muted);font-weight:500;background:var(--panel);
padding:2px 9px;border-radius:999px;border:1px solid var(--line)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:18px 0}
.kpi{background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);
border-radius:16px;padding:18px;transition:transform .2s,border-color .2s}
.kpi:hover{transform:translateY(-3px);border-color:rgba(0,224,164,.4)}
.kpi .big{font-size:27px;font-weight:800;display:flex;align-items:center;gap:8px}
.kpi .lbl{color:var(--muted);font-size:12.5px;margin-top:4px}
.flag{width:24px;height:17px;border-radius:3px;object-fit:cover;vertical-align:middle;
box-shadow:0 1px 3px rgba(0,0,0,.5);background:#243049}
.flag.lg{width:40px;height:28px;border-radius:5px}
.flag.sm{width:18px;height:13px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px}
.layout{display:grid;grid-template-columns:1.35fr 1fr;gap:18px;align-items:start}
@media(max-width:880px){.layout{grid-template-columns:1fr}}
.sticky{position:sticky;top:14px}
input.search{width:100%;background:var(--panel);border:1px solid var(--line);color:var(--text);
border-radius:10px;padding:9px 12px;font-size:14px;margin-bottom:10px}
input.search:focus{outline:none;border-color:var(--green)}
table{width:100%;border-collapse:collapse}
th,td{padding:8px 9px;text-align:left;font-size:13.5px;border-bottom:1px solid var(--line)}
th{color:var(--muted);cursor:pointer;user-select:none;font-weight:600;white-space:nowrap}
th:hover{color:var(--text)}
tr.row{cursor:pointer;transition:background .15s}
tr.row:hover{background:var(--panel2)}
tr.sel{background:rgba(255,211,77,.1)}tr.sel td:nth-child(2){color:var(--gold);font-weight:700}
.bar{position:relative;background:var(--line);border-radius:6px;height:20px;overflow:hidden}
.fill{position:absolute;left:0;top:0;height:100%;border-radius:6px;
transition:width .6s cubic-bezier(.22,1,.36,1)}
.bar span{position:relative;padding-left:8px;line-height:20px;font-size:11.5px;font-weight:700}
.row-team{display:flex;align-items:center;gap:8px}
.prow{display:flex;align-items:center;gap:10px;font-size:13px;margin:7px 0}
.prow .lb{width:108px;color:var(--muted);flex-shrink:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
.group{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.group h3{margin:0 0 10px;font-size:14px;color:var(--muted);letter-spacing:.5px}
.g-row{display:flex;align-items:center;gap:8px;margin:7px 0;font-size:13px}
.g-row .nm{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;
align-items:center;gap:7px}
.g-row .pos{width:16px;color:#5d6a85;font-weight:700}
.mini{flex:0 0 56px;height:7px;background:var(--line);border-radius:4px;overflow:hidden}
.mini>div{height:100%;background:var(--green);border-radius:4px;transition:width .6s}
select{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;
padding:8px 12px;font-size:14px}
.lab{display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center;
background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);
border-radius:16px;padding:20px}
@media(max-width:680px){.lab{grid-template-columns:1fr;text-align:center}}
.lab .vs{font-weight:800;color:var(--muted)}
.wdl{display:flex;height:30px;border-radius:8px;overflow:hidden;margin:14px 0 8px;font-size:12px;
font-weight:700}
.wdl>div{display:flex;align-items:center;justify-content:center;color:#0c1018;min-width:28px}
.scorelines{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;color:var(--muted);font-size:13px}
.scorelines b{color:var(--text)}
.mlab-out{margin-top:16px}
.brk{display:flex;gap:14px;overflow-x:auto;padding-bottom:12px}
.bcol{min-width:184px;flex:0 0 auto}
.bcol h4{margin:0 0 10px;font-size:11px;color:var(--muted);text-align:center;letter-spacing:.5px;text-transform:uppercase}
.bmatch{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px;
margin-bottom:10px;font-size:12.5px}
.bteam{display:flex;align-items:center;gap:6px;padding:3px 0}
.bteam.win{color:var(--green);font-weight:700}.bteam.out{color:#6b7488}
.bteam .p{margin-left:auto;font-size:10px;color:#5d6a85}
.mtable td:nth-child(3){min-width:170px}
.score3{color:var(--muted);font-size:12px}.score3 b{color:var(--text)}
.note{color:var(--muted);font-size:13.5px}
.bt{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:12px}
.bt .hit{color:var(--green);font-weight:700}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:6px 12px;
font-size:13px;display:flex;align-items:center;gap:7px}
.btn{background:linear-gradient(160deg,var(--green),#00b083);color:#06231b;border:none;
border-radius:10px;padding:10px 18px;font-size:14px;font-weight:700;cursor:pointer;transition:filter .2s}
.btn:hover{filter:brightness(1.1)}
.foot{margin-top:36px;color:#5d6a85;font-size:12px;text-align:center}
a{color:var(--green)}
"""

_JS = r"""
const D=DATA, T=D.teams, byName=Object.fromEntries(T.map(t=>[t.team,t]));
const STAGES=[["p_ko","Last 32"],["p_r16","Round of 16"],["p_qf","Quarter-finals"],
["p_sf","Semi-finals"],["p_final","Final"],["p_champion","Champion"]];
const pct=x=>(x*100).toFixed(1)+"%", pc0=x=>(x*100).toFixed(0)+"%";
let sortKey="p_champion",sortDir=-1,selected=D.teams[0].team,query="";

function flag(team,cls){const t=byName[team]||{};const c=t.code!==undefined?t.code:(D.favorite.team===team?D.favorite.code:"");
  if(c) return `<img class="flag ${cls||''}" src="https://flagcdn.com/w40/${c}.png" alt="" loading="lazy">`;
  return `<span>${t.flag||"🏳️"}</span>`;}
function bar(p,color,label){const w=Math.min(Math.max(p*100,1),100);
  return `<div class="bar"><div class="fill" style="width:${w}%;background:${color}"></div>
  <span>${label==null?pct(p):label}</span></div>`;}

function renderRanking(){
  const cols=[["p_champion","Champion"],["p_final","Final"],["p_sf","Semis"],["p_qf","QF"],["p_ko","Adv."]];
  let rows=[...T].sort((a,b)=>sortDir*(a[sortKey]-b[sortKey]));
  if(query) rows=rows.filter(t=>t.team.toLowerCase().includes(query));
  let h=`<table><thead><tr><th>#</th><th>Team</th><th>Gr</th><th>Elo</th>`;
  cols.forEach(c=>h+=`<th data-k="${c[0]}">${c[1]}${sortKey===c[0]?(sortDir<0?" ▾":" ▴"):""}</th>`);
  h+=`</tr></thead><tbody>`;
  rows.forEach((t,i)=>{h+=`<tr class="row${t.team===selected?' sel':''}" data-t="${t.team}">
  <td>${i+1}</td><td><span class="row-team">${flag(t.team)} ${t.team}</span></td>
  <td>${t.group}</td><td>${t.elo}</td>
  <td style="min-width:150px">${bar(t.p_champion,"#00e0a4")}</td>
  <td>${pct(t.p_final)}</td><td>${pct(t.p_sf)}</td><td>${pct(t.p_qf)}</td><td>${pct(t.p_ko)}</td></tr>`;});
  h+=`</tbody></table>`;
  const c=document.getElementById("ranking");c.innerHTML=h;
  c.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{const k=th.dataset.k;
    if(k===sortKey)sortDir*=-1;else{sortKey=k;sortDir=-1;}renderRanking();});
  c.querySelectorAll("tr.row").forEach(tr=>tr.onclick=()=>{selected=tr.dataset.t;
    renderRanking();renderDetail();renderGroups();});
}

function renderDetail(){
  const t=byName[selected];
  let h=`<div style="display:flex;align-items:center;gap:12px">${flag(t.team,"lg")}
  <div><div style="font-size:21px;font-weight:800">${t.team}</div>
  <div class="tag">Group ${t.group} · Elo ${t.elo} · ${t.exp_points.toFixed(1)} exp. pts</div></div></div>
  <div style="margin-top:14px">`;
  STAGES.forEach(s=>{h+=`<div class="prow"><span class="lb">${s[1]}</span>${bar(t[s[0]],"#ffd34d")}</div>`;});
  h+=`</div>`;
  if(t.opponents&&Object.keys(t.opponents).length){
    h+=`<div style="margin-top:16px;font-weight:700">Most likely opponents</div>`;
    for(const[rnd,info] of Object.entries(t.opponents)){
      const opps=info.opponents.map(o=>`${flag(o.team,"sm")} ${o.team} <span style="color:#8b95ab">${pc0(o.p_cond)}</span>`).join(" · ");
      h+=`<div class="note" style="margin-top:5px"><b>${rnd}</b> <span class="tag">reaches ${pc0(info.p_reach)}</span><br>${opps}</div>`;}
  }
  document.getElementById("detail").innerHTML=h;
}

function renderGroups(){
  const groups={};T.forEach(t=>(groups[t.group]=groups[t.group]||[]).push(t));
  let h="";Object.keys(groups).sort().forEach(L=>{
    const g=groups[L].slice().sort((a,b)=>b.exp_points-a.exp_points);
    h+=`<div class="group"><h3>GROUP ${L}</h3>`;
    g.forEach((t,i)=>{const sel=t.team===selected?'style="color:#ffd34d;font-weight:700"':'';
      h+=`<div class="g-row"><span class="pos">${i+1}</span>
      <span class="nm" ${sel}>${flag(t.team,"sm")} ${t.team}</span>
      <span style="color:#8b95ab;width:34px;text-align:right">${pc0(t.p_1st)}</span>
      <div class="mini"><div style="width:${Math.min(t.p_ko*100,100)}%"></div></div></div>`;});
    h+=`</div>`;});
  document.getElementById("groups").innerHTML=h;
}

function renderMatches(){
  const sel=document.getElementById("mfilter").value;
  const ms=D.matches.filter(m=>sel==="all"||m.group===sel);
  let h=`<table class="mtable"><thead><tr><th>Date</th><th>Gr</th><th>Match</th>
  <th>Top scorelines</th><th>W/D/L</th></tr></thead><tbody>`;
  ms.forEach(m=>{const top=m.top.map((s,i)=>`<b>${s.score}</b> ${pc0(s.p)}`).join(" · ");
    h+=`<tr><td>${m.date.slice(5)}</td><td>${m.group}</td>
    <td><span class="row-team">${flag(m.home,"sm")} ${m.home} – ${m.away} ${flag(m.away,"sm")}</span></td>
    <td class="score3">${top}</td>
    <td style="color:#8b95ab">${pc0(m.p_home)}/${pc0(m.p_draw)}/${pc0(m.p_away)}</td></tr>`;});
  h+=`</tbody></table>`;document.getElementById("matches").innerHTML=h;
}

/* ---- Match Lab: client-side Poisson (Dixon-Coles) head-to-head ---- */
function poisPmf(k,l){let p=Math.exp(-l);for(let i=1;i<=k;i++)p*=l/i;return p;}
function lambdas(a,b){const M=D.model,sa=D.state[a],sb=D.state[b];
  const rows=[{elo_self:sa.elo,elo_opp:sb.elo,venue:0,is_friendly:0,self_gf:sa.gf,opp_ga:sb.ga},
              {elo_self:sb.elo,elo_opp:sa.elo,venue:0,is_friendly:0,self_gf:sb.gf,opp_ga:sa.ga}];
  return rows.map(r=>{let eta=M.beta[0];M.features.forEach((f,i)=>{let v=r[f];
    if(M.continuous.includes(f))v=(v-M.mu[f])/M.sd[f];eta+=M.beta[i+1]*v;});return Math.exp(eta);});}
function h2h(a,b){const [la,lb]=lambdas(a,b),rho=D.model.rho,N=10;
  let G=[];for(let i=0;i<=N;i++){G[i]=[];for(let j=0;j<=N;j++)G[i][j]=poisPmf(i,la)*poisPmf(j,lb);}
  G[0][0]*=1-la*lb*rho;G[0][1]*=1+la*rho;G[1][0]*=1+lb*rho;G[1][1]*=1-rho;
  let s=0;G.forEach(r=>r.forEach(v=>s+=v));
  let ph=0,pd=0,pa=0,flat=[];
  for(let i=0;i<=N;i++)for(let j=0;j<=N;j++){const v=G[i][j]/s;
    if(i>j)ph+=v;else if(i===j)pd+=v;else pa+=v;flat.push([i,j,v]);}
  flat.sort((x,y)=>y[2]-x[2]);
  return {la,lb,ph,pd,pa,top:flat.slice(0,3).map(([i,j,v])=>({s:i+"-"+j,p:v}))};}
function renderLab(){
  const a=document.getElementById("labA").value,b=document.getElementById("labB").value;
  if(a===b){document.getElementById("lab-out").innerHTML="<p class='note' style='text-align:center'>Pick two different teams.</p>";return;}
  const r=h2h(a,b);
  const W=r.ph*100,Dr=r.pd*100,L=r.pa*100;
  const top=r.top.map(s=>`<span><b>${s.s}</b> ${pc0(s.p)}</span>`).join("");
  document.getElementById("lab-out").innerHTML=`
  <div class="wdl"><div style="width:${W}%;background:#00e0a4">${W.toFixed(0)}%</div>
  <div style="width:${Dr}%;background:#8b95ab">${Dr.toFixed(0)}%</div>
  <div style="width:${L}%;background:#ffd34d">${L.toFixed(0)}%</div></div>
  <div class="note" style="text-align:center">${flag(a,'sm')} win · draw · ${flag(b,'sm')} win
   &nbsp;|&nbsp; expected goals <b>${r.la.toFixed(2)} – ${r.lb.toFixed(2)}</b></div>
  <div class="scorelines" style="margin-top:10px">Most likely scores: ${top}</div>`;
}
function buildLab(){
  const opts=T.map(t=>`<option value="${t.team}">${t.team}</option>`).join("");
  const sel=id=>`<select id="${id}" onchange="renderLab()">${opts}</select>`;
  document.getElementById("lab").innerHTML=`<div>${sel("labA")}</div>
  <div class="vs">VS</div><div style="text-align:right">${sel("labB")}</div>`;
  document.getElementById("labA").value=D.favorite.team;
  document.getElementById("labB").value=D.second.team;
  renderLab();
}

function renderBracket(){
  let h="";D.bracket.forEach(rd=>{h+=`<div class="bcol"><h4>${rd.label}</h4>`;
    rd.matches.forEach(m=>{const hw=m.adv===m.home;
      h+=`<div class="bmatch"><div class="bteam ${hw?'win':'out'}">${flag(m.home,'sm')} ${m.home}<span class="p">${pc0(m.p)}</span></div>
      <div class="bteam ${!hw?'win':'out'}">${flag(m.away,'sm')} ${m.away}</div></div>`;});
    h+=`</div>`;});
  document.getElementById("bracket").innerHTML=h;
}

function renderAnalysis(){
  const a=D.analysis;if(!a)return;
  const god=a.group_of_death[0];
  const dh=a.dark_horses.map(d=>`<div class="chip">${flag(d.team,'sm')} ${d.team} <span style="color:#8b95ab">${pc0(d.p_qf)} QF</span></div>`).join("");
  const lf=a.likely_finals.map(f=>`<div class="chip">${flag(f.a,'sm')} ${f.a} v ${f.b} ${flag(f.b,'sm')} <span style="color:#8b95ab">${pc0(f.p)}</span></div>`).join("");
  document.getElementById("analysis").innerHTML=`
  <p class="note"><b>💀 Group of death:</b> Group ${god.group} — ${flag(god.teams[0],'sm')} ${god.teams[0]} & ${flag(god.teams[1],'sm')} ${god.teams[1]}
   (${pc0(god.sum_champion)} combined title odds).</p>
  <p class="note" style="margin-top:12px"><b>🐎 Dark horses</b> (lower seeds, deep runs):</p><div class="chips">${dh}</div>
  <p class="note" style="margin-top:14px"><b>🏆 Most likely finals:</b></p><div class="chips">${lf}</div>`;
}

function renderBacktest(){
  if(!D.backtests.length){document.getElementById("backtest").style.display="none";return;}
  let h="";D.backtests.forEach(b=>{const top=b.top5.map(x=>`${x.team} ${pc0(x.p)}`).join(" · ");
    h+=`<div class="bt"><b>${b.year} World Cup</b> — actual champion: <span class="hit">${b.champ_real}</span>
    (was the #${b.champ_rank} favourite, ${pc0(b.champ_p)})<br>
    <span class="note">Skill over ${b.n} matches: accuracy ${pc0(b.acc)} (naive ${pc0(b.acc_naive)})
    · RPS ${b.rps.toFixed(3)} (naive ${b.rps_naive.toFixed(3)})<br>Pre-tournament favourites: ${top}</span></div>`;});
  document.getElementById("backtest").innerHTML=h;
}

function renderGolden(){
  const g=D.golden_boot||[];if(!g.length)return;const mx=g[0].proj;
  let h=`<table><thead><tr><th>#</th><th>Player</th><th>Team</th><th>Recent</th><th>World Cup projection</th></tr></thead><tbody>`;
  g.forEach((p,i)=>{h+=`<tr><td>${i+1}</td><td>${p.scorer}</td>
  <td><span class="row-team">${flag(p.team,'sm')} ${p.team}</span></td><td>${p.goals}</td>
  <td style="min-width:150px">${bar(p.proj/mx,"#00e0a4",p.proj.toFixed(1)+" goals")}</td></tr>`;});
  h+=`</tbody></table>`;document.getElementById("golden").innerHTML=h;
}

function simulateOne(){let r=Math.random(),acc=0,champ=T[T.length-1].team;
  for(const t of T){acc+=t.p_champion;if(r<=acc){champ=t.team;break;}}
  document.getElementById("sim-out").innerHTML=`🎲 In this timeline: <b>${flag(champ,'sm')} ${champ}</b> lifts the trophy!`;}

renderRanking();renderDetail();renderGroups();renderMatches();buildLab();
renderBracket();renderAnalysis();renderBacktest();renderGolden();
document.getElementById("mfilter").onchange=renderMatches;
document.getElementById("search").oninput=e=>{query=e.target.value.toLowerCase();renderRanking();};
document.getElementById("sim-btn").onclick=simulateOne;
"""


def _facts_html(f: dict) -> str:
    if not f or "team_card" not in f:
        return ""
    g, c, pk = f["global"], f["team_card"], f["penalties"]
    ups = "".join(
        f"<tr><td>{u['date'][:4]}</td><td>{u['winner']} {u['score']} {u['loser']}</td>"
        f"<td style='color:#8b95ab'>{u['tournament'][:24]}</td><td>+{u['gap']}</td></tr>"
        for u in f["upsets"])
    kings = " · ".join(f"{x['team']} {x['pct']}%" for x in pk["best"][:4])
    gt, ht = f["goal_trend"][-1], f["home_trend"][-1]
    peak = f"{c['peak_elo'][0]} ({c['peak_elo'][1]})" if c.get("peak_elo") else "—"
    card = (f"<div class='kpi' style='grid-column:1/-1'><div class='lbl'>History card — "
            f"{c['team']} (the current favourite)</div><div class='note' style='margin-top:6px'>"
            f"<b>{c['w']}W-{c['d']}D-{c['l']}L</b> ({c['win_pct']}%) · biggest win "
            f"{c['biggest_win']} · worst loss {c['worst_loss']} · longest unbeaten run "
            f"<b>{c['longest_unbeaten']}</b> · rival {c['rival']} ({c['rival_record']}) · "
            f"peak Elo {peak}</div></div>")
    return (
        "<h2>🕰️ Historical facts <span class='tag'>1872–2026</span></h2>"
        f"<p class='note'>Biggest win ever: <b>{g['biggest_win'][0]}</b> ({g['biggest_win'][1]}) · "
        f"top international scorer: <b>{g['top_scorer'][0]}</b> ({g['top_scorer'][1]} goals) · "
        f"{g['n_matches']:,} matches, {g['total_goals']:,} goals.</p>"
        f"<div class='cards'>{card}</div>"
        "<p class='note'>Biggest upsets ever (underdog won, by Elo):</p>"
        "<table><tr><th>Year</th><th>Match</th><th>Tournament</th><th>Δelo</th></tr>"
        f"{ups}</table>"
        f"<p class='note'>👟 Shooting first wins <b>{pk['first_shooter_pct']}%</b> of shootouts · "
        f"penalty kings: {kings}. ⚽ Goals/match in the {gt['decade']}s: <b>{gt['avg_goals']}</b> · "
        f"home wins in the {ht['decade']}s: <b>{ht['home_win_pct']}%</b>.</p>"
    )


def build_interactive(data: dict, out_path) -> Path:
    val = data.get("validation")
    val_html = ""
    if val and "full" in val and "naive" in val:
        f, nv = val["full"], val["naive"]
        val_html = (f'<p class="note" style="text-align:center;margin-top:6px">Validated on '
                    f'~4500 unseen matches: <b>{f["acc"]*100:.0f}%</b> accuracy · '
                    f'<b>RPS {f["rps"]:.3f}</b> (naive {nv["rps"]:.3f}) — bookmaker level.</p>')

    ev = data.get("evolution")
    ev_html = ""
    if ev and (ev.get("has_chart") or ev.get("movers")):
        parts = ["<h2>📊 Odds over time <span class='tag'>since the last update</span></h2>"]
        if ev.get("has_chart"):
            parts.append("<img src='odds_evolution.png' alt='odds over time' "
                         "style='width:100%;border-radius:14px'>")
        if ev.get("movers"):
            rows = "".join(
                f"<tr><td>{m['team']}</td><td>{m['prev']*100:.1f}%</td>"
                f"<td>{m['now']*100:.1f}%</td><td style='color:{'#00e0a4' if m['delta']>=0 else '#ff6b6b'};"
                f"font-weight:700'>{m['delta']*100:+.1f}</td></tr>" for m in ev["movers"])
            parts.append("<p class='note'>Biggest movers:</p><table><tr><th>Team</th>"
                         f"<th>Before</th><th>Now</th><th>Δ</th></tr>{rows}</table>")
        ev_html = "".join(parts)

    mgroups = sorted({m["group"] for m in data["matches"]})
    opts = '<option value="all">All groups</option>' + "".join(
        f'<option value="{g}">Group {g}</option>' for g in mgroups)
    fav, sec, fin = data["favorite"], data["second"], data["finalists"]
    contenders = sum(1 for t in data["teams"] if t["p_champion"] > 0.05)
    fl = lambda d: (f'<img class="flag" src="https://flagcdn.com/w40/{d["code"]}.png">'
                    if d.get("code") else d.get("flag", ""))
    mode = (data.get("mode_label") + " · ") if data.get("mode_label") else ""

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>FIFA World Cup 2026 — ML Prediction</title>"
        f"<style>{_CSS}</style></head><body><div class='wrap'>"
        "<div class='hero'><div class='badge'>MACHINE LEARNING PREDICTION</div>"
        "<h1>🏆 FIFA World Cup 2026</h1>"
        f"<p class='sub'>{mode}Poisson (Dixon-Coles) + Elo · {data['n_sims']:,} Monte Carlo "
        f"simulations · click a team or try the Match Lab</p>"
        f"{val_html}</div>"
        "<div class='cards'>"
        f"<div class='kpi'><div class='big'>{fl(fav)} {fav['p']*100:.1f}%</div>"
        f"<div class='lbl'>Favourite: {fav['team']}</div></div>"
        f"<div class='kpi'><div class='big'>{fl(sec)} {sec['p']*100:.1f}%</div>"
        f"<div class='lbl'>2nd favourite: {sec['team']}</div></div>"
        f"<div class='kpi'><div class='big'>{fl(fin[0])} {fl(fin[1])}</div>"
        f"<div class='lbl'>Most likely finalists</div></div>"
        f"<div class='kpi'><div class='big'>{contenders}</div>"
        f"<div class='lbl'>teams above 5% to win</div></div></div>"
        f"{ev_html}"
        "<h2>🆚 Match Lab <span class='tag'>head-to-head, live</span></h2>"
        "<div class='lab' id='lab'></div><div class='mlab-out' id='lab-out'></div>"
        "<h2>Ranking & path <span class='tag'>click a row</span></h2>"
        "<div class='layout'><div><input class='search' id='search' "
        "placeholder='🔎 Search a team…'><div id='ranking'></div></div>"
        "<div class='panel sticky' id='detail'></div></div>"
        "<h2>The 12 groups <span class='tag'>expected standings</span></h2>"
        "<div class='grid' id='groups'></div>"
        "<h2>Group-stage matches <span class='tag'>top-3 scorelines</span></h2>"
        f"<select id='mfilter'>{opts}</select><div id='matches' style='margin-top:10px'></div>"
        "<h2>Most likely bracket <span class='tag'>favourites</span></h2>"
        "<div class='brk' id='bracket'></div>"
        "<h2>📈 Extra analysis</h2><div id='analysis'></div>"
        "<h2>🎲 Roll a tournament <span class='tag'>weighted by the odds</span></h2>"
        "<button class='btn' id='sim-btn'>Simulate one World Cup</button>"
        "<p class='note' id='sim-out' style='margin-top:12px'></p>"
        "<h2>Does it work? Backtesting 2018 + 2022</h2><div id='backtest'></div>"
        "<h2>👟 Golden Boot <span class='tag'>top scorer</span></h2><div id='golden'></div>"
        "<h2>📉 Elo through history <span class='tag'>1960 → 2026</span></h2>"
        "<img src='elo_race.gif' alt='Elo through history' style='width:100%;border-radius:14px'>"
        f"{_facts_html(data.get('facts'))}"
        "<p class='foot'>Data-driven model, just for fun. ⚽ "
        "Data: International football results 1872–2026.</p>"
        "</div>"
        f"<script>const DATA={json.dumps(data, ensure_ascii=False)};</script>"
        f"<script>{_JS}</script></body></html>"
    )
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path
