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
from . import fifa as FIFA
from . import goals as GOALS
from . import goldenboot as GB
from . import predictions as PR
from . import schedule as SCH
from .tournament import HOSTS, LATER, OFFICIAL_GROUPS, R32, THIRD_ELIGIBLE
from .viz import CODES, FAVICON, FLAGS, SITE, elo_by_year


def _num(x, default=1.3):
    x = float(x)
    return x if x == x else default  # NaN guard


def _fixtures(bundle) -> list:
    """The 72 group fixtures: played ones carry their result, the rest are simulated."""
    tl = {t: L for L, ts in OFFICIAL_GROUPS.items() for t in ts}
    out = []
    for g in bundle["wc_played"].itertuples(index=False):
        out.append({"home": g.home_team, "away": g.away_team, "group": tl[g.home_team],
                    "neutral": bool(g.neutral), "played": True,
                    "hs": int(g.home_score), "as": int(g.away_score)})
    for g in bundle["wc_remaining"].itertuples(index=False):
        out.append({"home": g.home_team, "away": g.away_team, "group": tl[g.home_team],
                    "neutral": bool(g.neutral), "played": False})
    return out


def _standings(bundle) -> dict:
    """Current group points / goal-difference from the World Cup games played."""
    st = {}
    for g in bundle["wc_played"].itertuples(index=False):
        hs, as_ = int(g.home_score), int(g.away_score)
        for team, gf, ga in ((g.home_team, hs, as_), (g.away_team, as_, hs)):
            s = st.setdefault(team, {"played": 0, "pts": 0, "gf": 0, "ga": 0})
            s["played"] += 1
            s["gf"] += gf
            s["ga"] += ga
            s["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
    for s in st.values():
        s["gd"] = s["gf"] - s["ga"]
    return st


def collect(bundle, trained, table, val=None, backtests=None,
            gb_before=None, mode_label=None, evolution=None,
            odds_history=None, mega_backtest=None, played_review=None) -> dict:
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
        "fifa_rank": FIFA.rank_of(r["team"]),
        "opponents": PR.opponents_for(table, r["team"]),
    } for _, r in table.iterrows()]

    st = _standings(bundle)                      # live points from games played
    for t in teams:
        s = st.get(t["team"], {"played": 0, "pts": 0, "gd": 0, "gf": 0})
        t["played"], t["pts"] = s["played"], s["pts"]
        t["gd"], t["gf"] = s["gd"], s["gf"]

    matches = PR.match_predictions(bundle, trained, topn=3)

    bracket = [{"label": label,
                "matches": [{"home": m["home"], "away": m["away"],
                             "adv": m["advances"], "p": m["p"], "m": m["m"]}
                            for m in ms]}
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
    form = PR.recent_form(bundle, list(names))
    for t in teams:
        t["form"] = form.get(t["team"], "")
    # rich extras for the team page: last results, all-time top scorers, team card
    recents = PR.recent_results(bundle, list(names))
    df_elo = trained.get("df_elo")
    gsa = GB.load_goalscorers()
    gsa = gsa[~gsa["own_goal"]]
    scorers = {tm: [{"scorer": s, "goals": int(c)} for s, c in
                    gsa[gsa["team"] == tm]["scorer"].value_counts().head(5).items()]
               for tm in names}
    for t in teams:
        t["recent"] = recents.get(t["team"], [])
        t["scorers"] = scorers.get(t["team"], [])
        if df_elo is not None:
            t["card"] = FACTS.team_card(df_elo, t["team"])
    h2h = PR.h2h_records(bundle, list(names))

    # goal timelines for the played matches + the all-time "anatomy of a goal"
    goals_an = GOALS.scoring_analysis()
    top_scorers = []
    if played_review:
        from collections import Counter
        cnt = Counter()
        for m in played_review:
            m["goals"] = GOALS.match_goals(m["home"], m["away"], m["date"])
            for gl in m["goals"]:
                if not gl["og"]:
                    cnt[(gl["scorer"], gl["team"])] += 1
        top_scorers = [{"scorer": s, "team": t, "goals": c}
                       for (s, t), c in cnt.most_common(6)]
    goals_an["top_scorers"] = top_scorers

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
        "mega_backtest": mega_backtest,
        "golden_boot": golden, "params": trained.get("params", {}),
        "mode_label": mode_label, "evolution": evolution,
        "facts": (FACTS.fun_facts(trained["df_elo"], team=fav["team"])
                  if trained.get("df_elo") is not None else None),
        "model": model, "state": state, "h2h": h2h,
        "played_review": played_review,
        "shoot_b1": float(trained.get("shootout", {}).get("b1", 0.0)),
        "odds_history": odds_history,
        "elo_by_year": elo_by_year(bundle["matches"]),
        "kickoffs": SCH.all_lisbon(),
        "fifa": FIFA.compare(table),
        "goals": goals_an,
        "fixtures": _fixtures(bundle),
        "structure": {
            "groups": OFFICIAL_GROUPS,
            "r32": {str(k): [list(s) for s in v] for k, v in R32.items()},
            "third_elig": {str(k): v for k, v in THIRD_ELIGIBLE.items()},
            "later": {str(k): list(v) for k, v in LATER.items()},
            "hosts": sorted(HOSTS),
        },
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
body{margin:0;color:var(--text);
font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
line-height:1.5;background:
radial-gradient(1200px 560px at 50% -160px,rgba(0,224,164,.12),transparent 70%),
radial-gradient(900px 420px at 92% 3%,rgba(255,211,77,.07),transparent 60%),
radial-gradient(900px 520px at 5% 16%,rgba(91,141,239,.06),transparent 60%),
linear-gradient(180deg,#0c1018,#090d15 78%);background-attachment:fixed}
body::before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;opacity:.6;
background-image:repeating-linear-gradient(118deg,rgba(255,255,255,.016) 0 1px,transparent 1px 30px)}
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
h2.col{cursor:pointer;user-select:none}
h2.col::after{content:"⌄";margin-left:auto;color:var(--muted);font-size:21px;line-height:1;
transition:transform .25s}
h2.col.open::after{transform:rotate(180deg)}
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
.sticky{position:sticky;top:60px}
.tab{padding-top:4px}
.tabnav{position:sticky;top:0;z-index:40;display:flex;gap:5px;align-items:center;flex-wrap:wrap;
padding:9px 0;margin:0 0 8px;background:rgba(12,16,24,.92);
-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.tabnav button{background:transparent;border:1px solid transparent;color:var(--muted);font-size:13.5px;
font-weight:700;padding:8px 13px;border-radius:10px;cursor:pointer;transition:all .15s;font-family:inherit}
.tabnav button:hover{color:var(--text)}
.tabnav button.on{background:var(--panel);color:var(--green);border-color:var(--line)}
.tabhome{text-decoration:none;color:var(--text);font-size:15px;padding:7px 9px;border-radius:10px;
border:1px solid var(--line);margin-right:3px}
.tabhome:hover{border-color:var(--green)}
@media(max-width:560px){.tabnav{gap:3px}.tabnav button{padding:7px 9px;font-size:12.5px}
.tabhome{padding:6px 8px}}
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
.barwrap{display:flex;align-items:center;gap:8px;width:100%}
.bar{position:relative;flex:1;min-width:30px;background:var(--line);border-radius:6px;height:18px;overflow:hidden}
.fill{position:absolute;left:0;top:0;height:100%;border-radius:6px;
transition:width .6s cubic-bezier(.22,1,.36,1)}
.barlbl{flex:0 0 auto;min-width:46px;text-align:right;font-size:12px;font-weight:700;
color:var(--text);white-space:nowrap}
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
.g-row .grec{flex:0 0 auto;color:#8b95ab;font-size:12px;white-space:nowrap}
.g-row .qbadge{flex:0 0 50px;text-align:right;font-weight:800;font-size:12px}
.qbadge.q-in{color:#00e0a4}.qbadge.q-out{color:#ff6b6b}.qbadge.q-hunt{color:#cbd3e1}
.pcard{background:#161d2b;border:1px solid #243049;border-radius:12px;padding:12px 14px;margin:0 0 10px}
.pcard.plk{opacity:.6}
.prow{display:flex;align-items:center;gap:8px;font-size:14.5px;font-weight:700}
.pteam{flex:1;display:flex;align-items:center;gap:6px;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pteam.ar{justify-content:flex-end;text-align:right}
.psc{display:inline-flex;align-items:center;gap:4px;flex:0 0 auto}
.psc b{min-width:16px;text-align:center;font-size:16px}.psc i{color:#8b95ab;font-style:normal}
.pmeta{color:#8b95ab;font-size:12px;margin-top:7px}
.pdone{color:#00e0a4;font-weight:700}
.psugg{margin-top:9px;font-size:12.5px;color:#ffd34d}
.pchip{background:#1c2536;border:1px solid #243049;color:#cbd3e1;border-radius:999px;
padding:5px 11px;font-size:12.5px;cursor:pointer;font-weight:600;margin:0 5px 5px 0}
.pchip:hover{border-color:#00e0a4;color:#fff}
.pboard{display:flex;align-items:center;justify-content:center;gap:14px;background:#161d2b;
border:1px solid #243049;border-radius:14px;padding:14px;margin-bottom:8px}
.pb{text-align:center;min-width:64px}
.pbn{font-size:30px;font-weight:800;font-family:'Outfit',sans-serif}
.pbl{color:#8b95ab;font-size:12.5px}
.pbvs{font-weight:700;font-size:13.5px;text-align:center;flex:1;max-width:170px}
.pstats{color:#8b95ab;font-size:13px;text-align:center;margin-bottom:18px}
.psec{font-size:15px;margin:18px 0 10px}
.presrow{display:flex;justify-content:space-between;font-size:13.5px;margin-top:7px}
.presrow.mac{color:#9fb0c9}
.ppts{font-weight:800}.ppts.g{color:#00e0a4}.ppts.a{color:#ffd34d}.ppts.r{color:#8b95ab}
.pbeat{color:#00e0a4;font-weight:700;font-size:12.5px;text-align:center;margin-top:6px}
.pbeat.lose{color:#8b95ab}
.pauth{display:flex;align-items:center;justify-content:space-between;gap:10px;background:#161d2b;
border:1px solid #243049;border-radius:12px;padding:9px 14px;margin-bottom:12px;font-size:13.5px}
.pbtn{background:#243049;color:#eef1f7;border:0;border-radius:999px;padding:6px 14px;font-size:13px;font-weight:700;cursor:pointer}
.pbtn:hover{background:#2c3a57}.pbtn.pg{background:#00e0a4;color:#062018}.pbtn.pg:hover{filter:brightness(1.08)}
.plink{background:none;border:0;color:#8b95ab;font-size:12px;cursor:pointer;text-decoration:underline;padding:0;margin-left:4px}
.plink:hover{color:#00e0a4}
.lbclick{cursor:pointer}.lbclick:hover{background:rgba(0,224,164,.06)}
.pmodbg{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:200;display:flex;align-items:center;justify-content:center;padding:18px}
.pmod{background:#0f1726;border:1px solid #243049;border-radius:16px;max-width:520px;width:100%;max-height:82vh;overflow:auto;padding:16px 18px}
.tpopen{display:block;width:100%;margin-top:12px;background:rgba(0,224,164,.13);color:#00e0a4;border:1px solid rgba(0,224,164,.4);border-radius:10px;padding:9px;font-weight:700;cursor:pointer;font-size:13px}
.tpopen:hover{background:rgba(0,224,164,.22)}
.g-row.gclick{cursor:pointer}.g-row.gclick:hover{background:rgba(255,255,255,.04)}
.tpg{position:relative;background:#0f1726;border:1px solid #243049;border-radius:16px;max-width:660px;width:100%;max-height:88vh;overflow:auto;padding:22px 22px 18px}
.tpg .tpx{position:absolute;top:12px;right:12px}
.tphd{display:flex;align-items:center;gap:14px;margin:0 30px 2px 0}
.tpnm{font-size:24px;font-weight:800;line-height:1.15}
.tpform{margin-top:7px}
.tpkpis{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:16px 0 4px}
.tpk{background:#161d2b;border:1px solid #243049;border-radius:12px;padding:12px 8px;text-align:center}
.tpkv{font-size:23px;font-weight:800;color:#ffd34d}
.tpkl{font-size:11px;color:#8b95ab;margin-top:3px}
.tpsec{margin:20px 0 9px;font-size:12px;letter-spacing:.5px;text-transform:uppercase;color:#8b95ab}
.tpgr{display:flex;align-items:center;gap:9px;padding:6px 8px;border-radius:8px;font-size:14px}
.tpgr.me{background:rgba(255,211,77,.13);font-weight:700}
.tpgr .pos{width:16px;color:#8b95ab;text-align:center}.tpgr .nm{flex:1}
.tpgr .grec{color:#8b95ab;font-size:12px}.tpgr .adv{color:#00e0a4;font-weight:700;font-size:13px}
.tpfin{margin-top:8px}
.tpm{display:flex;align-items:center;gap:10px;padding:7px 4px;border-bottom:1px solid #18202f;font-size:14px}
.tpmko{color:#8b95ab;font-size:12px;white-space:nowrap;min-width:104px}
.tpmo{flex:1}.tpmr{text-align:right;white-space:nowrap}
.tpgbr{display:flex;justify-content:space-between;padding:5px 8px;font-size:14px}
.tpshare{margin-top:16px;text-align:center}
.tpfacts{font-size:13.5px;line-height:1.9}.tpfacts b{color:#eef1f7}
.pmodh{display:flex;align-items:center;justify-content:space-between;font-size:17px;margin-bottom:2px}
.presrow2{display:flex;justify-content:space-between;gap:10px;font-size:13.5px;padding:8px 0;border-bottom:1px solid #1c2536}
.presrow2:last-child{border:none}
.pcrowd{margin-top:8px;font-size:12.5px;color:#9fb0c9}
.lbx{background:#161d2b;border:1px solid #243049;border-radius:12px;padding:4px 14px;margin-bottom:6px}
.lbrow{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid #1c2536;font-size:14px}
.lbrow:last-child{border:none}
.lbrank{width:22px;color:#8b95ab;font-weight:700;text-align:right}
.lbname{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lbpts{font-weight:800}
.lbmac{color:#ffd34d}.lbmac .lbrank{color:#ffd34d}.lbme{color:#00e0a4}.lbme .lbpts{color:#00e0a4}
.lbrow.lbrival{color:#ff9d6b}.lbrow.lbrival .lbpts{color:#ff9d6b}
.pbadges{display:flex;flex-wrap:wrap;justify-content:center;gap:7px;margin:-6px 0 16px}
.pbadge{background:rgba(255,211,77,.12);border:1px solid rgba(255,211,77,.35);color:#ffd34d;border-radius:999px;padding:4px 11px;font-size:12px;font-weight:700;cursor:default}
.prival{background:rgba(255,107,107,.1);border:1px solid rgba(255,107,107,.3);border-radius:10px;padding:8px 12px;margin-bottom:8px;font-size:13.5px;text-align:center}
.pbrlock{background:#161d2b;border:1px solid #243049;border-radius:12px;padding:18px;text-align:center;font-size:14px;color:#9fb0c9}
.pcard.pmine{border-color:rgba(0,224,164,.5)}
.ptoast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%) translateY(18px);
background:#00e0a4;color:#062018;font-weight:800;padding:10px 20px;border-radius:999px;
font-size:14px;opacity:0;transition:all .25s;z-index:100;pointer-events:none}
.ptoast.show{opacity:1;transform:translateX(-50%) translateY(0)}
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
.heat{display:grid;gap:2px;max-width:330px}
.hc{aspect-ratio:1;display:flex;align-items:center;justify-content:center;font-size:9px;
border-radius:3px;color:#06231b;background:var(--panel2)}
.hc.hh{background:transparent!important;color:#8b95ab;font-size:10px}
.labgrid{display:grid;grid-template-columns:auto 1fr;gap:22px;margin-top:16px;align-items:start}
@media(max-width:680px){.labgrid{grid-template-columns:1fr}}
.fd{display:inline-block;width:19px;height:19px;line-height:19px;text-align:center;
border-radius:5px;font-size:11px;font-weight:700;margin-right:2px}
.fd.W{background:#00e0a4;color:#06231b}.fd.D{background:#5d6a85;color:#eef1f7}
.fd.L{background:#ff6b6b;color:#fff}
.rollbox{background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);
border-radius:14px;padding:18px;margin-bottom:14px}
.champ{font-size:22px;font-weight:800;font-family:'Outfit',sans-serif}
.g-row.q .nm{color:#00e0a4}.g-row.q3 .nm{color:#ffd34d}
h1,h2,.hero h1,.kpi .big{font-family:'Outfit',-apple-system,Segoe UI,sans-serif}
/* ---- bracket tree (two halves converging on the final) ---- */
.tree{display:flex;gap:0;overflow-x:auto;padding:8px 0 16px}
.tcol{display:flex;flex-direction:column;justify-content:space-around;flex:1 1 0;
min-width:96px;padding:0 4px}
.tcol.tc-final{justify-content:center;flex:1.1 1 0}
.tch{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;
text-align:center;margin-bottom:6px;font-weight:700}
.tmatch{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:9px;
padding:5px 7px;margin:6px 0;font-size:10.5px}
.tc-final .tmatch{border-color:rgba(255,211,77,.55);
box-shadow:0 0 18px rgba(255,211,77,.12)}
.tt{display:flex;align-items:center;gap:4px;padding:2px 0;color:#7c879c}
.tt.win{color:var(--text);font-weight:700}
.tn{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tv{font-size:9.5px;color:var(--muted);font-weight:700}
.tt.win .tv{color:var(--green)}
.tpe{font-size:8.5px;color:var(--gold);text-align:right;margin-top:1px}
.tree .flag.sm{width:14px;height:10px}
.tcol.l .tmatch::after{content:"";position:absolute;left:100%;top:50%;width:8px;height:2px;background:var(--line)}
.tcol.r .tmatch::after{content:"";position:absolute;right:100%;top:50%;width:8px;height:2px;background:var(--line)}
.tcol.tc-final .tmatch::after{display:none}
/* ---- results so far (predicted vs actual) ---- */
.rev td{vertical-align:middle}
.badge2{font-size:11px;padding:2px 8px;border-radius:999px;font-weight:700;white-space:nowrap}
.badge2.y{background:rgba(0,224,164,.16);color:var(--green)}
.badge2.n{background:rgba(255,107,107,.15);color:var(--red)}
/* ---- betting markets grid ---- */
.mkts{display:grid;grid-template-columns:repeat(auto-fill,minmax(104px,1fr));gap:8px;margin-top:2px}
.mk{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:7px 10px}
.mk span{color:var(--muted);font-size:10.5px;display:flex;align-items:center;gap:4px}
.mk b{display:block;font-size:15px;color:var(--text);margin-top:2px}
/* ---- recent head-to-head meetings + roll group results ---- */
.h2hl{margin-top:8px}.h2hl .r{display:flex;align-items:center;gap:6px;padding:2px 0;
font-size:12.5px;color:var(--muted)}.h2hl .r b{color:var(--text)}
.gres{margin-top:8px;font-size:11.5px;color:var(--muted)}
.gres summary{cursor:pointer;color:var(--green);font-size:11.5px}
.gres .gr{display:flex;align-items:center;gap:5px;padding:2px 0;margin-top:2px}
.gres .gr b{color:var(--text)}
/* ---- goal timelines + 'anatomy of a goal' minute bars ---- */
.goalsline{font-size:11.5px;color:var(--muted);margin-top:4px;white-space:normal}
.goalsline b{color:var(--text)}
.gminutes{display:flex;gap:7px;align-items:flex-end;height:140px;margin-top:6px;max-width:480px}
.gminute{flex:1;display:flex;flex-direction:column;align-items:center;gap:5px;height:100%}
.gmbar{flex:1;width:100%;max-width:48px;background:var(--line);border-radius:6px 6px 0 0;
display:flex;align-items:flex-end;overflow:hidden}
.gmbar>div{width:100%;border-radius:6px 6px 0 0;transition:height .7s cubic-bezier(.22,1,.36,1)}
.gmlbl{font-size:10px;color:var(--muted)}.gmval{font-size:10px;color:var(--muted);font-weight:700}
/* ---- today's matches ---- */
.todaycards{display:grid;grid-template-columns:repeat(auto-fill,minmax(216px,1fr));gap:10px;margin-top:4px}
.todaycard{background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);
border-radius:14px;padding:12px 14px}
.tdtime{font-size:12px;color:var(--green);font-weight:700;margin-bottom:6px}
.tdteams{font-weight:700;display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:14px}
.tdvs{color:var(--muted);font-weight:400;font-size:12px}
.tdpred,.tdres{font-size:12.5px;color:var(--muted);margin-top:8px}
.tdres b,.tdpred b{color:var(--text)}
/* ---- "what if?" editor ---- */
.wifgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.wifm{display:flex;align-items:center;gap:6px;font-size:12.5px;margin:5px 0}
.wifm.wifp{color:var(--muted)}.wifm.wifp b{color:var(--text)}
.wifteam{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.wifteam.ar{text-align:right}
.wifsc{display:inline-flex;align-items:center;gap:3px;flex:0 0 auto}
.wifsc i{font-style:normal;color:var(--muted)}
.wifsc>b{min-width:11px;text-align:center;color:var(--text)}
.wifb{background:var(--panel);color:var(--muted);border:1px solid var(--line);border-radius:6px;
width:20px;height:20px;line-height:18px;text-align:center;font-size:13px;font-weight:700;cursor:pointer;padding:0}
.wifb:hover{border-color:var(--green);color:var(--green)}
.wift{width:100%;margin-top:8px;border-top:1px solid var(--line)}
.wift td{padding:3px 4px;font-size:12px;border:none}
.wift .pos{width:14px;color:#5d6a85}
.wift tr.q td:nth-child(2){color:#00e0a4;font-weight:700}
.wift tr.q3 td:nth-child(2){color:#ffd34d;font-weight:700}
.bywchamp{font-size:18px;font-weight:800;font-family:'Outfit',sans-serif;margin:12px 0 8px}
.tt[data-m]{cursor:pointer}
.tt[data-m]:hover{background:rgba(0,224,164,.12);border-radius:5px}
.homebtn{position:fixed;top:14px;left:14px;z-index:50;display:inline-flex;align-items:center;
gap:6px;background:rgba(22,29,43,.85);-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);
border:1px solid var(--line);color:var(--text);text-decoration:none;font-size:13px;font-weight:600;
padding:7px 13px;border-radius:999px;transition:all .2s}
.homebtn:hover{border-color:rgba(0,224,164,.5);color:var(--green);transform:translateY(-1px)}
@media(max-width:560px){.homebtn{top:8px;left:8px;font-size:12px;padding:6px 10px}}
/* ---- mobile / responsive ---- */
@media(max-width:680px){
  html{-webkit-text-size-adjust:100%}
  body{background-attachment:scroll}
  .wrap{padding:0 12px 56px}
  .hero{padding:26px 0 16px}
  h2{font-size:18px;margin:28px 0 12px}
  /* every wide table becomes its own horizontally-scrollable block instead of
     being crushed; cells stay on one line and the table swipes sideways */
  table{display:block;overflow-x:auto;-webkit-overflow-scrolling:touch;
    white-space:nowrap;font-size:12.5px}
  th,td{padding:7px 9px}
  .cards{gap:10px}.kpi{padding:15px}.kpi .big{font-size:23px}
  .grid{grid-template-columns:1fr;gap:12px}
  .heat{max-width:100%}
  .tcol{min-width:80px;padding:0 3px}
  .tch{font-size:8px}.tmatch{font-size:9.5px;padding:4px 5px}
  .prow .lb{width:84px}
  .mkts{grid-template-columns:repeat(auto-fill,minmax(92px,1fr))}
  .scorelines{gap:6px}
}
@media(max-width:430px){
  .hero h1{font-size:25px}
  .cards{grid-template-columns:1fr 1fr}
  .kpi .big{font-size:20px}
  th,td{padding:6px 7px;font-size:12px}
  .chip{font-size:12px;padding:5px 10px}
}
"""

_JS = r"""
const D=DATA, T=D.teams, byName=Object.fromEntries(T.map(t=>[t.team,t]));
const STAGES=[["p_ko","Last 32"],["p_r16","Round of 16"],["p_qf","Quarter-finals"],
["p_sf","Semi-finals"],["p_final","Final"],["p_champion","Champion"]];
const pct=x=>(x*100).toFixed(1)+"%", pc0=x=>(x*100).toFixed(0)+"%";
const pctv=x=>{const v=x*100;if(v<=0)return"0%";if(v<0.01)return"<0.01%";
  if(v<0.1)return v.toFixed(2)+"%";return v.toFixed(1)+"%";};
let sortKey="p_champion",sortDir=-1,selected=D.teams[0].team,query="";
const qteam=new URLSearchParams(location.search).get("team");  // ?team= deep link
if(qteam&&byName[qteam])selected=qteam;

function flag(team,cls){const t=byName[team]||{};const c=t.code!==undefined?t.code:(D.favorite.team===team?D.favorite.code:"");
  if(c) return `<img class="flag ${cls||''}" src="https://flagcdn.com/w40/${c}.png" alt="" loading="lazy">`;
  return `<span>${t.flag||"🏳️"}</span>`;}
function bar(p,color,label){const w=Math.min(Math.max(p*100,0),100);
  return `<div class="barwrap"><div class="bar"><div class="fill" style="width:${w}%;background:${color}"></div></div>
  <span class="barlbl">${label==null?pctv(p):label}</span></div>`;}

function renderRanking(){
  const cols=[["p_champion","Champion"],["p_final","Final"],["p_sf","Semis"],
  ["p_qf","QF"],["p_r16","R16"],["p_ko","R32"]];
  let rows=[...T].sort((a,b)=>sortDir*(a[sortKey]-b[sortKey]));
  if(query) rows=rows.filter(t=>t.team.toLowerCase().includes(query));
  let h=`<table><thead><tr><th>#</th><th>Team</th><th>Gr</th><th>Elo</th>`;
  cols.forEach(c=>h+=`<th data-k="${c[0]}">${c[1]}${sortKey===c[0]?(sortDir<0?" ▾":" ▴"):""}</th>`);
  h+=`</tr></thead><tbody>`;
  rows.forEach((t,i)=>{h+=`<tr class="row${t.team===selected?' sel':''}" data-t="${t.team}">
  <td>${i+1}</td><td><span class="row-team">${flag(t.team)} ${t.team}</span></td>
  <td>${t.group}</td><td>${t.elo}</td>
  <td style="min-width:170px">${bar(t.p_champion,"#00e0a4")}</td>
  <td>${pctv(t.p_final)}</td><td>${pctv(t.p_sf)}</td><td>${pctv(t.p_qf)}</td>
  <td>${pctv(t.p_r16)}</td><td>${pctv(t.p_ko)}</td></tr>`;});
  h+=`</tbody></table>`;
  const c=document.getElementById("ranking");c.innerHTML=h;
  c.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{const k=th.dataset.k;
    if(k===sortKey)sortDir*=-1;else{sortKey=k;sortDir=-1;}renderRanking();});
  c.querySelectorAll("tr.row").forEach(tr=>tr.onclick=()=>{selected=tr.dataset.t;
    history.replaceState(null,"","?team="+encodeURIComponent(selected));
    renderRanking();renderDetail();renderGroups();});
}

function renderDetail(){
  const t=byName[selected];
  let h=`<div style="display:flex;align-items:center;gap:12px">${flag(t.team,"lg")}
  <div><div style="font-size:21px;font-weight:800">${t.team}</div>
  <div class="tag">Group ${t.group} · Elo ${t.elo}${t.fifa_rank?' · FIFA #'+t.fifa_rank:''} · ${t.exp_points.toFixed(1)} exp. pts</div></div></div>
  <button class="tpopen" id="tpopen">🔍 Open ${t.team}'s full page ↗</button>
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
  document.getElementById("tpopen").onclick=()=>teamPage(selected);
}

function teamRank(name){return [...D.teams].sort((a,b)=>b.p_champion-a.p_champion).findIndex(t=>t.team===name)+1;}
function teamPage(name){
  const t=byName[name];if(!t)return;
  history.replaceState(null,"","?team="+encodeURIComponent(name));
  const pr={};(D.played_review||[]).forEach(m=>pr[m.home+"|"+m.away]=m);
  let h=`<div class="tpg"><button class="pbtn tpx" id="tpx">✕</button>`
    +`<div class="tphd">${flag(t.team,"lg")}<div><div class="tpnm">${t.team}</div>`
    +`<div class="tag">Group ${t.group} · Elo ${t.elo}${t.fifa_rank?' · FIFA #'+t.fifa_rank:''} · #${teamRank(name)} favourite</div>`
    +`<div class="tpform">${fdot(t.form)}</div></div></div>`
    +`<div class="tpkpis">`
    +`<div class="tpk"><div class="tpkv">${pc0(t.p_champion)}</div><div class="tpkl">🏆 Champion</div></div>`
    +`<div class="tpk"><div class="tpkv">${pc0(t.p_final)}</div><div class="tpkl">🥈 Reach final</div></div>`
    +`<div class="tpk"><div class="tpkv">${pc0(t.p_ko)}</div><div class="tpkl">🎟️ Advance group</div></div></div>`
    +`<h4 class="tpsec">🏆 Path to the final</h4>`;
  STAGES.forEach(s=>{h+=`<div class="prow"><span class="lb">${s[1]}</span>${bar(t[s[0]],"#ffd34d")}</div>`;});
  if(t.opponents&&Object.keys(t.opponents).length)
    for(const[rnd,info] of Object.entries(t.opponents)){
      const opps=info.opponents.map(o=>`${flag(o.team,"sm")} ${o.team} <span class="note">${pc0(o.p_cond)}</span>`).join(" · ");
      h+=`<div class="note" style="margin-top:6px"><b>${rnd}</b> <span class="tag">reaches ${pc0(info.p_reach)}</span><br>${opps}</div>`;}
  const grp=[...D.teams].filter(x=>x.group===t.group).sort((a,b)=>(b.pts-a.pts)||(b.gd-a.gd)||(b.gf-a.gf)||(b.exp_points-a.exp_points));
  h+=`<h4 class="tpsec">🎟️ Group ${t.group}</h4>`;
  grp.forEach((x,i)=>{const me=x.team===name,rec=x.played?`P${x.played} · ${x.pts}pt${x.pts!==1?'s':''}`:'—';
    h+=`<div class="tpgr${me?' me':''}"><span class="pos">${i+1}</span><span class="nm">${flag(x.team,'sm')} ${x.team}</span><span class="grec">${rec}</span><span class="adv">${pc0(x.p_ko)}</span></div>`;});
  h+=`<div class="note tpfin">Expected finish — 1st ${pc0(t.p_1st)} · 2nd ${pc0(t.p_2nd)} · 3rd ${pc0(t.p_3rd)} · 4th ${pc0(t.p_4th)}</div>`;
  const mkey=m=>m.home+"|"+m.away,seen={},list=[];   // played + upcoming, chronological
  (D.played_review||[]).filter(m=>m.home===name||m.away===name).forEach(m=>{seen[mkey(m)]=1;list.push(m);});
  (D.matches||[]).filter(m=>(m.home===name||m.away===name)&&!seen[mkey(m)]).forEach(m=>list.push(m));
  list.sort((a,b)=>String(a.date||"").localeCompare(String(b.date||"")));
  if(list.length){h+=`<h4 class="tpsec">📅 Matches</h4>`;
    list.forEach(m=>{const home=m.home===name,opp=home?m.away:m.home,res=pr[mkey(m)];let right;
      if(res){const sc=res.hs+"-"+res["as"],ml=String(res.ml_score).split("-").map(Number),
        aw=Math.sign(res.hs-res["as"]),mw=Math.sign(ml[0]-ml[1]);
        right=`<b>${sc}</b> ${res.ml_score===sc?'🎯':aw===mw?'✅':'❌'}`;}
      else{const pw=home?m.p_home:m.p_away,pl=home?m.p_away:m.p_home;
        right=`<span class="note">W ${pc0(pw)} · D ${pc0(m.p_draw)} · L ${pc0(pl)}</span>`;}
      h+=`<div class="tpm"><span class="tpmko">${koLabel(m.home,m.away,m.date)}</span><span class="tpmo">${home?'vs':'@'} ${flag(opp,'sm')} ${opp}</span><span class="tpmr">${right}</span></div>`;});}
  if(t.recent&&t.recent.length){h+=`<h4 class="tpsec">📈 Recent form <span class="tag">last ${t.recent.length}</span></h4>`;
    t.recent.forEach(r=>{const rs=r.gf>r.ga?'W':r.gf<r.ga?'L':'D';
      h+=`<div class="tpm"><span class="tpmko">${r.date}</span><span class="tpmo">${r.home?'vs':'@'} ${flag(r.opp,'sm')} ${r.opp}</span><span class="tpmr"><b>${r.gf}-${r.ga}</b> <span class="fd ${rs}">${rs}</span></span></div>`;});}
  const gs=(D.golden_boot||[]).filter(g=>g.team===name).slice(0,5);
  if(gs.length){h+=`<h4 class="tpsec">👟 Golden Boot contenders</h4>`;
    gs.forEach(g=>{h+=`<div class="tpgbr"><span>${flag(g.team,'sm')} ${g.scorer}</span><span class="note">${g.proj.toFixed(1)} proj goals</span></div>`;});}
  if(t.scorers&&t.scorers.length){h+=`<h4 class="tpsec">⚽ All-time top scorers</h4>`;
    t.scorers.forEach((s,i)=>{h+=`<div class="tpgbr"><span>${i+1}. ${s.scorer}</span><span class="note">${s.goals} goals</span></div>`;});}
  const c=t.card;
  if(c&&c.games){h+=`<h4 class="tpsec">📊 All-time record</h4><div class="tpfacts">`
    +`<div><b>${c.w}W · ${c.d}D · ${c.l}L</b> <span class="note">(${c.win_pct}% win over ${c.games} games)</span></div>`
    +`<div>🏆 Biggest win — <b>${c.biggest_win}</b></div>`
    +`<div>💀 Worst loss — <b>${c.worst_loss}</b></div>`
    +`<div>🛡️ Longest unbeaten run — <b>${c.longest_unbeaten}</b> games</div>`
    +(c.peak_elo?`<div>📈 Peak Elo — <b>${c.peak_elo[0]}</b> in ${c.peak_elo[1]}</div>`:'')
    +(c.rival?`<div>😤 Most-played rival — <b>${c.rival}</b> (${c.rival_record})</div>`:'')
    +`</div>`;}
  h+=`<div class="note tpshare">🔗 Shareable — this link opens straight to ${t.team}.</div></div>`;
  const old=document.getElementById("tpbg");if(old)old.remove();
  document.body.insertAdjacentHTML("beforeend",`<div class="pmodbg" id="tpbg">${h}</div>`);
  const bg=document.getElementById("tpbg");
  const tpClose=()=>{bg.remove();document.removeEventListener("keydown",tpEsc);   // pure popup: drop ?team= on close, no reload
    const u=new URLSearchParams(location.search);u.delete("team");u.set("tab","teams");
    history.replaceState(null,"","?"+u);};
  function tpEsc(e){if(e.key==="Escape")tpClose();}
  document.addEventListener("keydown",tpEsc);
  bg.onclick=e=>{if(e.target===bg)tpClose();};
  document.getElementById("tpx").onclick=tpClose;
}

function renderGroups(){
  const groups={};T.forEach(t=>(groups[t.group]=groups[t.group]||[]).push(t));
  const ord=(a,b)=>(b.pts-a.pts)||(b.gd-a.gd)||(b.gf-a.gf)||(b.exp_points-a.exp_points);
  let h="";Object.keys(groups).sort().forEach(L=>{
    const g=groups[L].slice().sort(ord);
    h+=`<div class="group"><h3>GROUP ${L}</h3>`;
    g.forEach((t,i)=>{const sel=t.team===selected?'style="color:#ffd34d;font-weight:700"':'';
      const adv=Math.round(t.p_ko*100);
      const cls=adv>=90?"q-in":adv<=10?"q-out":"q-hunt";
      const rec=t.played?`P${t.played} · ${t.pts}pt${t.pts!==1?"s":""}`:"—";
      h+=`<div class="g-row gclick" data-t="${t.team}"><span class="pos">${i+1}</span>
      <span class="nm" ${sel}>${flag(t.team,"sm")} ${t.team}</span>
      <span class="grec">${rec}</span>
      <span class="qbadge ${cls}" title="model's chance to reach the knockouts">${adv}%</span></div>`;});
    h+=`</div>`;});
  const gc=document.getElementById("groups");gc.innerHTML=h;
  gc.querySelectorAll(".g-row").forEach(r=>r.onclick=()=>teamPage(r.dataset.t));
}

function koLabel(home,away,date){const k=D.kickoffs&&D.kickoffs[[home,away].sort().join("|")];
  return k?k.label:(date?date.slice(5):"");}
function renderMatches(){
  const sel=document.getElementById("mfilter").value;
  const ms=D.matches.filter(m=>sel==="all"||m.group===sel);
  let h=`<table class="mtable"><thead><tr><th>Kickoff (WEST)</th><th>Gr</th><th>Match</th>
  <th>Top scorelines</th><th>W/D/L</th></tr></thead><tbody>`;
  ms.forEach(m=>{const top=m.top.map((s,i)=>`<b>${s.score}</b> ${pc0(s.p)}`).join(" · ");
    h+=`<tr><td style="white-space:nowrap">${koLabel(m.home,m.away,m.date)}</td><td>${m.group}</td>
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
  let s=0;G.forEach(r=>r.forEach(v=>s+=v));G=G.map(r=>r.map(v=>v/s));
  let ph=0,pd=0,pa=0,o15=0,o25=0,o35=0,btts=0,csa=0,csb=0,flat=[];
  for(let i=0;i<=N;i++)for(let j=0;j<=N;j++){const v=G[i][j],tot=i+j;
    if(i>j)ph+=v;else if(i===j)pd+=v;else pa+=v;
    if(tot>=2)o15+=v;if(tot>=3)o25+=v;if(tot>=4)o35+=v;
    if(i>=1&&j>=1)btts+=v;if(j===0)csa+=v;if(i===0)csb+=v;
    flat.push([i,j,v]);}
  flat.sort((x,y)=>y[2]-x[2]);
  return {la,lb,ph,pd,pa,o15,o25,o35,btts,csa,csb,G,
          top:flat.slice(0,5).map(([i,j,v])=>({s:i+"-"+j,p:v}))};}
function fdot(s){return (s||"").split("").map(c=>`<span class="fd ${c}">${c}</span>`).join("")||"<span class='note'>n/a</span>";}
function renderLab(){
  const a=document.getElementById("labA").value,b=document.getElementById("labB").value;
  if(a===b){document.getElementById("lab-out").innerHTML="<p class='note' style='text-align:center'>Pick two different teams.</p>";return;}
  const r=h2h(a,b),W=r.ph*100,Dr=r.pd*100,L=r.pa*100,M=5;
  let hm=`<div class="heat" style="grid-template-columns:auto repeat(${M+1},1fr)"><div class="hc hh"></div>`;
  for(let j=0;j<=M;j++)hm+=`<div class="hc hh">${j}</div>`;
  for(let i=0;i<=M;i++){hm+=`<div class="hc hh">${i}</div>`;
    for(let j=0;j<=M;j++){const v=r.G[i][j],al=Math.min(v/0.12,1);
      hm+=`<div class="hc" style="background:rgba(0,224,164,${al.toFixed(2)})" title="${a} ${i}-${j} ${b}: ${pc0(v)} chance">${v>=0.045?Math.round(v*100):""}</div>`;}}
  hm+=`</div>`;
  const hh=D.h2h[[a,b].sort().join("|")];
  let hist="No previous meetings in the data.",recent="";
  if(hh){const isa=hh.t1===a,aw=isa?hh.w1:hh.w2,bw=isa?hh.w2:hh.w1;
    hist=`All-time: <b>${a}</b> ${aw}W · ${hh.d}D · ${bw}W <b>${b}</b> <span style="color:#8b95ab">(${hh.n} meetings)</span>`;
    if(hh.recent&&hh.recent.length)
      recent=`<div class="h2hl"><div class="note" style="margin-bottom:2px">Last meetings</div>`+
      hh.recent.map(m=>`<div class="r">${m.date.slice(0,4)} &nbsp;${flag(m.home,'sm')} `+
        `<b>${m.home}</b> <b style="color:#ffd34d">${m.hs}-${m["as"]}</b> <b>${m.away}</b> ${flag(m.away,'sm')}</div>`
      ).join("")+`</div>`;}
  const mk=(l,v)=>`<div class="mk"><span>${l}</span><b>${pc0(v)}</b></div>`;
  const tops=r.top.map(t=>`<b>${t.s}</b> ${pc0(t.p)}`).join(" · ");
  const ki=D.kickoffs&&D.kickoffs[[a,b].sort().join("|")];
  const koLine=ki?`<div class="note" style="text-align:center;margin-top:4px">🕒 World Cup kickoff: <b>${ki.label}</b> &nbsp;<span style="color:#5d6a85">WEST (UTC+1)</span></div>`:"";
  const markets=`<div class="mkts">
    ${mk("Over 1.5",r.o15)}${mk("Over 2.5",r.o25)}${mk("Over 3.5",r.o35)}${mk("Under 2.5",1-r.o25)}
    ${mk("Both teams score",r.btts)}${mk(flag(a,'sm')+" clean sheet",r.csa)}${mk(flag(b,'sm')+" clean sheet",r.csb)}
    ${mk(flag(a,'sm')+" win or draw",r.ph+r.pd)}${mk(flag(b,'sm')+" win or draw",r.pa+r.pd)}${mk("Anyone but a draw",r.ph+r.pa)}</div>`;
  document.getElementById("lab-out").innerHTML=`
  <div class="wdl"><div style="width:${W}%;background:#00e0a4">${W.toFixed(0)}%</div>
  <div style="width:${Dr}%;background:#8b95ab">${Dr.toFixed(0)}%</div>
  <div style="width:${L}%;background:#ffd34d">${L.toFixed(0)}%</div></div>
  <div class="note" style="text-align:center">${flag(a,'sm')} win · draw · ${flag(b,'sm')} win
   &nbsp;|&nbsp; xG <b>${r.la.toFixed(2)} – ${r.lb.toFixed(2)}</b></div>
  ${koLine}
  <div class="labgrid"><div>
    <div class="note" style="margin-bottom:6px">Chance of each exact score (%) — rows ${flag(a,'sm')} ${a} goals, cols ${flag(b,'sm')} ${b} goals</div>${hm}</div>
    <div><div class="note" style="margin-bottom:9px">⚽ Most likely scores: ${tops}</div>${markets}
    <div class="note" style="margin-top:14px">${hist}</div>${recent}
    <div class="note" style="margin-top:10px">Recent form &nbsp; ${flag(a,'sm')} ${fdot(byName[a].form)}
     &nbsp; ${flag(b,'sm')} ${fdot(byName[b].form)}</div></div></div>`;
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

/* ---- connected bracket tree (two halves converging on the final) ---- */
const BORDER={
 "Round of 32":[74,77,73,75,83,84,81,82,76,78,79,80,86,88,85,87],
 "Round of 16":[89,90,93,94,91,92,95,96],
 "Quarter-finals":[97,98,99,100],"Semi-finals":[101,102],"Final":[104]};
function tmatch(m){const wa=m.win&&m.win===m.a,wb=m.win&&m.win===m.b;
  const da=m.byw?` data-m="${m.m}" data-team="${m.a||''}"`:"",db=m.byw?` data-m="${m.m}" data-team="${m.b||''}"`:"";
  return `<div class="tmatch">
   <div class="tt ${wa?'win':''}"${da}>${flag(m.a,'sm')}<span class="tn">${m.a||'?'}</span><span class="tv">${m.va||''}</span></div>
   <div class="tt ${wb?'win':''}"${db}>${flag(m.b,'sm')}<span class="tn">${m.b||'?'}</span><span class="tv">${m.vb||''}</span></div>
   ${m.pe?'<div class="tpe">on penalties</div>':''}</div>`;}
function bracketTree(rounds){
  const ord=l=>BORDER[l]||[];
  const S=rounds.map(r=>({label:r.label,
    ms:r.ms.slice().sort((x,y)=>ord(r.label).indexOf(x.m)-ord(r.label).indexOf(y.m))}));
  const fi=S.length-1,mid=r=>Math.ceil(r.ms.length/2);
  let cols=[];
  for(let k=0;k<fi;k++)cols.push({label:S[k].label,side:'l',ms:S[k].ms.slice(0,mid(S[k]))});
  cols.push({label:S[fi].label,side:'c',ms:S[fi].ms});
  for(let k=fi-1;k>=0;k--)cols.push({label:S[k].label,side:'r',ms:S[k].ms.slice(mid(S[k]))});
  let h='<div class="tree">';
  cols.forEach(c=>{h+=`<div class="tcol ${c.side} ${c.side==='c'?'tc-final':''}"><div class="tch">${c.label}</div>`;
    c.ms.forEach(m=>h+=tmatch(m));h+='</div>';});
  return h+'</div>';
}
function renderBracket(){
  const rounds=D.bracket.map(rd=>({label:rd.label,
    ms:rd.matches.map(m=>{const wa=m.adv===m.home;
      return {a:m.home,b:m.away,win:m.adv,m:m.m,va:wa?pct(m.p):"",vb:wa?"":pct(m.p)};})}));
  document.getElementById("bracket").innerHTML=bracketTree(rounds);
}
function renderReview(){
  const rv=D.played_review,el=document.getElementById("review");if(!el)return;
  if(!rv||!rv.length){el.innerHTML="<p class='note'>No World Cup matches have been played yet.</p>";return;}
  const hits=rv.filter(m=>m.hit).length,oc=k=>k==="home"?"home win":k==="away"?"away win":"draw";
  let h=`<p class="note">Before a ball was kicked, the blind pre-tournament model picked the most likely <b>result</b> (home win / draw / away win) for every game — and got that right in <b>${hits}/${rv.length}</b> so far. The "most likely score" is just the single most probable exact scoreline (naturally low-odds, rarely spot-on), so judge the model on the result, not the score.</p>`;
  h+=`<table class="rev"><thead><tr><th>Date</th><th>Match</th><th>Actual result</th>
  <th title="Home win / Draw / Away win — the bold one is the model's pick">Model W / D / L</th>
  <th title="Single most probable exact score">Most likely score</th><th>Verdict</th></tr></thead><tbody>`;
  rv.forEach(m=>{
    const probs=[["home",m.p_home],["draw",m.p_draw],["away",m.p_away]];
    const wdl=probs.map(([k,v])=>{const pick=k===m.pred;
      const c=pick?(m.hit?"#00e0a4":"#ff6b6b"):"#8b95ab";
      return `<span style="color:${c};font-weight:${pick?700:400}">${pc0(v)}</span>`;}).join(" / ");
    const gl=(m.goals||[]).map(x=>`${x.minute!=null?x.minute+"'":""} ${x.scorer}${x.pen?" (p)":""}${x.og?" (og)":""}`).join(" · ");
    h+=`<tr><td style="white-space:nowrap">${koLabel(m.home,m.away,m.date)}</td>
    <td><span class="row-team">${flag(m.home,'sm')} ${m.home} <b>${m.hs}-${m["as"]}</b> ${m.away} ${flag(m.away,'sm')}</span>${gl?`<div class="goalsline">⚽ ${gl}</div>`:""}</td>
    <td>${oc(m.actual)}</td><td>${wdl}</td>
    <td>${m.ml_score===(m.hs+"-"+m["as"])
      ?`<span class="badge2 y" title="nailed the exact score!">🎯 ${m.ml_score} <span style="font-weight:400">${pc0(m.p_ml)}</span></span>`
      :`<span style="color:#8b95ab">${m.ml_score} <span style="font-size:11px">${pc0(m.p_ml)}</span></span>`}</td>
    <td>${m.hit?'<span class="badge2 y">✓ right result</span>':'<span class="badge2 n">✗ (gave it '+pc0(m.p_actual)+')</span>'}</td></tr>`;});
  document.getElementById("review").innerHTML=h+`</tbody></table>`;
}

function renderFifa(){
  const f=D.fifa,el=document.getElementById("fifa");if(!el)return;
  if(!f||!f.rows||!f.rows.length){el.style.display="none";return;}
  const chip=r=>`<div class="chip">${flag(r.team,'sm')} ${r.team}
    <span style="color:#8b95ab">model #${r.model_rank} · FIFA #${r.fifa_rank48}</span>
    <b style="color:${r.edge>0?'#00e0a4':'#ff6b6b'}">${r.edge>0?'+':''}${r.edge}</b></div>`;
  const believers=[...f.rows].sort((a,b)=>b.edge-a.edge).slice(0,5);
  const doubters=[...f.rows].sort((a,b)=>a.edge-b.edge).slice(0,5);
  let h=`<p class="note">The model's own strength ranking (by Elo) vs the official
   <b>FIFA ranking</b> (${f.date}), both re-ranked among the 48 finalists. They line up
   closely — rank correlation <b>${f.spearman.toFixed(2)}</b> — but the model has opinions:</p>
   <p class="note" style="margin-top:10px"><b>📈 Model's believers</b> — rates them higher than FIFA:</p>
   <div class="chips">${believers.map(chip).join("")}</div>
   <p class="note" style="margin-top:12px"><b>📉 Model's doubters</b> — rates them lower than FIFA:</p>
   <div class="chips">${doubters.map(chip).join("")}</div>`;
  h+=`<table style="margin-top:16px"><thead><tr><th>Team</th><th>Model rank</th>
   <th>FIFA rank</th><th>FIFA world #</th><th>FIFA pts</th><th>Δ</th></tr></thead><tbody>`;
  f.rows.forEach(r=>{const c=r.edge>0?'#00e0a4':(r.edge<0?'#ff6b6b':'#8b95ab');
    h+=`<tr><td><span class="row-team">${flag(r.team,'sm')} ${r.team}</span></td>
    <td>#${r.model_rank}</td><td>#${r.fifa_rank48}</td>
    <td style="color:#8b95ab">#${r.fifa_rank}</td>
    <td style="color:#8b95ab">${r.fifa_pts.toFixed(0)}</td>
    <td style="color:${c};font-weight:700">${r.edge>0?'+':''}${r.edge}</td></tr>`;});
  el.innerHTML=h+`</tbody></table>`;
}
/* ---- "what if?" editor: you set the remaining group results, tables update ---- */
let wifScores=null;
function wifInit(){
  wifScores={};
  const predBy={};(D.matches||[]).forEach(m=>predBy[[m.home,m.away].sort().join("|")]=m);
  (D.fixtures||[]).forEach((f,i)=>{if(f.played)return;
    const p=predBy[[f.home,f.away].sort().join("|")];let hs=1,as=1;
    if(p&&p.top&&p.top[0]){const s=p.top[0].score.split("-");hs=+s[0];as=+s[1];}
    wifScores[i]={hs,as};});  // default = the model's most likely score
}
function wifSc(i,f){return f.played?[f.hs,f["as"]]:[wifScores[i].hs,wifScores[i].as];}
function wifGroups(){
  const G=D.structure.groups,fx=D.fixtures,gres={};
  for(const L in G){const ts=G[L],st={};ts.forEach(t=>st[t]={t,pts:0,gf:0,ga:0});
    fx.forEach((f,i)=>{if(f.group!==L)return;const[hs,as]=wifSc(i,f);
      const H=st[f.home],A=st[f.away];H.gf+=hs;H.ga+=as;A.gf+=as;A.ga+=hs;
      if(hs>as)H.pts+=3;else if(as>hs)A.pts+=3;else{H.pts++;A.pts++;}});
    gres[L]={order:ts.slice().sort((x,y)=>st[y].pts-st[x].pts||(st[y].gf-st[y].ga)-(st[x].gf-st[x].ga)||st[y].gf-st[x].gf||(x<y?-1:1)),st};}
  const thirds=Object.keys(gres).map(L=>{const s=gres[L].st[gres[L].order[2]];
    return{L,pts:s.pts,gd:s.gf-s.ga,gf:s.gf};});
  thirds.sort((a,b)=>b.pts-a.pts||b.gd-a.gd||b.gf-a.gf||(a.L<b.L?-1:1));
  return{gres,qual3:new Set(thirds.slice(0,8).map(t=>t.L))};
}
function wifRender(){
  const el=document.getElementById("whatif");if(!el)return;if(!wifScores)wifInit();
  const {gres,qual3}=wifGroups(),G=D.structure.groups,fx=D.fixtures;
  const sn=t=>t.length>14?t.slice(0,13)+"…":t;
  const sb=(i,s,d,l)=>`<button class="wifb" data-i="${i}" data-s="${s}" data-d="${d}">${l}</button>`;
  let h=`<div class="wifgrid">`;
  for(const L in G){h+=`<div class="group"><h3>GROUP ${L}</h3>`;
    fx.forEach((f,i)=>{if(f.group!==L)return;const[hs,as]=wifSc(i,f);
      if(f.played)h+=`<div class="wifm wifp">${flag(f.home,'sm')} <span class="wifteam">${sn(f.home)}</span> <b>${hs}-${as}</b> <span class="wifteam ar">${sn(f.away)}</span> ${flag(f.away,'sm')}</div>`;
      else h+=`<div class="wifm">${flag(f.home,'sm')} <span class="wifteam">${sn(f.home)}</span>
        <span class="wifsc">${sb(i,'h',-1,'−')}<b>${hs}</b>${sb(i,'h',1,'+')}<i>-</i>${sb(i,'a',-1,'−')}<b>${as}</b>${sb(i,'a',1,'+')}</span>
        <span class="wifteam ar">${sn(f.away)}</span> ${flag(f.away,'sm')}</div>`;});
    h+=`<table class="wift"><tbody>`;
    gres[L].order.forEach((t,idx)=>{const s=gres[L].st[t],gd=s.gf-s.ga,c=idx<2?'q':(idx===2&&qual3.has(L)?'q3':'');
      h+=`<tr class="${c}"><td class="pos">${idx+1}</td><td>${flag(t,'sm')} ${sn(t)}</td><td>${s.pts}</td><td>${gd>=0?'+':''}${gd}</td></tr>`;});
    h+=`</tbody></table></div>`;}
  el.innerHTML=h+`</div>`;
  el.querySelectorAll(".wifsc .wifb").forEach(btn=>btn.onclick=()=>{const i=+btn.dataset.i,k=btn.dataset.s==='h'?'hs':'as';
    wifScores[i][k]=Math.max(0,Math.min(19,wifScores[i][k]+ +btn.dataset.d));wifRender();bywRender();});
}
/* ---- "build your World Cup": the qualifiers feed a bracket; the model picks
   each knockout winner, you tap the other team to override, up to the champion ---- */
let bywState={};
function assignThirds(qualSet){  // 8 best thirds -> their official R32 slots
  const slots=Object.keys(D.structure.third_elig),el=D.structure.third_elig;
  const qual=[...qualSet].sort(),used=new Set(),sm={};
  (function asg(i){if(i===slots.length)return true;const s=slots[i];
    for(const L of qual){if(used.has(L)||!el[s].includes(L))continue;
      used.add(L);sm[s]=L;if(asg(i+1))return true;used.delete(L);delete sm[s];}
    return false;})(0);
  return sm;
}
function bywWinner(m,a,b){
  if(!a||!b)return a||b||null;
  const ov=bywState[m];if(ov===a||ov===b)return ov;        // your override
  const r=h2h(a,b);return (r.ph+r.pd/2)>=0.5?a:b;          // else the model's pick
}
function bywBracket(){
  const {gres,qual3}=wifGroups(),W={},RU={},TH={};
  for(const L in gres){const o=gres[L].order;W[L]=o[0];RU[L]=o[1];TH[L]=o[2];}
  const sm=assignThirds(qual3);
  const teamOf=sp=>sp[0]==="W"?W[sp[1]]:sp[0]==="RU"?RU[sp[1]]:TH[sm[String(sp[1])]];
  const win={},ties=[],R=D.structure.r32,LA=D.structure.later;
  for(const m in R){const a=teamOf(R[m][0]),b=teamOf(R[m][1]),w=bywWinner(+m,a,b);
    win[m]=w;ties.push({m:+m,rn:"Round of 32",a,b,w});}
  Object.keys(LA).map(Number).sort((x,y)=>x-y).forEach(m=>{
    const a=win[LA[m][0]],b=win[LA[m][1]],w=bywWinner(m,a,b);win[m]=w;
    ties.push({m,rn:roundName(m),a,b,w});});
  return {ties,champ:win[104]};
}
function bywRender(){
  const el=document.getElementById("byw");if(!el)return;if(!wifScores)wifInit();
  const {ties,champ}=bywBracket();
  const rmap={"Round of 32":[],"Round of 16":[],"Quarter-finals":[],"Semi-finals":[],"Final":[]};
  ties.forEach(t=>rmap[t.rn].push({a:t.a,b:t.b,win:t.w,m:t.m,byw:true}));
  const rounds=Object.keys(rmap).map(l=>({label:l,ms:rmap[l]}));
  el.innerHTML=`<div class="bywchamp">🏆 Your champion: ${champ?flag(champ,'sm')+" <b>"+champ+"</b>":"—"}</div>`+bracketTree(rounds);
  el.querySelectorAll(".tt[data-m]").forEach(tt=>tt.onclick=()=>{const t=tt.dataset.team;
    if(t){bywState[+tt.dataset.m]=t;bywRender();}});
}
/* ---- Beat the Machine: predict upcoming matches and score against the model.
   Phase 1 — saved in localStorage, no backend. Only on the live dashboard. ---- */
const PKEY="wc2026_preds_v1";
const SUPA_URL="https://ddpulrjqfxwbvoktdzic.supabase.co";
const SUPA_KEY="sb_publishable_dImupBnCWwySzdyVYFBgew_wfVYoLeP";
let sb=null,sbUser=null,lbRows=[],predCache=null,myName=null,crowdMap={};
function predLoad(){if(predCache)return predCache;
  try{predCache=JSON.parse(localStorage.getItem(PKEY))||{};}catch(e){predCache={};}return predCache;}
function predSet(key,h,a){predLoad();predCache[key]=[h,a];
  try{localStorage.setItem(PKEY,JSON.stringify(predCache));}catch(e){}
  if(sb&&sbUser)sb.from("predictions").upsert({user_id:sbUser.id,match_id:key,pred_home:h,pred_away:a})
    .then(({error})=>{if(error){console.warn("pred upsert:",error.message);predToast("⚠️ Couldn't save — try again");}
      else supaLbRefresh();});
  predToast("✓ Pick saved");predRender();}
function predToast(msg){let t=document.getElementById("ptoast");
  if(!t){t=document.createElement("div");t.id="ptoast";t.className="ptoast";document.body.appendChild(t);}
  t.textContent=msg;t.classList.add("show");clearTimeout(t._h);
  t._h=setTimeout(()=>t.classList.remove("show"),1500);}
let _lbT=null;
function supaLbRefresh(){if(!sb)return;clearTimeout(_lbT);_lbT=setTimeout(async()=>{
  try{const{data}=await sb.from("leaderboard").select("*").order("points",{ascending:false}).limit(50);
    lbRows=data||[];predRender();}catch(e){}},900);}
async function supaInit(){
  if(typeof supabase==="undefined"||!document.getElementById("predict"))return;
  try{sb=supabase.createClient(SUPA_URL,SUPA_KEY);
    sb.auth.onAuthStateChange((_e,s)=>{sbUser=s?s.user:null;if(!s){myName=null;predCache={};}supaSync();});
    const{data:{session}}=await sb.auth.getSession();
    sbUser=session?session.user:null;await supaSync();
  }catch(e){console.warn("Supabase init:",e);}}
async function supaSync(){
  if(!sb)return;
  try{if(sbUser){const{data}=await sb.from("predictions").select("match_id,pred_home,pred_away");
      if(data){predCache={};data.forEach(p=>predCache[p.match_id]=[p.pred_home,p.pred_away]);
        try{localStorage.setItem(PKEY,JSON.stringify(predCache));}catch(e){}}
      const{data:pf}=await sb.from("profiles").select("name").eq("id",sbUser.id).maybeSingle();
      myName=pf&&pf.name?pf.name:((sbUser.user_metadata||{}).full_name||(sbUser.user_metadata||{}).name||null);
      const{data:bk}=await sb.from("brackets").select("picks").eq("user_id",sbUser.id).maybeSingle();
      pbPicks=(bk&&bk.picks)?bk.picks:{};}
    const{data:lb}=await sb.from("leaderboard").select("*").order("points",{ascending:false}).limit(50);
    lbRows=lb||[];
    const{data:cw}=await sb.from("match_crowd").select("*");
    crowdMap={};(cw||[]).forEach(c=>crowdMap[c.match_id]=c);}catch(e){console.warn("Supabase sync:",e);}
  predRender();}
async function supaSignIn(){if(sb)try{await sb.auth.signInWithOAuth(
  {provider:"google",options:{redirectTo:location.href.split("#")[0]}});}
  catch(e){alert("Couldn't start Google sign-in. Try again.");}}
async function supaSignOut(){if(!sb)return;await sb.auth.signOut();
  sbUser=null;myName=null;predCache={};lbRows=[];predRender();supaSync();}
async function supaEditName(){if(!sb||!sbUser)return;
  const n=(prompt("Choose your username (shown on the leaderboard):",myName||"")||"").trim().slice(0,24);
  if(!n)return;
  const{error}=await sb.from("profiles").update({name:n}).eq("id",sbUser.id);
  if(error){predToast("⚠️ Couldn't save name");return;}
  myName=n;predToast("✓ Username saved");supaSync();}
async function showProfile(uid,name){
  if(!sb)return;
  let preds={};
  try{const{data}=await sb.from("predictions").select("match_id,pred_home,pred_away").eq("user_id",uid);
    (data||[]).forEach(p=>preds[p.match_id]=[p.pred_home,p.pred_away]);}catch(e){}
  let pts=0,n=0,exact=0,beat=0;const rows=[];
  (D.played_review||[]).forEach(m=>{const key=m.home+"|"+m.away;if(!(key in preds))return;
    const a=[m.hs,m["as"]],yp=predScore(preds[key],a),mp=predScore(m.ml_score.split("-").map(Number),a);
    pts+=yp;n++;if(yp===5)exact++;if(yp>mp)beat++;rows.push({m,pick:preds[key],a,yp,mp});});
  const tag=p=>p===5?'🎯 +5':p===3?'✅ +3':p===2?'✅ +2':p===1?'⚽ +1':'❌ +0';
  let html=`<div class="pmodbg" id="pmodbg"><div class="pmod"><div class="pmodh"><b>👤 ${name}</b>`
    +`<button class="pbtn" id="pmodx">✕</button></div>`
    +`<div class="pstats" style="margin:4px 0 12px">${n} scored pick${n!==1?'s':''} · ${pts} pts · 🎯 ${exact} exact · 🏆 beat the model ${beat}×</div>`;
  const _pb=predBadges(rows);
  if(_pb.length)html+=`<div class="pbadges">`+_pb.map(x=>`<span class="pbadge" title="${x[2]}">${x[0]} ${x[1]}</span>`).join("")+`</div>`;
  if(rows.length)rows.reverse().forEach(({m,pick,a,yp})=>{
    html+=`<div class="presrow2"><span>${flag(m.home,'sm')} ${m.home} <b>${a[0]}-${a[1]}</b> ${m.away} ${flag(m.away,'sm')}</span>`
      +`<span>picked <b>${pick.join('-')}</b> <span class="ppts ${yp>=2?'g':yp?'a':'r'}">${tag(yp)}</span></span></div>`;});
  else html+=`<div class="note" style="text-align:center;padding:14px 0">No scored picks yet — only matches that have kicked off appear here.</div>`;
  html+=`</div></div>`;
  const old=document.getElementById("pmodbg");if(old)old.remove();
  document.body.insertAdjacentHTML("beforeend",html);
  const bg=document.getElementById("pmodbg");
  bg.onclick=e=>{if(e.target===bg)bg.remove();};
  document.getElementById("pmodx").onclick=()=>bg.remove();}
function predScore(p,a){let s=0;const pd=p[0]-p[1],ad=a[0]-a[1];  // cumulative
  if((pd>0)===(ad>0)&&(pd<0)===(ad<0))s+=2;         // right result
  if(pd===ad)s+=1;                                  // exact goal difference
  if(p[0]===a[0])s+=1;                              // home team's exact goals
  if(p[1]===a[1])s+=1;                              // away team's exact goals
  return s;}
function predKO(home,away){const k=D.kickoffs&&D.kickoffs[[home,away].sort().join("|")];
  return k?new Date(k.date+"T"+k.hm+":00+01:00"):null;}   // WEST = UTC+1
function predModel(key){const m=(D.matches||[]).find(x=>x.home+"|"+x.away===key);
  return m&&m.top&&m.top[0]?m.top[0].score.split("-").map(Number):[1,1];}
/* ---- Knockout-bracket prediction (separate game; opens once the groups finish) ---- */
let pbPicks={},_pbT=null;
function pbBracket(picks){
  const {gres,qual3}=wifGroups(),W={},RU={},TH={};
  for(const L in gres){const o=gres[L].order;W[L]=o[0];RU[L]=o[1];TH[L]=o[2];}
  const sm=assignThirds(qual3);
  const teamOf=sp=>sp[0]==="W"?W[sp[1]]:sp[0]==="RU"?RU[sp[1]]:TH[sm[String(sp[1])]];
  const win={},ties=[],R=D.structure.r32,LA=D.structure.later;
  const pk=(m,a,b)=>{const p=picks[m];return (p===a||p===b)?p:null;};   // your pick only — no model auto-fill
  for(const m in R){const a=teamOf(R[m][0]),b=teamOf(R[m][1]),w=pk(+m,a,b);
    win[m]=w;ties.push({m:+m,rn:"Round of 32",a,b,w});}
  Object.keys(LA).map(Number).sort((x,y)=>x-y).forEach(m=>{
    const a=win[LA[m][0]],b=win[LA[m][1]],w=pk(m,a,b);win[m]=w;
    ties.push({m,rn:roundName(m),a,b,w});});
  return {ties,champ:win[104]};
}
function predBracketRender(){
  const el=document.getElementById("predbracket");if(!el)return;
  if((D.matches||[]).length){el.innerHTML=`<div class="pbrlock">🔒 Opens when the group stage finishes (~28 Jun). You'll fill the full knockout bracket — every tie from the Round of 32 to the title — on its own leaderboard.</div>`;return;}
  if(!sbUser){el.innerHTML=`<div class="pbrlock">🔒 Sign in to fill your knockout bracket.</div>`;return;}
  if(!wifScores)wifInit();
  const {ties,champ}=pbBracket(pbPicks);
  const rmap={"Round of 32":[],"Round of 16":[],"Quarter-finals":[],"Semi-finals":[],"Final":[]};
  ties.forEach(t=>rmap[t.rn].push({a:t.a,b:t.b,win:t.w,m:t.m,byw:true}));
  const rounds=Object.keys(rmap).map(l=>({label:l,ms:rmap[l]})),n=ties.filter(t=>t.w).length;
  el.innerHTML=`<div class="bywchamp">🏆 Your champion: ${champ?flag(champ,'sm')+" <b>"+champ+"</b>":'<span class="note">tap your way through the bracket</span>'}</div>`
    +`<div class="note" style="text-align:center;margin:0 0 8px">Tap a team to send them through · ${n}/${ties.length} ties picked</div>`
    +bracketTree(rounds);
  el.querySelectorAll(".tt[data-m]").forEach(tt=>tt.onclick=()=>{if(tt.dataset.team)pbSet(+tt.dataset.m,tt.dataset.team);});
}
function pbSet(m,team){
  pbPicks[m]=team;predBracketRender();
  if(sb&&sbUser){clearTimeout(_pbT);_pbT=setTimeout(()=>{
    sb.from("brackets").upsert({user_id:sbUser.id,picks:pbPicks},{onConflict:"user_id"})
      .then(({error})=>{predToast(error?"⚠️ Couldn't save bracket":"✓ Bracket saved");});},600);}
}
function predBadges(sc){const b=[];   // achievements earned from scored picks
  if(sc.some(s=>s.yp===5))b.push(['🎯','Bullseye','nailed an exact score']);
  let run=0,mx=0;sc.forEach(s=>{if(s.yp>=2){run++;if(run>mx)mx=run;}else run=0;});
  if(mx>=3)b.push(['🔥','On fire',mx+' correct results in a row']);
  const beat=sc.filter(s=>s.yp>s.mp).length;
  if(beat>=3)b.push(['🧠','Machine beater','beat the model '+beat+' times']);
  const byday={};sc.forEach(s=>{(byday[s.m.date]=byday[s.m.date]||[]).push(s);});
  if(Object.values(byday).some(d=>d.length>=2&&d.every(s=>s.yp>=2)))b.push(['💯','Perfect matchday','every pick right on a matchday']);
  if(sc.some(s=>s.yp>=2&&s.m.p_actual<0.30))b.push(['🦅','Giant-killer','called a result the model gave under 30%']);
  return b;}
function predRender(){
  const el=document.getElementById("predict");if(!el)return;
  const store=predLoad(),now=new Date();
  let you=0,mac=0,n=0,hits=0,exact=0,beat=0,run=0,streak=0;const scored=[];
  (D.played_review||[]).forEach(m=>{const key=m.home+"|"+m.away;if(!(key in store))return;
    const a=[m.hs,m["as"]],yp=predScore(store[key],a),mp=predScore(m.ml_score.split("-").map(Number),a);
    you+=yp;mac+=mp;n++;if(yp>=2){hits++;run++;}else run=0;streak=run;if(yp===5)exact++;if(yp>mp)beat++;
    scored.push({m,yp,mp,a});});
  const signedIn=!!sbUser;
  let h=signedIn
    ?`<div class="pauth"><span>👤 <b>${myName||'You'}</b> <button class="plink" id="p-editname">✎ username</button></span><button class="pbtn" id="p-signout">Sign out</button></div>`
    :`<div class="pauth"><span>🔒 Sign in to predict &amp; climb the leaderboard</span><button class="pbtn pg" id="p-signin">Sign in with Google</button></div>`;
  if(signedIn){if(n>0){const lead=you>mac?`You're ${you-mac} ahead 🟢`:you<mac?`The model's ${mac-you} ahead 🔴`:`Dead level 🤝`;
    h+=`<div class="pboard"><div class="pb"><div class="pbn">${you}</div><div class="pbl">You</div></div>`
      +`<div class="pbvs">${lead}</div><div class="pb"><div class="pbn">${mac}</div><div class="pbl">🤖 Machine</div></div></div>`
      +`<div class="pstats">over ${n} match${n>1?'es':''} · ${Math.round(hits/n*100)}% result hit-rate · 🎯 ${exact} exact · 🔥 streak ${streak}${beat?` · 🏆 beat the model ${beat}×`:''}</div>`;
    const _bd=predBadges(scored);
    if(_bd.length)h+=`<div class="pbadges">`+_bd.map(x=>`<span class="pbadge" title="${x[2]}">${x[0]} ${x[1]}</span>`).join("")+`</div>`;
  }else h+=`<div class="pstats" style="padding:6px 0 14px">Make your picks below — your score vs the model shows up here after kickoff. 👇</div>`;}
  const _lb=lbRows.filter(r=>!r.is_model&&(r.played>0||(sbUser&&r.uid===sbUser.id)));
  if(_lb.length){const myIdx=sbUser?_lb.findIndex(r=>r.uid===sbUser.id):-1;
    h+=`<h3 class="psec">🏆 Global leaderboard <span class="tag">tap a player to see their picks</span></h3>`;
    if(myIdx>0){const rv=_lb[myIdx-1],gap=rv.points-_lb[myIdx].points;
      h+=`<div class="prival">⚔️ <b>${rv.name||'Player'}</b> is ${gap} pt${gap!==1?'s':''} ahead — catch them!</div>`;}
    else if(myIdx===0&&_lb.length>1)h+=`<div class="prival">👑 You lead the leaderboard — defend your spot!</div>`;
    h+=`<div class="lbx">`;
    _lb.slice(0,30).forEach((r,i)=>{const me=sbUser&&r.uid===sbUser.id,riv=i===myIdx-1;
      h+=`<div class="lbrow lbclick${me?' lbme':''}${riv?' lbrival':''}" data-uid="${r.uid}" data-name="${(r.name||'Player').replace(/"/g,'&quot;')}"><span class="lbrank">${i+1}</span>`
        +`<span class="lbname">${me?'🟢':riv?'⚔️':'🧑'} ${r.name||'Player'}${me?' · you':''}</span>`
        +`<span class="lbpts">${r.points}</span></div>`;});
    h+=`</div>`;}
  if(signedIn){
  const up=(D.matches||[]).map(m=>({m,ko:predKO(m.home,m.away)}))
    .sort((a,b)=>(a.ko?a.ko:9e15)-(b.ko?b.ko:9e15));
  const more=Math.max(0,up.length-12),show=up.slice(0,12);
  h+=`<h3 class="psec">📥 Predict the upcoming matches</h3>`;
  const stp=(key,s,d,l)=>`<button class="wifb" data-k="${key}" data-s="${s}" data-d="${d}">${l}</button>`;
  show.forEach(({m,ko})=>{const key=m.home+"|"+m.away,mine=key in store,cur=store[key]||predModel(key),
    locked=ko&&now>=ko,kl=ko?D.kickoffs[[m.home,m.away].sort().join("|")].label:"";
    let sc;
    if(locked)sc=mine?`<span class="psc"><b>${cur[0]}</b><i>-</i><b>${cur[1]}</b></span>`:`<span class="psc" style="color:#5d6a85">—</span>`;
    else{const dh=mine?cur[0]:'—',da=mine?cur[1]:'—';   // — until the user actually picks (don't pre-fill the model's call)
      sc=`<span class="psc">${stp(key,'h',-1,'−')}<b>${dh}</b>${stp(key,'h',1,'+')}<i>-</i>${stp(key,'a',-1,'−')}<b>${da}</b>${stp(key,'a',1,'+')}</span>`;}
    h+=`<div class="pcard${locked?' plk':''}${mine?' pmine':''}"><div class="prow"><span class="pteam">${flag(m.home,'sm')} ${m.home}</span>${sc}<span class="pteam ar">${m.away} ${flag(m.away,'sm')}</span></div>`;
    const meta=locked?('🔒 locked'+(mine?' · <span class="pdone">✓ your pick</span>':' — not predicted')):('🕒 '+kl+(mine?' · <span class="pdone">✓ your pick</span>':''));
    h+=`<div class="pmeta">${meta}</div>`;
    if(!locked&&m.top)h+=`<div class="psugg">💡 The model's call: `+m.top.map(s=>`<button class="pchip" data-k="${key}" data-sc="${s.score}">${s.score} · ${Math.round(s.p*100)}%</button>`).join("")+`</div>`;
    const _cw=crowdMap[key];if(_cw&&_cw.n>0)h+=`<div class="pcrowd">👥 Crowd (${_cw.n}): ${m.home} ${_cw.home_pct}% · Draw ${_cw.draw_pct}% · ${m.away} ${_cw.away_pct}%</div>`;
    h+=`</div>`;});
  if(more)h+=`<p class="note">…and ${more} more group match${more>1?'es':''} to come — predict the imminent ones first.</p>`;
  if(scored.length){h+=`<h3 class="psec">📊 Your results so far</h3>`;
    const tag=p=>p===5?'🎯 +5':p===3?'✅ +3':p===2?'✅ +2':p===1?'⚽ +1':'❌ +0';
    scored.slice().reverse().forEach(({m,yp,mp,a})=>{const key=m.home+"|"+m.away;
      h+=`<div class="pcard pres"><div class="prow"><span class="pteam">${flag(m.home,'sm')} ${m.home}</span> <b>${a[0]} – ${a[1]}</b> <span class="pteam ar">${m.away} ${flag(m.away,'sm')}</span></div>`
        +`<div class="presrow"><span>You said <b>${store[key].join('-')}</b></span><span class="ppts ${yp>=2?'g':yp?'a':'r'}">${tag(yp)}</span></div>`
        +`<div class="presrow mac"><span>🤖 Machine said <b>${m.ml_score}</b></span><span class="ppts ${mp>=2?'g':mp?'a':'r'}">${tag(mp)}</span></div>`
        +(yp>mp?'<div class="pbeat">You beat the model! 🎉</div>':mp>yp?'<div class="pbeat lose">Model won this one</div>':'')+`</div>`;});}
  }else h+=`<div class="pcard" style="text-align:center;padding:22px 16px"><div style="font-size:15px;font-weight:700;margin-bottom:6px">🔒 Sign in to make your predictions</div><div class="note" style="margin:0">Log in with Google to pick scores, challenge the model and climb the leaderboard.</div><button class="pbtn pg" id="p-signin2" style="margin-top:12px">Sign in with Google</button></div>`;
  h+=`<h3 class="psec">🏆 Knockout bracket <span class="tag">your path to the title · separate ranking</span></h3><div id="predbracket"></div>`;
  el.innerHTML=h;
  predBracketRender();
  const _si=document.getElementById("p-signin");if(_si)_si.onclick=supaSignIn;
  const _si2=document.getElementById("p-signin2");if(_si2)_si2.onclick=supaSignIn;
  const _so=document.getElementById("p-signout");if(_so)_so.onclick=supaSignOut;
  const _en=document.getElementById("p-editname");if(_en)_en.onclick=supaEditName;
  el.querySelectorAll(".lbclick").forEach(r=>r.onclick=()=>showProfile(r.dataset.uid,r.dataset.name));
  el.querySelectorAll(".psc .wifb").forEach(b=>b.onclick=()=>{const key=b.dataset.k,fld=b.dataset.s==='h'?0:1;
    const cur=(predLoad()[key]||[0,0]).slice();cur[fld]=Math.max(0,Math.min(19,cur[fld]+ +b.dataset.d));predSet(key,cur[0],cur[1]);});
  el.querySelectorAll(".pchip").forEach(c=>c.onclick=()=>{const s=c.dataset.sc.split("-").map(Number);predSet(c.dataset.k,s[0],s[1]);});
}
function renderToday(){
  const sec=document.getElementById("todaysec");if(!sec)return;
  const ko=D.kickoffs||{},K=(a,b)=>ko[[a,b].sort().join("|")];
  const predBy={};(D.matches||[]).forEach(m=>predBy[[m.home,m.away].sort().join("|")]=m);
  const revBy={};(D.played_review||[]).forEach(m=>revBy[[m.home,m.away].sort().join("|")]=m);
  const fx=(D.fixtures||[]).map(f=>{const k=K(f.home,f.away);return k?Object.assign({},f,{ko:k}):null;}).filter(Boolean);
  const now=new Date(),lis=new Date(now.getTime()+(now.getTimezoneOffset()+60)*60000);  // Lisbon = UTC+1
  const today=lis.toISOString().slice(0,10);
  let day=fx.filter(f=>f.ko.date===today),head="Today's matches";
  if(!day.length){const fut=fx.filter(f=>f.ko.date>today).sort((a,b)=>a.ko.date<b.ko.date?-1:1);
    if(fut.length){const d=fut[0].ko.date;day=fx.filter(f=>f.ko.date===d);
      head="Next up · "+fut.find(f=>f.ko.date===d).ko.label.split(" · ")[0];}}
  if(!day.length){sec.style.display="none";return;}
  day.sort((a,b)=>a.ko.hm<b.ko.hm?-1:1);
  const hd=document.getElementById("todayhead");if(hd)hd.textContent=head;
  let h=`<div class="todaycards">`;
  day.forEach(f=>{const key=[f.home,f.away].sort().join("|"),rev=revBy[key],pred=predBy[key];let info="";
    if(rev)info=`<div class="tdres">Full time <b>${rev.hs}-${rev["as"]}</b> ${rev.hit?'<span class="badge2 y">model ✓</span>':'<span class="badge2 n">model ✗</span>'}</div>`;
    else if(f.played)info=`<div class="tdres">Full time <b>${f.hs}-${f["as"]}</b></div>`;
    else if(pred)info=`<div class="tdpred">Model ${pc0(pred.p_home)} / ${pc0(pred.p_draw)} / ${pc0(pred.p_away)} · likely <b>${pred.top[0].score}</b></div>`;
    h+=`<div class="todaycard"><div class="tdtime">🕒 ${f.ko.hm} · Group ${f.group}</div>
    <div class="tdteams">${flag(f.home,'sm')} ${f.home} <span class="tdvs">v</span> ${f.away} ${flag(f.away,'sm')}</div>${info}</div>`;});
  document.getElementById("today").innerHTML=h+`</div>`;
}
function renderSurprises(){
  const rv=D.played_review,el=document.getElementById("surprises");if(!el)return;
  if(!rv||!rv.length){el.style.display="none";return;}
  const shocks=[...rv].sort((a,b)=>a.p_actual-b.p_actual).slice(0,5);
  let h=`<p class="note">The results the model least saw coming — it gave the actual outcome only a slim chance:</p><div class="chips">`;
  h+=shocks.map(m=>`<div class="chip">${flag(m.home,'sm')} ${m.home} <b>${m.hs}-${m["as"]}</b> ${m.away} ${flag(m.away,'sm')} <span style="color:#ff6b6b;font-weight:700">${pc0(m.p_actual)}</span></div>`).join("");
  el.innerHTML=h+`</div>`;
}
function renderGoals(){
  const g=D.goals,el=document.getElementById("goals-an");if(!el)return;
  if(!g||!g.by_minute){el.style.display="none";return;}
  const mx=Math.max(...g.by_minute.map(b=>b.count))||1;
  const bars=g.by_minute.map(b=>{const top=b.count===mx;
    return `<div class="gminute"><div class="gmval">${b.count.toLocaleString()}</div>
    <div class="gmbar" title="${b.label} min: ${b.count.toLocaleString()} goals">
    <div style="height:${(b.count/mx*100).toFixed(0)}%;background:${top?'linear-gradient(#ffd34d,#e0a800)':'linear-gradient(var(--green),#00b083)'}"></div></div>
    <div class="gmlbl">${b.label}</div></div>`;}).join("");
  let h=`<p class="note">An x-ray of <b>${g.total_goals.toLocaleString()}</b> international goals since ${g.since} — World Cups, qualifiers and friendlies, the lot.</p>
  <div class="note" style="margin:8px 0 2px">⏱️ When goals are scored <span class="tag">by minute</span></div>
  <div class="gminutes">${bars}</div>
  <p class="note" style="margin-top:12px">The final 15 minutes are the deadliest. 🎯 <b>${g.pen_pct}%</b> of goals come from the penalty spot · 🙃 <b>${g.og_pct}%</b> are own goals.</p>`;
  if(g.top_scorers&&g.top_scorers.length){
    h+=`<p class="note" style="margin-top:12px"><b>👟 Top scorers of the World Cup so far:</b></p><div class="chips">`+
      g.top_scorers.map(s=>`<div class="chip">${flag(s.team,'sm')} ${s.scorer} <b style="color:#00e0a4">${s.goals}</b></div>`).join("")+`</div>`;}
  el.innerHTML=h;
}
function renderAnalysis(){
  const a=D.analysis;if(!a)return;
  const god=a.group_of_death[0];
  const dh=a.dark_horses.map(d=>`<div class="chip">${flag(d.team,'sm')} ${d.team} <span style="color:#8b95ab">${pc0(d.p_qf)} QF</span></div>`).join("");
  const lf=a.likely_finals.map(f=>`<div class="chip">${flag(f.a,'sm')} ${f.a} v ${f.b} ${flag(f.b,'sm')} <span style="color:#8b95ab">${pc0(f.p)}</span></div>`).join("");
  const teamsList=god.teams.map((t,i)=>`${flag(t,'sm')} ${t} <span style="color:#8b95ab">${god.elos[i]}</span>`).join(" · ");
  document.getElementById("analysis").innerHTML=`
  <p class="note"><b>💀 Group of death — Group ${god.group}:</b> average Elo <b>${god.avg_elo}</b>,
   and even the 3rd-strongest side sits on Elo <b>${god.third_elo}</b> — a brutal fight to advance.<br>${teamsList}</p>
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

function poisS(l){let L=Math.exp(-l),k=0,p=1;do{k++;p*=Math.random();}while(p>L);return k-1;}
function lamV(a,b,va){const M=D.model,sa=D.state[a],sb=D.state[b];
  const rows=[{elo_self:sa.elo,elo_opp:sb.elo,venue:va,is_friendly:0,self_gf:sa.gf,opp_ga:sb.ga},
              {elo_self:sb.elo,elo_opp:sa.elo,venue:-va,is_friendly:0,self_gf:sb.gf,opp_ga:sa.ga}];
  return rows.map(r=>{let e=M.beta[0];M.features.forEach((f,i)=>{let v=r[f];
    if(M.continuous.includes(f))v=(v-M.mu[f])/M.sd[f];e+=M.beta[i+1]*v;});return Math.exp(e);});}
function hostV(x,y){const H=D.structure.hosts,hx=H.includes(x),hy=H.includes(y);
  return hx&&!hy?0.5:(hy&&!hx?-0.5:0);}
function koGame(x,y){const[lx,ly]=lamV(x,y,hostV(x,y));let sx=poisS(lx),sy=poisS(ly),pe=false,w;
  if(sx>sy)w=x;else if(sy>sx)w=y;else{sx+=poisS(lx/3);sy+=poisS(ly/3);
    if(sx>sy)w=x;else if(sy>sx)w=y;else{pe=true;
      const pa=1/(1+Math.exp(-D.shoot_b1*(D.state[x].elo-D.state[y].elo)));w=Math.random()<pa?x:y;}}
  return{w,sx,sy,pe};}
function roundName(m){m=+m;return m<=96?"Round of 16":m<=100?"Quarter-finals":m<=102?"Semi-finals":"Final";}
function rollTournament(){
  const G=D.structure.groups,gres={};
  for(const L in G){const ts=G[L],st={},gm=[];ts.forEach(t=>st[t]={t,pts:0,gd:0,gf:0});
    D.fixtures.filter(f=>f.group===L).forEach(f=>{let hs,as_;
      if(f.played){hs=f.hs;as_=f["as"];}else{const nv=f.neutral?0:1;
        const[lh,la]=lamV(f.home,f.away,nv);hs=poisS(lh);as_=poisS(la);}
      const H=st[f.home],A=st[f.away];
      if(hs>as_)H.pts+=3;else if(as_>hs)A.pts+=3;else{H.pts++;A.pts++;}
      H.gd+=hs-as_;A.gd+=as_-hs;H.gf+=hs;A.gf+=as_;
      gm.push({home:f.home,away:f.away,hs,as:as_,played:!!f.played});});
    const order=ts.slice().sort((x,y)=>st[y].pts-st[x].pts||st[y].gd-st[x].gd||st[y].gf-st[x].gf||Math.random()-.5);
    gres[L]={order,st,matches:gm};}
  const win={},ru={},third={},ts3=[];
  for(const L in gres){const o=gres[L].order;win[L]=o[0];ru[L]=o[1];third[L]=o[2];
    ts3.push(Object.assign({L},gres[L].st[o[2]]));}
  ts3.sort((a,b)=>b.pts-a.pts||b.gd-a.gd||b.gf-a.gf||Math.random()-.5);
  const qual=ts3.slice(0,8).map(x=>x.L),slots=Object.keys(D.structure.third_elig),el=D.structure.third_elig;
  (function(){const used=new Set();window._sm={};
    (function asg(i){if(i===slots.length)return true;const s=slots[i];
      for(const L of qual){if(used.has(L)||!el[s].includes(L))continue;
        used.add(L);window._sm[s]=L;if(asg(i+1))return true;used.delete(L);delete window._sm[s];}
      return false;})(0);})();
  const sm=window._sm;
  const teamOf=sp=>sp[0]==="W"?win[sp[1]]:sp[0]==="RU"?ru[sp[1]]:third[sm[String(sp[1])]];
  const wk={},ko=[],R=D.structure.r32;
  for(const m in R){const a=teamOf(R[m][0]),b=teamOf(R[m][1]),r=koGame(a,b);
    wk[m]=r.w;ko.push({rn:"Round of 32",m:+m,a,b,sx:r.sx,sy:r.sy,pe:r.pe,w:r.w});}
  const LA=D.structure.later;
  Object.keys(LA).map(Number).sort((x,y)=>x-y).forEach(m=>{
    const a=wk[LA[m][0]],b=wk[LA[m][1]],r=koGame(a,b);
    wk[m]=r.w;ko.push({rn:roundName(m),m:+m,a,b,sx:r.sx,sy:r.sy,pe:r.pe,w:r.w});});
  renderRoll(gres,qual,ko,wk[104]);
}
function renderRoll(gres,qual,ko,champ){
  const eloOf=t=>(byName[t]||{}).elo||0;
  const sf=ko.filter(g=>g.rn==="Semi-finals").flatMap(g=>[g.a,g.b]);
  const sup=sf.slice().sort((a,b)=>eloOf(a)-eloOf(b))[0];
  let h=`<div class="rollbox"><div class="champ">🏆 ${flag(champ)} ${champ} are champions!</div>`;
  if(sup)h+=`<div class="note" style="margin-top:4px">Surprise package: ${flag(sup,'sm')} <b>${sup}</b> made the semis (Elo ${eloOf(sup)}).</div>`;
  h+=`</div><h3 style="margin:6px 0 8px">Group tables <span class='tag'>expand for match results</span></h3><div class="grid">`;
  for(const L in gres){h+=`<div class="group"><h3>GROUP ${L}</h3>`;
    gres[L].order.forEach((t,i)=>{const c=i<2?'q':(i===2&&qual.includes(L)?'q3':'');const s=gres[L].st[t];
      h+=`<div class="g-row ${c}"><span class="pos">${i+1}</span>
      <span class="nm">${flag(t,'sm')} ${t}</span>
      <span style="color:#8b95ab;font-size:11px">${s.pts}p ${s.gd>=0?'+':''}${s.gd}</span></div>`;});
    const mr=gres[L].matches.map(g=>`<div class="gr">${flag(g.home,'sm')} ${g.home} <b>${g.hs}-${g["as"]}</b> ${g.away} ${flag(g.away,'sm')}${g.played?' <span title="real result" style="color:#00e0a4">●</span>':''}</div>`).join("");
    h+=`<details class="gres"><summary>match results</summary>${mr}</details></div>`;}
  h+=`</div><h3 style="margin:20px 0 8px">Knockout bracket</h3>`;
  const rmap={"Round of 32":[],"Round of 16":[],"Quarter-finals":[],"Semi-finals":[],"Final":[]};
  ko.forEach(g=>rmap[g.rn].push({a:g.a,b:g.b,win:g.w,m:g.m,va:""+g.sx,vb:""+g.sy,pe:g.pe}));
  h+=bracketTree(Object.keys(rmap).map(l=>({label:l,ms:rmap[l]})));
  document.getElementById("sim-out").innerHTML=h;
}

function renderOdds(){
  const oh=D.odds_history;if(!oh||typeof Chart==="undefined")return;
  const cols=["#00e0a4","#ffd34d","#5b8def","#ff6b6b","#c08cff","#ff9f43","#4dd0e1","#9ccc65"];
  oddsChart=new Chart(document.getElementById("odds-chart"),{type:"line",
    data:{labels:oh.dates.map(d=>d.slice(5)),datasets:oh.series.map((s,i)=>({label:s.team,
      data:s.data.map(v=>+(v*100).toFixed(1)),borderColor:cols[i%8],backgroundColor:cols[i%8],
      tension:.3,borderWidth:i===0?3:2,pointRadius:3}))},
    options:{maintainAspectRatio:false,
      plugins:{legend:{labels:{color:"#8b95ab",boxWidth:12,font:{size:11}}}},
      scales:{y:{title:{display:true,text:"Title probability (%)",color:"#8b95ab"},
        ticks:{color:"#8b95ab"},grid:{color:"#243049"}},
        x:{ticks:{color:"#8b95ab"},grid:{color:"#243049"}}}}});}
let eloChart=null,oddsChart=null;
function showTab(name,scroll){
  document.querySelectorAll(".tab").forEach(t=>t.style.display=t.id==="tab-"+name?"":"none");
  document.querySelectorAll(".tabnav button").forEach(b=>b.classList.toggle("on",b.dataset.tab===name));
  const u=new URLSearchParams(location.search);u.set("tab",name);
  history.replaceState(null,"","?"+u.toString());
  [oddsChart,eloChart].forEach(c=>{if(c)try{c.resize();}catch(e){}});
  if(scroll){const n=document.querySelector(".tabnav");if(n)n.scrollIntoView({block:"start"});}
}
function renderEloYear(){
  if(typeof Chart==="undefined")return;
  const y=document.getElementById("elo-year").value;
  document.getElementById("elo-year-lbl").textContent=y;
  const rows=(D.elo_by_year[y]||[]).slice().reverse();
  if(eloChart)eloChart.destroy();
  const RC=["#00e0a4","#ffd34d","#5b8def","#ff6b6b","#c08cff","#ff9f43","#4dd0e1",
    "#9ccc65","#f78fb3","#7ed6df","#e77f67","#f6c945"];
  eloChart=new Chart(document.getElementById("elo-chart"),{type:"bar",
    data:{labels:rows.map(r=>r.team),datasets:[{data:rows.map(r=>r.elo),
      backgroundColor:rows.map((_,i)=>RC[i%RC.length])}]},
    options:{indexAxis:"y",maintainAspectRatio:false,
      plugins:{legend:{display:false},title:{display:true,text:"Top teams by Elo — "+y,
        color:"#eef1f7",font:{size:15,family:"Outfit"}}},
      scales:{x:{min:1400,ticks:{color:"#8b95ab"},grid:{color:"#243049"}},
        y:{ticks:{color:"#eef1f7"},grid:{display:false}}}}});}

/* collapsible sections: tap the heading to expand/collapse (chevron). The
   sections marked 'mcol' start collapsed on phones to keep the page short. */
function makeCollapsible(){
  const mob=innerWidth<=680;
  document.querySelectorAll("h2.col").forEach(h=>{
    const body=document.createElement("div");body.className="secbody";
    let n=h.nextElementSibling;
    while(n&&n.tagName!=="H2"&&!n.classList.contains("foot")){
      const nx=n.nextElementSibling;body.appendChild(n);n=nx;}
    h.after(body);
    const set=o=>{h.classList.toggle("open",o);body.style.display=o?"":"none";};
    h.addEventListener("click",()=>set(!h.classList.contains("open")));
    set(!(mob&&h.classList.contains("mcol")));
  });
}
renderRanking();renderDetail();renderGroups();renderMatches();buildLab();
renderBracket();renderAnalysis();renderBacktest();renderGolden();renderOdds();renderReview();renderFifa();renderGoals();
renderToday();renderSurprises();
if(D.elo_by_year&&Object.keys(D.elo_by_year).length){renderEloYear();
  document.getElementById("elo-year").oninput=renderEloYear;}
document.getElementById("mfilter").onchange=renderMatches;
document.getElementById("search").oninput=e=>{query=e.target.value.toLowerCase();renderRanking();};
document.getElementById("sim-btn").onclick=rollTournament;
if(document.getElementById("whatif")){wifRender();
  const _rb=document.getElementById("wif-reset");if(_rb)_rb.onclick=()=>{wifInit();wifRender();bywRender();};}
if(document.getElementById("byw")){bywRender();
  const _br=document.getElementById("byw-reset");if(_br)_br.onclick=()=>{bywState={};bywRender();};}
if(document.getElementById("predict")){predRender();supaInit();}
makeCollapsible();
document.querySelectorAll(".tabnav button").forEach(b=>b.onclick=()=>showTab(b.dataset.tab,true));
let _tab=new URLSearchParams(location.search).get("tab");
if(qteam&&byName[qteam])_tab="teams";   // a ?team= link always lands on the Teams tab
showTab(["overview","teams","play","predict","insights"].includes(_tab)?_tab:"overview",false);
const _sec=new URLSearchParams(location.search).get("sec");   // deep-link to a section
if(_sec){const _se=document.getElementById(_sec);
  if(_se)setTimeout(()=>_se.scrollIntoView({behavior:"smooth",block:"center"}),180);}
if(qteam&&byName[qteam]){const r=document.querySelector("tr.sel");
  if(r)r.scrollIntoView({behavior:"smooth",block:"center"});
  teamPage(qteam);}
"""


def _mega_html(m: dict) -> str:
    if not m:
        return ""
    o = m["overall"]
    rows = "".join(
        f"<tr><td>{c['competition']}</td><td>{c['n']}</td>"
        f"<td><b>{c['acc']*100:.0f}%</b></td><td style='color:#8b95ab'>{c['acc_naive']*100:.0f}%</td>"
        f"<td>{c['rps']:.3f}</td></tr>" for c in m["by_competition"])
    yr = m.get("start_year", 2002)
    return (
        f"<p class='note'>The honest test: for <b>every</b> major tournament since "
        f"{yr}, the model is retrained on only what was known <b>before that "
        f"tournament kicked off</b> and then predicts it blind — exactly how you'd "
        f"have used it in real time. That's <b>{o['n']:,} matches across "
        f"{o['n_editions']} editions</b> (World Cups, Euros, Copa América, Nations "
        f"League, Asian &amp; African Cups, Gold Cups). Verdict: <b>{o['acc']*100:.0f}% "
        f"of results called correctly</b> (RPS {o['rps']:.3f}) vs the naive base "
        f"rate's {o['acc_naive']*100:.0f}% ({o['rps_naive']:.3f}).</p>"
        "<table><tr><th>Competition</th><th>Matches</th><th>Model</th>"
        f"<th>Naive</th><th>RPS</th></tr>{rows}</table>")


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
        "<h2 class='col mcol'>🕰️ Historical facts <span class='tag'>1872–2026</span></h2>"
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
    oh = data.get("odds_history")
    ev_html = ""
    if oh or (ev and ev.get("movers")):
        parts = ["<h2>📊 Odds over time <span class='tag'>since kickoff</span></h2>"]
        if oh:
            parts.append("<div style='position:relative;height:340px'>"
                         "<canvas id='odds-chart'></canvas></div>")
        if ev and ev.get("movers"):
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
    eby_years = sorted(int(y) for y in data.get("elo_by_year", {}))
    ely_min, ely_max = (eby_years[0], eby_years[-1]) if eby_years else (1960, 2026)
    fl = lambda d: (f'<img class="flag" src="https://flagcdn.com/w40/{d["code"]}.png">'
                    if d.get("code") else d.get("flag", ""))
    mode = (data.get("mode_label") + " · ") if data.get("mode_label") else ""
    pre = "Pre-tournament" in (data.get("mode_label") or "")
    ogdir = "/outputs_pretournament" if pre else "/outputs"
    # Beat-the-Machine prediction game: live version only (needs real results)
    pred_btn = "" if pre else "<button data-tab='predict'>⚔️ Predict</button>"
    pred_tab = ("" if pre else
        "<div class='tab' id='tab-predict'>"
        "<h2>⚔️ Beat the Machine <span class='tag'>you vs the model</span></h2>"
        "<p class='note'>Predict the upcoming matches — tap the model's call to fill it in, "
        "or set your own with −/+. Locked at kickoff. Points stack up: ✅ right result +2 · 📏 goal "
        "difference +1 · ⚽ each team's exact goals +1 (perfect = 🎯 5). The model plays too — can you beat it? Sign in with "
        "Google to make your picks, choose a username and climb the global leaderboard.</p>"
        "<div id='predict'></div></div>")
    og = (  # Open Graph (rich link previews) + PWA / add-to-home-screen
        f'<meta property="og:title" content="World Cup 2026 — ML prediction'
        f'{" (pre-tournament)" if pre else ""}">'
        '<meta property="og:description" content="Who wins the 2026 World Cup? A '
        f'Poisson + Elo model over {data["n_sims"]:,} simulations — title odds, the '
        'bracket, a Match Lab and more.">'
        f'<meta property="og:image" content="{SITE}{ogdir}/share_card.png">'
        '<meta property="og:image:width" content="1200">'
        '<meta property="og:image:height" content="630">'
        f'<meta property="og:url" content="{SITE}{ogdir}/dashboard.html">'
        '<meta property="og:type" content="website">'
        '<meta name="twitter:card" content="summary_large_image">'
        '<meta name="theme-color" content="#0c1018">'
        '<meta name="apple-mobile-web-app-capable" content="yes">'
        '<meta name="apple-mobile-web-app-title" content="WC 2026">'
        '<link rel="manifest" href="../manifest.json">'
        '<link rel="apple-touch-icon" href="../apple-touch-icon.png">'
    )
    review_html = ("<h2 class='col mcol'>✅ Results so far <span class='tag'>predicted vs actual</span></h2>"
                   "<div id='review'></div>"
                   "<h2 class='col mcol'>😱 Biggest surprises <span class='tag'>model's worst calls</span></h2>"
                   "<div id='surprises'></div>") if data.get("played_review") else ""

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>FIFA World Cup 2026 — ML Prediction</title>"
        f"{FAVICON}{og}"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;800"
        "&display=swap' rel='stylesheet'>"
        "<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'></script>"
        "<script src='https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2'></script>"
        f"<style>{_CSS}</style></head><body>"
        "<div class='wrap'>"
        "<div class='hero'><div class='badge'>MACHINE LEARNING PREDICTION</div>"
        "<h1>🏆 FIFA World Cup 2026</h1>"
        f"<p class='sub'>{mode}Poisson (Dixon-Coles) + Elo · {data['n_sims']:,} Monte Carlo "
        f"simulations · click a team or try the Match Lab</p>"
        f"{val_html}</div>"
        "<div class='tabnav'>"
        "<a class='tabhome' href='../index.html' aria-label='Home'>🏠</a>"
        "<button data-tab='overview' class='on'>🔮 Overview</button>"
        "<button data-tab='teams'>📊 Teams</button>"
        "<button data-tab='play'>🎮 Play</button>"
        f"{pred_btn}"
        "<button data-tab='insights'>📈 Insights</button></div>"
        "<div class='tab' id='tab-overview'>"
        "<div class='cards'>"
        f"<div class='kpi'><div class='big'>{fl(fav)} {fav['p']*100:.1f}%</div>"
        f"<div class='lbl'>Favourite: {fav['team']}</div></div>"
        f"<div class='kpi'><div class='big'>{fl(sec)} {sec['p']*100:.1f}%</div>"
        f"<div class='lbl'>2nd favourite: {sec['team']}</div></div>"
        f"<div class='kpi'><div class='big'>{fl(fin[0])} {fl(fin[1])}</div>"
        f"<div class='lbl'>Most likely finalists</div></div>"
        f"<div class='kpi'><div class='big'>{contenders}</div>"
        f"<div class='lbl'>teams above 5% to win</div></div></div>"
        "<div id='todaysec'><h2>📅 <span id='todayhead'>Today's matches</span> "
        "<span class='tag'>kickoffs in WEST (UTC+1)</span></h2><div id='today'></div></div>"
        f"{ev_html}"
        "<h2 class='col mcol'>Most likely bracket <span class='tag'>the favourites' path</span></h2>"
        "<div id='bracket'></div>"
        "</div>"
        "<div class='tab' id='tab-teams'>"
        "<h2>Ranking & path <span class='tag'>click a row</span></h2>"
        "<div class='layout'><div><input class='search' id='search' "
        "placeholder='🔎 Search a team…'><div id='ranking'></div></div>"
        "<div class='panel sticky' id='detail'></div></div>"
        "<h2 class='col mcol'>The 12 groups <span class='tag'>live standings · who advances</span></h2>"
        "<p style='color:#8b95ab;font-size:13px;margin:-2px 0 12px;max-width:760px'>"
        "Sorted by points from games already played. The figure on the right is the model's "
        "<b>chance each team reaches the knockouts</b> — it already accounts for the "
        "8-best-third-placed-teams rule. <b style='color:#00e0a4'>Green</b> = very likely, "
        "<b style='color:#ff6b6b'>red</b> = very unlikely.</p>"
        "<div class='grid' id='groups'></div>"
        "<h2 class='col mcol'>Group-stage matches <span class='tag'>times in WEST (UTC+1)</span></h2>"
        f"<select id='mfilter'>{opts}</select><div id='matches' style='margin-top:10px'></div>"
        "<h2 class='col mcol'>📈 Extra analysis</h2><div id='analysis'></div>"
        "</div>"
        "<div class='tab' id='tab-play'>"
        "<h2>🆚 Match Lab <span class='tag'>head-to-head, live</span></h2>"
        "<div class='lab' id='lab'></div><div class='mlab-out' id='lab-out'></div>"
        "<h2 class='col mcol'>🏆 Build your World Cup <span class='tag'>you decide</span></h2>"
        "<p class='note'><b>1.</b> Set the score of every remaining group match (−/+ the "
        "goals, from the model's prediction); the tables and who qualifies update live. "
        "<button class='btn' id='wif-reset' style='padding:6px 12px;font-size:13px'>↺ Reset scores</button></p>"
        "<div id='whatif'></div>"
        "<p class='note' style='margin-top:20px'><b>2.</b> Now the knockouts — the model picks "
        "each winner, but <b>tap the other team</b> to send your pick through, all the way to "
        "your champion. <button class='btn' id='byw-reset' style='padding:6px 12px;font-size:13px'>↺ Model's bracket</button></p>"
        "<div id='byw'></div>"
        "<h2 class='col mcol'>🎲 Roll a tournament <span class='tag'>one full simulation</span></h2>"
        "<p class='note'>One possible World Cup — group tables, every knockout result "
        "and the champion. Roll again for another timeline.</p>"
        "<button class='btn' id='sim-btn'>🎲 Roll a World Cup</button>"
        "<div id='sim-out' style='margin-top:14px'></div>"
        "</div>"
        "<div class='tab' id='tab-insights'>"
        f"{review_html}"
        "<h2 class='col mcol'>🏅 Model vs FIFA ranking <span class='tag'>does the model agree?</span></h2>"
        "<div id='fifa'></div>"
        "<h2 class='col mcol'>🎯 Does it actually work? <span class='tag'>track record</span></h2>"
        f"{_mega_html(data.get('mega_backtest'))}"
        "<p class='note' style='margin-top:16px'>And zooming in on two tournaments the "
        "model had never seen when it was trained:</p>"
        "<div id='backtest'></div>"
        "<h2 class='col mcol'>👟 Golden Boot <span class='tag'>top scorer</span></h2><div id='golden'></div>"
        "<h2 class='col mcol'>⚽ Anatomy of a goal <span class='tag'>every international goal</span></h2>"
        "<div id='goals-an'></div>"
        "<h2>📉 Elo through history <span class='tag'>drag the year</span></h2>"
        "<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px'>"
        "<span class='note'>Year</span>"
        f"<input type='range' id='elo-year' min='{ely_min}' max='{ely_max}' "
        f"value='{ely_max}' step='1' style='flex:1'>"
        f"<span id='elo-year-lbl' style='font-weight:700;width:48px;text-align:right'>{ely_max}</span></div>"
        "<div style='position:relative;height:430px'><canvas id='elo-chart'></canvas></div>"
        f"{_facts_html(data.get('facts'))}"
        "</div>"
        f"{pred_tab}"
        "<p class='foot'>Data-driven model, just for fun. ⚽ "
        "Data: International football results 1872–2026.</p>"
        "</div>"
        f"<script>const DATA={json.dumps(data, ensure_ascii=False)};</script>"
        f"<script>{_JS}</script></body></html>"
    )
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path
