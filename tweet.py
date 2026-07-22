"""Build a daily World Cup 2026 briefing and (a) write a copy-ready page for
posting by hand, and (b) optionally auto-post it to X.

Content, from the freshly generated ``outputs/`` CSVs:
  • one preview per today's match — outcome odds (W/D/L) + the 3 likeliest scores
  • the latest results vs the model — winner called ✅ / exact score 🎯 / upset 😱
  • the tightest group's race to the knockouts
  • the live title odds + biggest mover

MANUAL by default (free): every run writes ``outputs/tweets.html`` (cards with
Copy + "Tweet ▸" buttons) and ``outputs/daily_tweets.txt``. Posting costs nothing.

AUTO-POST is opt-in: set the env var ``AUTO_TWEET=true`` (and the four X_* secrets)
to also publish, capped at ``MAX_TWEETS`` (default 3) to bound the per-post cost.

    python tweet.py --dry-run   # build + write the kit, print, never post
    python tweet.py --test      # post a single "hello world" (credential check)
"""
from __future__ import annotations

import html
import os
import sys
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

SITE = "https://worldcup2026ml.pt/"
REVIEW = SITE + "outputs/review.html"   # retrospective hub — the closing tweets point here
OUT = Path(__file__).resolve().parent / "outputs"
KEYS = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")
LIMIT = 280
URL_LEN = 23
COST = 0.20
MAX_TWEETS = int(os.environ.get("MAX_TWEETS") or 0)   # 0 = no cap (full thread)

INTRO = ("🤖⚽ Hello! This account posts daily machine-learning predictions for "
         "the 2026 World Cup — Elo + Poisson + Monte Carlo over 150 years of "
         f"football. Just for fun.\n\nLive dashboard 👇\n{SITE}")

# Evergreen tweets — recruitment + trackers that live in the copy-ready kit but
# are NEVER part of the auto-posted daily thread (so nothing floods on a schedule).
COME_PLAY = ("The World Cup 2026 prediction league is live 🏆\n\n"
             "Call the scores. Climb the table. Prove you know your football "
             "better than everyone else.\n\n"
             f"Come play 👉 {SITE}")
BRACKET = ("🏆 The World Cup 2026 knockouts are here.\n\n"
           "Fill in your full bracket — every tie from the Round of 32 to the "
           "final — and see if you can out-pick your friends.\n\n"
           f"Play your bracket 👉 {SITE}")

try:
    from wc2026.viz import FLAGS
except Exception:
    FLAGS = {}

# X counts each emoji as length 2, but Python's len() counts a multi-codepoint emoji
# by its code points — the England/Scotland/Wales "tag" flags are 7 each — which
# over-states the tweet length and needlessly drops ties that would really fit.
_LONG_FLAGS = [f for f in FLAGS.values() if len(f) > 2]


def _flag(team: str) -> str:
    return FLAGS.get(team, "")


def _surname(name: str) -> str:
    parts = str(name).split()
    return parts[-1] if parts else str(name)


def _pct(p) -> str:
    """Percent with no misleading bare '0%': a tiny but non-zero value shows '<1%'
    (a 0-count from the sim means 'below resolution', not truly impossible). An
    exact 0 (e.g. a team eliminated from its group) still shows '0%'."""
    v = round(float(p) * 100)
    if v == 0 and float(p) > 0:
        return "<1%"
    return f"{v}%"


def _eff_len(text: str) -> int:
    n = (len(text) - len(SITE) + URL_LEN) if SITE in text else len(text)
    for fl in _LONG_FLAGS:                     # discount the code-point surplus of tag flags
        if fl in text:
            n -= text.count(fl) * (len(fl) - 2)
    return n


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _days_old(iso: str) -> int:
    try:
        return (date.fromisoformat(_today_utc()) - date.fromisoformat(iso)).days
    except Exception:
        return 0


def _fit(lines, head, tail, sep="\n") -> str:
    body, n = "", 0
    for ln in lines:
        cand = body + (sep if body else "") + ln
        if _eff_len(head + cand + tail) > LIMIT:
            break
        body, n = cand, n + 1
    if 0 < n < len(lines):
        more = (sep if body else "") + f"…+{len(lines) - n} more on the site"
        if _eff_len(head + body + more + tail) <= LIMIT:
            body += more
    return body


# --------------------------------------------------------------------------- #
# Content builders
# --------------------------------------------------------------------------- #
def _match_preview(r) -> str:
    fl_h, fl_a = _flag(r.home) or "⚽", _flag(r.away) or "⚽"
    scores = " · ".join(f"{s.split(':')[0]} ({s.split(':')[1]}%)"
                        for s in str(r.top3).split(",")) if "top3" in r._fields else ""
    out = (f"⚽ {fl_h} {r.home} vs {r.away} {fl_a}\n\n"
           f"📊 {r.home} {r.p_home * 100:.0f}% · Draw {r.p_draw * 100:.0f}% · "
           f"{r.away} {r.p_away * 100:.0f}%")
    if scores:
        out += f"\n🎯 Likeliest scores: {scores}"
    return out


def _today_previews(today: str) -> list:
    gm = OUT / "group_matches.csv"
    if not gm.exists():
        return []
    d = pd.read_csv(gm)
    d = d[d["date"] == today]
    return [_match_preview(r) for r in d.itertuples(index=False)]


def _recap_text():
    p = OUT / "played_review.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).rename(columns={"as": "a_s"})
    if d.empty:
        return None
    last = sorted(d["date"].unique())[-1]
    if _days_old(last) > 2:
        return None
    day = d[d["date"] == last]
    hits, n = int(day["hit"].sum()), len(day)
    miss = day[(~day["hit"]) & (day["p_actual"] < 0.30)].sort_values("p_actual")
    upset_key = (miss.iloc[0]["home"], miss.iloc[0]["away"]) if not miss.empty else None
    lines = []
    for r in day.itertuples(index=False):
        actual = f"{r.hs}-{r.a_s}"
        if r.ml_score == actual:
            tag = "🎯 exact score!"
        elif r.hit:
            tag = f"✅ winner (we said {r.ml_score})"
        elif upset_key and (r.home, r.away) == upset_key:
            tag = f"😱 upset! (we gave it {r.p_actual * 100:.0f}%)"
        else:
            tag = f"❌ wrong (we said {r.ml_score})"
        lines.append(f"{_flag(r.home) or '⚽'} {r.home} {actual} {r.away} — {tag}")
    head = f"📊 How our model did — latest results ({hits}/{n} winners called):\n\n"
    return head + _fit(lines, head, "")


def _standings_from_review() -> dict:
    p = OUT / "played_review.csv"
    pts = {}
    if p.exists():
        d = pd.read_csv(p).rename(columns={"as": "a_s"})
        for r in d.itertuples(index=False):
            for team, gf, ga in ((r.home, r.hs, r.a_s), (r.away, r.a_s, r.hs)):
                s = pts.setdefault(team, {"played": 0, "pts": 0})
                s["played"] += 1
                s["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
    return pts


def _qualif_today(today: str) -> list:
    """One standings + advance-odds snapshot per group that plays today."""
    gm, pr = OUT / "group_matches.csv", OUT / "predictions.csv"
    if not gm.exists() or not pr.exists():
        return []
    groups = sorted(pd.read_csv(gm).query("date == @today")["group"].unique())
    if not groups:
        return []                                # e.g. a knockout day — skip
    d = pd.read_csv(pr)
    if "p_ko" not in d.columns or "group" not in d.columns:
        return []
    pts, out = _standings_from_review(), []
    for L in groups:
        g = d[d["group"] == L].sort_values("p_ko", ascending=False)
        lines = []
        for r in g.itertuples(index=False):
            s = pts.get(r.team, {"played": 0, "pts": 0})
            rec = f"{s['pts']}pt{'s' if s['pts'] != 1 else ''}" if s["played"] else "—"
            lines.append(f"{_flag(r.team) or '⚽'} {r.team} · {rec} · "
                         f"{_pct(r.p_ko)} to advance")
        head = f"🎟️ Group {L} today — standings & chance to reach the last 32:\n\n"
        out.append(head + _fit(lines, head, ""))
    return out


def _mover_line():
    try:
        h = pd.read_csv(OUT / "history.csv")
        dates = sorted(h["date"].unique())
        if len(dates) < 2:
            return None
        a = h[h["date"] == dates[-2]].set_index("team")["p_champion"]
        b = h[h["date"] == dates[-1]].set_index("team")["p_champion"]
        delta = (b - a).dropna()
        if delta.empty:
            return None
        team = delta.abs().idxmax()
        dv = float(delta[team]) * 100
        if abs(dv) < 1:
            return None
        return f"{'📈' if dv > 0 else '📉'} Biggest mover: {_flag(team)} {team} {dv:+.0f} pts"
    except Exception:
        return None


def _title_text():
    p = OUT / "predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).sort_values("p_champion", ascending=False)
    d = d[d["p_champion"] > 0].head(5)               # only teams still in it (drop eliminated 0%)
    lines = [f"{i}. {_flag(r.team)} {r.team} {r.p_champion * 100:.0f}%"
             for i, r in enumerate(d.itertuples(index=False), 1)]
    mover = _mover_line()
    head = "🏆 Title race — our live odds:\n\n"
    return head + "\n".join(lines) + (f"\n\n{mover}" if mover else "")


def _golden_text():
    p = OUT / "golden_boot.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if d.empty or "proj" not in d.columns:
        return None
    top = d.sort_values("proj", ascending=False).head(5)
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    lines = [f"{medals[i]} {getattr(r, 'flag', '') or _flag(r.team)} "
             f"{_surname(r.scorer)} {r.proj:.1f}"
             for i, r in enumerate(top.itertuples(index=False))]
    head = ("👟 Who wins the WC 2026 Golden Boot?\n\n"
            "The model's projection (goals by the final):\n")
    tail = f"\n\n👇 {SITE}"
    note = ""
    if "wc" in d.columns and d["wc"].max() and int(d["wc"].max()) >= 2:
        mx = int(d["wc"].max())
        names = [_surname(s) for s in d.loc[d["wc"] == mx, "scorer"].tolist()]
        if len(names) == 1:
            note = f"\n\n{names[0]} leads the scoring so far — {mx}."
        elif len(names) <= 3:
            note = f"\n\n{' & '.join(names)} lead the scoring so far — {mx} each."
    out = head + "\n".join(lines) + note + tail
    if _eff_len(out) > LIMIT:                 # too long → drop the scoring note
        out = head + "\n".join(lines) + tail
    return out


def _group_overviews() -> list:
    """One card per group still in play: points · win-group% / reach-last-32%.

    Generic (same shape for every group), copy-paste only. Self-gates to the
    group stage via group_matches.csv — a group with no remaining match drops
    out, and once the knockouts start the whole section disappears.
    """
    gm, pr = OUT / "group_matches.csv", OUT / "predictions.csv"
    if not gm.exists() or not pr.exists():
        return []
    rem = pd.read_csv(gm)
    if "group" not in rem.columns or rem.empty:
        return []
    groups = sorted(str(g) for g in rem["group"].dropna().unique()
                    if str(g) in list("ABCDEFGHIJKL"))
    d = pd.read_csv(pr)
    if not groups or not {"group", "p_1st", "p_ko"} <= set(d.columns):
        return []
    pts = _standings_from_review()
    out = []
    for L in groups:
        g = d[d["group"] == L].sort_values(["p_ko", "p_1st"], ascending=False)
        if g.empty:
            continue
        lines = []
        for r in g.itertuples(index=False):
            p = pts.get(r.team, {}).get("pts", 0)
            rec = f"{p} pt{'' if p == 1 else 's'}"
            lines.append(f"{_flag(r.team) or '⚽'} {r.team} — {rec} · "
                         f"{_pct(r.p_1st)} / {_pct(r.p_ko)}")
        head = f"📊 Group {L} — before the final round\nwin group / reach last 32\n\n"
        text = head + "\n".join(lines) + f"\n\n🔮 {SITE}"
        out.append((f"Group {L} overview · post any", text))
    return out


def _likely_opponents(rnd="Round of 32", n_teams=6) -> list:
    """One card per marquee team: its most likely opponents in the next KO round.

    Reads opponents.csv (exported by run_pipeline from the sim). Projected during
    the group stage; sharpens once the bracket is set. Copy-paste only.
    """
    op, pr = OUT / "opponents.csv", OUT / "predictions.csv"
    if not op.exists() or not pr.exists():
        return []
    o = pd.read_csv(op)
    if "round" not in o.columns:
        return []
    o = o[o["round"] == rnd]
    if o.empty:
        return []
    d = pd.read_csv(pr)
    picks = d.sort_values("p_champion", ascending=False)["team"].head(n_teams).tolist()
    if "Portugal" in set(d["team"]) and "Portugal" not in picks:
        picks.append("Portugal")
    out = []
    for tm in picks:
        g = o[o["team"] == tm].sort_values("p_cond", ascending=False).head(4)
        if g.empty:
            continue
        lines = [f"{_flag(r.opponent) or '⚽'} {r.opponent} {_pct(r.p_cond)}"
                 for r in g.itertuples(index=False)]
        head = (f"🔮 Who will {_flag(tm) or '⚽'} {tm} face in the {rnd}?\n\n"
                f"The model's likeliest opponents:\n")
        text = head + "\n".join(lines) + f"\n\n👇 {SITE}"
        out.append((f"{tm} · likely {rnd} opponent", text))
    return out


def _knockout_previews(today: str) -> list:
    """Win-or-go-home previews for today's knockout ties (knockout_matches.csv)."""
    p = OUT / "knockout_matches.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    d = d[d["date"] == today]
    return ["🏆 Knockout — win or go home.\n\n" + _match_preview(r)
            for r in d.itertuples(index=False)]


_KO_ORDER = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
_KO_PREV = {"Round of 16": "Round of 32 done", "Quarter-finals": "Round of 16 done",
            "Semi-finals": "Quarter-finals done", "Final": "Semis done"}
# Current round from how many knockout ties are already PLAYED — robust to a mid-round
# state where some R16 are done AND a QF is already on the board (counting upcoming ties
# would mix rounds). R32=16 ties, R16=8, QF=4, SF=2, Final=1 -> cumulative done 16/24/28/30/31.
_KO_BOUNDS = [(16, "Round of 32"), (24, "Round of 16"), (28, "Quarter-finals"),
              (30, "Semi-finals"), (31, "Final")]
# A round is "just drawn" (a fresh reveal moment) when the done-count sits EXACTLY on a
# boundary — 0 R32, 16 R16, 24 QF, 28 SF, 30 Final — i.e. none of it has been played yet.
_FRESH_ROUND = {0: "Round of 32", 16: "Round of 16", 24: "Quarter-finals",
                28: "Semi-finals", 30: "Final"}


def _ko_played_count() -> int:
    """How many knockout ties have already been played (from played_review.csv)."""
    p = OUT / "played_review.csv"
    if not p.exists():
        return 0
    try:
        d = pd.read_csv(p)
        return int((d["stage"] == "knockout").sum()) if "stage" in d.columns else 0
    except Exception:
        return 0


def _in_knockouts() -> bool:
    p = OUT / "knockout_matches.csv"
    try:
        return p.exists() and not pd.read_csv(p).empty
    except Exception:
        return False


def _current_ko_round() -> str:
    """The knockout round currently in progress, from how many KO ties are done.
    '' during the group stage (and once the trophy is lifted)."""
    if not _in_knockouts() and _ko_played_count() == 0:
        return ""
    n = _ko_played_count()
    for bound, name in _KO_BOUNDS:
        if n < bound:
            return name
    return ""


def _knockout_reveal() -> str:
    """One-off 'the <round> is SET' card — fires ONLY right when a round is drawn (the
    KO played-count is exactly on a round boundary), so it never shows a stale, half-
    played round. At that moment knockout_matches.csv holds exactly that round's ties
    (no next-round fixture exists until a game is played), so no filtering is needed."""
    p = OUT / "knockout_matches.csv"
    if not p.exists():
        return ""
    d = pd.read_csv(p)
    if d.empty:
        return ""
    rnd = _FRESH_ROUND.get(_ko_played_count())
    if not rnd:                                     # mid-round -> not a reveal moment
        return ""
    if rnd == "Final":
        r = d.iloc[-1]   # last-dated tie = the final; an earlier row here is the 3rd-place play-off
        return (f"🏆 It all comes down to this — the FINAL is SET!\n\n"
                f"{_flag(r['home']) or '⚽'} {r['home']} v {r['away']} "
                f"{_flag(r['away']) or '⚽'}\n\nWho lifts the trophy? 👇\n{SITE}")
    lines = [f"{_flag(r.home) or '⚽'} {r.home} v {r.away} {_flag(r.away) or '⚽'}"
             for r in d.itertuples(index=False)]
    verb = "are" if rnd in ("Quarter-finals", "Semi-finals") else "is"
    prev = _KO_PREV.get(rnd)
    head = (f"🏆 {prev} — the {rnd} {verb} SET!\n\n" if prev
            else f"🏆 Groups done — the {rnd} {verb} SET!\n\n")
    tail = f"\n\nFull bracket 👇\n{SITE}"
    return head + _fit(lines, head, tail) + tail


def _path_to_final(teams=("Argentina", "Spain", "France", "Portugal", "Brazil"), n=4) -> list:
    """Per marquee team: its likeliest opponents on the road still ahead — only the
    rounds still to play, so once we're in the R16 it skips the finished R32."""
    p = OUT / "opponents.csv"
    if not p.exists():
        return []
    o = pd.read_csv(p)
    if o.empty or "round" not in o.columns:
        return []
    short = {"Round of 32": "R32", "Round of 16": "R16", "Quarter-finals": "QF",
             "Semi-finals": "SF", "Final": "Final"}
    cur = _current_ko_round()
    active = _KO_ORDER[_KO_ORDER.index(cur):] if cur in _KO_ORDER else _KO_ORDER
    out = []
    for tm in teams[:n]:
        g = o[o["team"] == tm]
        if g.empty:
            continue
        lines = []
        for rnd in active:
            gr = g[g["round"] == rnd].sort_values("p_cond", ascending=False)
            if not gr.empty:
                r = gr.iloc[0]
                lines.append(f"{short[rnd]}: {_flag(r['opponent']) or '⚽'} {r['opponent']} ({_pct(r['p_cond'])})")
        if lines:
            head = f"🛣️ {_flag(tm) or '⚽'} {tm} — likeliest road to the final:\n\n"
            out.append((f"{tm} · path to the final",
                        head + "\n".join(lines) + f"\n\n👇 {SITE}"))
    return out


def _survival_text() -> str:
    """During the knockouts: how many teams are left + the surviving title odds."""
    p = OUT / "predictions.csv"
    if (not _in_knockouts() and not _ko_played_count()) or not p.exists():   # group stage -> title-race tweet covers it
        return ""
    n_alive = 32 - _ko_played_count()                # each knockout tie eliminates one team
    if n_alive < 2:                                  # trophy already lifted
        return ""
    d = pd.read_csv(p).sort_values("p_champion", ascending=False)
    d = d[d["p_champion"] > 0]                        # only teams still alive (drop eliminated 0%)
    head = f"⚔️ {n_alive} teams left — who lifts the trophy?\n\n"
    lines = [f"{i}. {_flag(r.team)} {r.team} {_pct(r.p_champion)}"
             for i, r in enumerate(d.head(6).itertuples(index=False), 1)]
    return head + "\n".join(lines) + f"\n\n👇 {SITE}"


def _champion():
    """(team, flag) once the title is settled (a team at ~100% p_champion), else None."""
    p = OUT / "predictions.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
        w = d[d["p_champion"] >= 0.999]
        if len(w):
            t = str(w.iloc[0]["team"])
            return (t, _flag(t))
    except Exception:
        pass
    return None


def _model_hits():
    """(correct results, total played) from played_review.csv, else None."""
    p = OUT / "played_review.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
        return (int(d["hit"].sum()), len(d))
    except Exception:
        return None


def _champion_tweet(champ) -> str:
    team, fl = champ
    return (f"🏆 {fl} {team} are your FIFA World Cup 2026 champions!\n\n"
            f"And our blind model — trained before a ball was kicked — had them lifting "
            f"the trophy. 🤖\n\nRelive the whole tournament 👇\n{SITE}")


def _reportcard_tweet(champ) -> str:
    team, fl = champ
    hits = _model_hits()
    res = f"📊 {hits[0]}/{hits[1]} match results called right\n" if hits else ""
    return (f"🤖 Our model's blind, pre-tournament call vs reality:\n\n"
            f"🏆 Champion: {fl} {team} — called ✅\n"
            f"{res}\n"
            f"The full report card 👇\n{REVIEW}")


def _golden_final_tweet() -> str:
    p = OUT / "golden_boot.csv"
    if not p.exists():
        return ""
    try:
        d = pd.read_csv(p)
        if "wc" not in d.columns or not int(d["wc"].max() or 0):
            return ""
        top = d.sort_values("wc", ascending=False).iloc[0]
        n = int(top["wc"])
        if n < 1:
            return ""
        team = str(top["team"])
        return (f"👟 The FIFA World Cup 2026 Golden Boot:\n\n"
                f"🥇 {_flag(team)} {_surname(top['scorer'])} — {n} goals\n\n"
                f"See the full scoring race 👇\n{REVIEW}")
    except Exception:
        return ""


def _farewell_tweet() -> str:
    return ("👋 That's a wrap on FIFA World Cup 2026.\n\n"
            "Thanks to everyone who played Beat the Machine and filled in a bracket 🙌\n\n"
            "The full retrospective is live — the numbers, every team's journey & how the "
            f"model did 👇\n{REVIEW}")


def evergreen_tweets() -> list:
    """(label, text) tweets for the kit — copy-paste only, never auto-posted."""
    champ = _champion()
    if champ:                                 # title settled -> a clean CLOSING set, not the live 'come play' stuff
        out = [("🏆 Champions", _champion_tweet(champ)),
               ("🤖 Model report card", _reportcard_tweet(champ))]
        gb = _golden_final_tweet()
        if gb:
            out.append(("👟 Golden Boot (final)", gb))
        out.append(("👋 Farewell & retrospective", _farewell_tweet()))
        return out
    out = [("Recruitment · post anytime", COME_PLAY)]
    g = _golden_text()
    if g:
        out.append(("Golden Boot · refreshed each build", g))
    out += _group_overviews()                 # one card per group still in play (group stage only)
    reveal = _knockout_reveal()               # 'the <round> is set' — round-aware (R32/R16/QF/SF/Final)
    if reveal:
        out.append((f"Knockouts · the {_current_ko_round() or 'next round'} is set", reveal))
    if not _current_ko_round():               # only before the bracket is drawn (group stage): once the
        out += _likely_opponents()            # ties are set the opponent is known, not "likely"
    out += _path_to_final()                   # marquee teams' road still ahead, round by round
    surv = _survival_text()                   # surviving teams' title odds (knockouts)
    if surv:
        out.append(("Survival odds · who wins it all", surv))
    out.append(("Bracket · knockouts are live", BRACKET))
    return out


def build_thread() -> list:
    today = _today_utc()
    tweets = list(_today_previews(today)) + list(_knockout_previews(today))   # group + knockout games today
    recap = _recap_text()
    if recap:
        tweets.append(recap)
    tweets += _qualif_today(today)
    title = _title_text()
    if title and not _champion():             # once the title is settled, drop the now-pointless 'title race'
        tweets.append(title)
    if not tweets:
        return []
    cta = (f"🔗 Full bracket, odds & Match Lab:\n{SITE}"
           "\n\n#WorldCup2026 #WorldCup #Football")
    if _eff_len(tweets[-1] + "\n\n" + cta) <= LIMIT:
        tweets[-1] += "\n\n" + cta
    else:
        tweets.append(cta)
    return tweets


# --------------------------------------------------------------------------- #
# Manual posting kit: a copy-ready page (+ plain text)
# --------------------------------------------------------------------------- #
def write_kit(tweets: list, evergreen: list = ()):
    if not OUT.exists():
        return
    (OUT / "daily_tweets.txt").write_text(
        ("\n\n" + "—" * 24 + "\n\n").join(list(tweets) + [t for _, t in evergreen]),
        encoding="utf-8")

    def _card(cid, meta, t):
        intent = "https://twitter.com/intent/tweet?text=" + urllib.parse.quote(t)
        return (
            f'<div class="card"><div class="meta"><span>{meta}</span>'
            f'<span class="cc">{_eff_len(t)}/280</span></div>'
            f'<pre id="{cid}">{html.escape(t)}</pre>'
            f'<div class="btns"><button onclick="cp(\'{cid}\')">📋 Copy</button>'
            f'<a class="tw" href="{intent}" target="_blank" rel="noopener">Tweet ▸</a>'
            f'</div></div>')

    cards = "".join(_card(f"t{i}", f"Tweet {i}/{len(tweets)}", t)
                    for i, t in enumerate(tweets, 1))
    daily_html = (f'<h2 class="sec">📅 Daily thread <span style="font-weight:400;color:#5d6a85">'
                  f'· refreshed each build</span></h2>{cards}') if tweets else ""
    ever = "".join(_card(f"e{j}", html.escape(lab), t)
                   for j, (lab, t) in enumerate(evergreen, 1))
    # once the tournament is over there is no daily thread, so the closing set is the whole kit
    ever_title = "🏆 Closing tweets — the tournament's over" if not tweets else "♻️ Evergreen &amp; on-demand"
    ever_sub = ("Post these to wrap up the World Cup — the champion, the model's report card & a farewell."
                if not tweets else
                "Post any of these whenever — not part of the daily thread, never auto-posted.")
    ever_html = (f'<h2 class="sec">{ever_title}</h2><p class="sub">{ever_sub}</p>{ever}') if evergreen else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Today's tweets · World Cup 2026</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0c1018;color:#eef1f7;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;
padding:34px 16px 60px;max-width:640px;margin:0 auto}}
h1{{font-size:21px;margin:0 0 4px}}.sub{{color:#8b95ab;font-size:14px;margin:0 0 22px}}
h2.sec{{font-size:15px;color:#cbd3e1;margin:30px 0 8px;border-top:1px solid #243049;padding-top:20px}}
a.home{{color:#00e0a4;text-decoration:none;font-size:13px}}
.card{{background:#161d2b;border:1px solid #243049;border-radius:14px;padding:14px 16px;margin:0 0 16px}}
.meta{{display:flex;justify-content:space-between;gap:10px;color:#8b95ab;font-size:12px;font-weight:700;margin-bottom:8px}}
pre{{white-space:pre-wrap;word-wrap:break-word;font-family:inherit;font-size:15px;line-height:1.5;margin:0 0 12px}}
.btns{{display:flex;gap:10px}}
button,.tw{{border:0;border-radius:999px;padding:8px 16px;font-size:14px;font-weight:700;cursor:pointer;text-decoration:none}}
button{{background:#243049;color:#eef1f7}}button:hover{{background:#2c3a57}}
.tw{{background:#00e0a4;color:#062018}}.tw:hover{{filter:brightness(1.08)}}
.ok{{background:#00e0a4!important;color:#062018!important}}
</style></head><body>
<a class="home" href="../index.html">← Home</a>
<h1>📋 World Cup 2026 — tweet kit</h1>
<p class="sub">Tap <b>Copy</b> and paste into X, or <b>Tweet ▸</b> to open X with it ready.
Post each one with Tweet ▸ as its own tweet (shows better than a thread on a new account), or reply to chain them.</p>
{daily_html}
{ever_html}
<script>function cp(id){{var el=document.getElementById(id);
navigator.clipboard.writeText(el.innerText);var b=el.parentNode.querySelector('button');
var o=b.innerText;b.innerText='✓ Copied';b.classList.add('ok');
setTimeout(function(){{b.innerText=o;b.classList.remove('ok');}},1400);}}</script>
</body></html>"""
    (OUT / "tweets.html").write_text(page, encoding="utf-8")


def post_thread(tweets: list) -> int:
    import tweepy
    c = tweepy.Client(
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_SECRET"],
    )
    prev, posted = None, 0
    for t in tweets:
        kw = {"text": t}
        if prev:
            kw["in_reply_to_tweet_id"] = prev
        prev = c.create_tweet(**kw).data["id"]
        posted += 1
    return posted


def main() -> int:
    dry = "--dry-run" in sys.argv
    if "--test" in sys.argv:                      # one-off credential check
        print("---- test tweet ----\n" + INTRO)
        if dry:
            print("(dry-run)")
            return 0
        missing = [k for k in KEYS if not os.environ.get(k)]
        if missing:
            print(f"Missing credentials {missing}.")
            return 1
        try:
            post_thread([INTRO])
            print("Posted ✓")
        except Exception as e:
            print("Post failed:", type(e).__name__, e)
            return 1
        return 0

    tweets = build_thread()
    evergreen = evergreen_tweets()
    if not tweets and not evergreen:
        print("Nothing to post (no data found).")
        return 0
    write_kit(tweets, evergreen)                  # always — the free manual path
    for i, t in enumerate(tweets, 1):
        print(f"---- tweet {i}/{len(tweets)} ({_eff_len(t)}/{LIMIT} chars) ----\n{t}")
    for lab, t in evergreen:
        print(f"---- {lab} ({_eff_len(t)}/{LIMIT} chars) ----\n{t}")
    auto = os.environ.get("AUTO_TWEET", "").lower() in ("1", "true", "yes")
    print(f"\n({len(tweets)} daily + {len(evergreen)} evergreen · kit → outputs/tweets.html · "
          f"auto-post {'ON' if auto else 'OFF'})")
    if dry:
        return 0
    if not auto:
        print("Manual mode — not posting (open tweets.html to post by hand).")
        return 0
    missing = [k for k in KEYS if not os.environ.get(k)]
    if missing:
        print(f"Missing credentials {missing}; skipping post.")
        return 0
    batch = tweets[:MAX_TWEETS] if MAX_TWEETS else tweets
    try:
        print(f"Auto-posting {len(batch)} (~${len(batch) * COST:.2f})…")
        print(f"Posted {post_thread(batch)} ✓")
    except Exception as e:
        print("Post failed:", type(e).__name__, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
