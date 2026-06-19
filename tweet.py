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

SITE = "https://manuelcralves.github.io/WORLD-CUP-2026/"
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


def _flag(team: str) -> str:
    return FLAGS.get(team, "")


def _surname(name: str) -> str:
    parts = str(name).split()
    return parts[-1] if parts else str(name)


def _eff_len(text: str) -> int:
    return len(text) - len(SITE) + URL_LEN if SITE in text else len(text)


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
                         f"{r.p_ko * 100:.0f}% to advance")
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
    d = pd.read_csv(p).sort_values("p_champion", ascending=False).head(5)
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


def evergreen_tweets() -> list:
    """(label, text) tweets for the kit — copy-paste only, never auto-posted."""
    out = [("Recruitment · post anytime", COME_PLAY)]
    g = _golden_text()
    if g:
        out.append(("Golden Boot · refreshed each build", g))
    out.append(("Bracket · for ~28 Jun (when knockouts open)", BRACKET))
    return out


def build_thread() -> list:
    today = _today_utc()
    tweets = list(_today_previews(today))
    recap = _recap_text()
    if recap:
        tweets.append(recap)
    tweets += _qualif_today(today)
    title = _title_text()
    if title:
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
    ever = "".join(_card(f"e{j}", html.escape(lab), t)
                   for j, (lab, t) in enumerate(evergreen, 1))
    ever_html = (f'<h2 class="sec">♻️ Evergreen &amp; on-demand</h2>'
                 f'<p class="sub">Post any of these whenever — not part of the daily '
                 f'thread, never auto-posted.</p>{ever}') if evergreen else ""
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
<h2 class="sec">📅 Daily thread <span style="font-weight:400;color:#5d6a85">· refreshed each build</span></h2>
{cards}
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
    if not tweets:
        print("Nothing to post (no data found).")
        return 0
    write_kit(tweets, evergreen_tweets())         # always — the free manual path
    for i, t in enumerate(tweets, 1):
        print(f"---- tweet {i}/{len(tweets)} ({_eff_len(t)}/{LIMIT} chars) ----\n{t}")
    auto = os.environ.get("AUTO_TWEET", "").lower() in ("1", "true", "yes")
    print(f"\n({len(tweets)} tweets · kit → outputs/tweets.html · "
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
