"""Post a daily World Cup 2026 briefing to X/Twitter, as a short thread.

Reads the freshly generated ``outputs/`` CSVs and builds up to MAX_TWEETS posts:
  1. today's fixtures with the model's pick
  2. the latest results vs the model, with the biggest upset highlighted
  3. the live title race (top-5 + biggest mover)
  4. the Golden Boot favourite
They are posted as a reply-chain (thread). Credentials come from four
environment variables, set as GitHub Secrets:

    X_API_KEY  X_API_SECRET  X_ACCESS_TOKEN  X_ACCESS_SECRET

If any is missing the script prints what it WOULD post and exits 0, so it is safe
to wire into CI before the keys exist. Useful flags:

    python tweet.py --dry-run      # preview the thread without posting
    python tweet.py --test         # post a single "hello world" (credential check)

MAX_TWEETS (env var, default 3) caps the thread length — each post costs the same,
so this is the cost knob.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

SITE = "https://manuelcralves.github.io/WORLD-CUP-2026/"
OUT = Path(__file__).resolve().parent / "outputs"
KEYS = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")
LIMIT = 280
URL_LEN = 23                       # t.co shortens every link to a fixed 23 chars
COST = 0.20                        # ~USD per post (pay-per-use), for the estimate
MAX_TWEETS = int(os.environ.get("MAX_TWEETS") or 3)

# One-off "hello world" used by `--test` to verify the credentials (and a nice
# first/pinned post for the account).
INTRO = ("🤖⚽ Hello! This account posts daily machine-learning predictions for "
         "the 2026 World Cup — Elo + Poisson + Monte Carlo over 150 years of "
         f"football. Just for fun.\n\nLive dashboard 👇\n{SITE}")

# Emoji flags render fine on X (unlike a Windows console). Optional import.
try:
    from wc2026.viz import FLAGS
except Exception:
    FLAGS = {}


def _flag(team: str) -> str:
    return FLAGS.get(team, "")


def _eff_len(text: str) -> int:
    """Tweet length as X counts it (every link weighs a flat 23 chars)."""
    return len(text) - len(SITE) + URL_LEN if SITE in text else len(text)


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _fit(lines, head, tail, sep="\n") -> str:
    """Join as many lines as fit within LIMIT (counting head + tail)."""
    body, n = "", 0
    for ln in lines:
        cand = body + (sep if body else "") + ln
        if _eff_len(head + cand + tail) > LIMIT:
            break
        body, n = cand, n + 1
    if 0 < n < len(lines):
        more = (sep if body else "") + f"…+{len(lines) - n} more"
        if _eff_len(head + body + more + tail) <= LIMIT:
            body += more
    return body


# --------------------------------------------------------------------------- #
# Content builders — each returns a tweet body (no link), or None.
# --------------------------------------------------------------------------- #
def _today_text(today: str):
    gm = OUT / "group_matches.csv"
    if not gm.exists():
        return None
    d = pd.read_csv(gm)
    d = d[d["date"] == today]
    if d.empty:
        return None
    lines = [f"{_flag(r.home) or '⚽'} {r.home}–{r.away} → {r.ml_score}"
             for r in d.itertuples(index=False)]
    head = "⚽ Matchday! Today's games & our model's call:\n\n"
    return head + _fit(lines, head, "")


def _recap_text():
    """The latest match day's results vs the model, with the biggest upset."""
    p = OUT / "played_review.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).rename(columns={"as": "a_s"})
    if d.empty:
        return None
    last = sorted(d["date"].unique())[-1]
    try:                                       # skip stale recaps on rest days
        if (date.fromisoformat(_today_utc()) - date.fromisoformat(last)).days > 2:
            return None
    except Exception:
        pass
    day = d[d["date"] == last]
    hits, n = int(day["hit"].sum()), len(day)
    lines = [f"{_flag(r.home) or '⚽'} {r.home} {r.hs}-{r.a_s} {r.away} "
             f"{'✅' if r.hit else '❌'}" for r in day.itertuples(index=False)]
    miss = day[(~day["hit"]) & (day["p_actual"] < 0.30)].sort_values("p_actual")
    upset = ""
    if not miss.empty:
        u = miss.iloc[0]
        who = (u["home"] if u["hs"] > u["a_s"]
               else u["away"] if u["a_s"] > u["hs"] else "the draw")
        upset = f"\n\n😱 Upset: {who} — our model gave it just {u['p_actual'] * 100:.0f}%"
    head = f"📊 Latest results vs our model ({hits}/{n} called right):\n\n"
    return head + _fit(lines, head, upset) + upset


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
    if d.empty:
        return None
    r = d.iloc[0]
    fl = r["flag"] if "flag" in d.columns and pd.notna(r["flag"]) else _flag(r["team"])
    return (f"👟 Golden Boot favourite: {fl} {r['scorer']} ({r['team']}) — "
            f"~{float(r['proj']):.0f} goals projected.")


def build_thread() -> list:
    """The day's briefing: a few varied tweets, capped at MAX_TWEETS."""
    today = _today_utc()
    cands = [c for c in (_today_text(today), _recap_text(),
                         _title_text(), _golden_text()) if c]
    cands = cands[:MAX_TWEETS]
    if not cands:
        return []
    cta = f"🔗 Live bracket, odds & Match Lab:\n{SITE}"
    if _eff_len(cands[-1] + "\n\n" + cta) <= LIMIT:
        cands[-1] += "\n\n" + cta
    elif len(cands) < MAX_TWEETS:
        cands.append(cta)
    return cands


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
    test = "--test" in sys.argv
    tweets = [INTRO] if test else build_thread()
    if not tweets:
        print("Nothing to post (no data found).")
        return 0
    for i, t in enumerate(tweets, 1):
        print(f"---- tweet {i}/{len(tweets)} ({_eff_len(t)}/{LIMIT} chars) ----\n{t}")
    print(f"---- thread of {len(tweets)} ≈ ${len(tweets) * COST:.2f} ----")
    missing = [k for k in KEYS if not os.environ.get(k)]
    if dry:
        print("(dry-run: not posting)")
        return 0
    if missing:
        print(f"Missing credentials {missing}; skipping post (no-op).")
        return 1 if test else 0
    try:
        n = post_thread(tweets)
        print(f"Posted {n} ✓")
    except Exception as e:
        print("Post failed:", type(e).__name__, e)
        return 1 if test else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
