"""Post a daily World Cup 2026 update to X/Twitter.

Reads the freshly generated ``outputs/`` CSVs and posts EITHER today's fixtures
with the model's pick, OR (on days with no matches) the current title odds plus
the biggest mover. Credentials come from four environment variables, set as
GitHub Secrets:

    X_API_KEY  X_API_SECRET  X_ACCESS_TOKEN  X_ACCESS_SECRET

If any is missing the script prints what it WOULD post and exits 0, so it is safe
to wire into CI before the keys exist. Preview locally without posting:

    python tweet.py --dry-run
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SITE = "https://manuelcralves.github.io/WORLD-CUP-2026/"
OUT = Path(__file__).resolve().parent / "outputs"
KEYS = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")
LIMIT = 280
URL_LEN = 23          # t.co shortens every link to a fixed 23 characters

# Emoji flags render fine on X (unlike a Windows console). Optional import.
try:
    from wc2026.viz import FLAGS
except Exception:
    FLAGS = {}


def _flag(team: str) -> str:
    return FLAGS.get(team, "")


def _eff_len(text: str) -> int:
    """Tweet length as X counts it (the URL always weighs 23 chars)."""
    return len(text) - len(SITE) + URL_LEN if SITE in text else len(text)


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _matchday_text(today: str) -> str | None:
    """Today's fixtures with the model's most-likely scoreline, if any."""
    gm = OUT / "group_matches.csv"
    if not gm.exists():
        return None
    d = pd.read_csv(gm)
    d = d[d["date"] == today]
    if d.empty:
        return None
    lines = [f"{_flag(r.home) or '⚽'} {r.home}–{r.away} → {r.ml_score}"
             for r in d.itertuples(index=False)]
    head = "🏆 World Cup 2026 — today's matches & the model's call:\n\n"
    tail = f"\n\nFull odds & bracket 👇\n{SITE}"
    body, shown = "", 0
    for ln in lines:
        if _eff_len(head + body + ln + "\n" + tail) > LIMIT:
            break
        body += ln + "\n"
        shown += 1
    if shown == 0:
        return None
    if shown < len(lines):
        extra = f"…+{len(lines) - shown} more\n"
        if _eff_len(head + body + extra + tail) <= LIMIT:
            body += extra
    return head + body.rstrip("\n") + tail


def _mover_line() -> str | None:
    """Biggest title-odds change between the two latest snapshots."""
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


def _odds_text() -> str | None:
    """Top-5 title odds plus the biggest mover (the no-matches fallback)."""
    p = OUT / "predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).sort_values("p_champion", ascending=False).head(5)
    head = "🏆 World Cup 2026 — title odds (live ML model):\n\n"
    lines = [f"{i}. {_flag(r.team)} {r.team} {r.p_champion * 100:.0f}%"
             for i, r in enumerate(d.itertuples(index=False), 1)]
    mover = _mover_line()
    tail = (f"\n\n{mover}" if mover else "") + f"\n\nFull dashboard 👇\n{SITE}"
    return head + "\n".join(lines) + tail


def build_text() -> str | None:
    return _matchday_text(_today_utc()) or _odds_text()


def post(text: str):
    import tweepy
    c = tweepy.Client(
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_SECRET"],
    )
    return c.create_tweet(text=text)


def main() -> int:
    dry = "--dry-run" in sys.argv
    text = build_text()
    if not text:
        print("Nothing to post (no data found).")
        return 0
    print("---- tweet ----\n" + text + f"\n---- {_eff_len(text)}/{LIMIT} chars ----")
    missing = [k for k in KEYS if not os.environ.get(k)]
    if dry:
        print("(dry-run: not posting)")
        return 0
    if missing:
        print(f"Missing credentials {missing}; skipping post (no-op).")
        return 0
    try:
        post(text)
        print("Posted ✓")
    except Exception as e:                       # never fail the daily build
        print("Post failed (not failing the build):", type(e).__name__, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
