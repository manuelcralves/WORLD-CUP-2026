"""FIFA World Cup 2026 rich-data fetcher -- Highlightly football API.

Pulls matches, line-ups and events (goals / cards / substitutions) for the World
Cup and caches them to CSVs under api_cache/, so re-runs only fetch what's NEW and
stay inside the free 100-requests/day BASIC limit.

The key is read from HIGHLIGHTLY_KEY -- never hard-coded, never committed.

    # PowerShell:
    $env:HIGHLIGHTLY_KEY = "your-key-here"
    python fetch_wc_data.py

If you registered through RapidAPI (not directly on highlightly.net):
    $env:HIGHLIGHTLY_RAPIDAPI = "1"
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

KEY = os.environ.get("HIGHLIGHTLY_KEY")
RAPID = os.environ.get("HIGHLIGHTLY_RAPIDAPI") == "1"
BASE = ("https://football-highlights-api.p.rapidapi.com" if RAPID
        else "https://soccer.highlightly.net")
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")    # Cloudflare 1010-blocks the python UA
HEADERS = {"x-rapidapi-key": KEY or "", "User-Agent": _UA}       # direct API uses x-rapidapi-key too
if RAPID:
    HEADERS["x-rapidapi-host"] = "football-highlights-api.p.rapidapi.com"

LEAGUE, SEASON = 1635, 2026          # 1635 = FIFA World Cup
CACHE = Path(__file__).resolve().parent / "api_cache"
DAILY_CAP = 90                       # TRUE cumulative cap across ALL of today's runs (under the 100/day free limit)
QUOTA_FILE = CACHE / ".quota"        # "<UTC-date> <count>", persisted in the committed cache so it survives across builds

_used = 0                            # requests made this run
_prior = 0                           # requests already spent today by earlier runs (loaded from .quota)
_today = ""


def _get(path: str, **params):
    """One API call. Counts against the cumulative daily cap; returns None on an HTTP error."""
    global _used
    if _prior + _used >= DAILY_CAP:
        print(f"\n[!] Reached today's cap ({DAILY_CAP} requests across all runs). Stops here; "
              f"resumes tomorrow -- the cache picks up exactly where it left off.")
        _save_quota()
        raise SystemExit(0)
    url = f"{BASE}/{path}" + (("?" + urllib.parse.urlencode(params)) if params else "")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        _used += 1
        print(f"  [!] {path} -> HTTP {e.code}")
        return None
    _used += 1
    return payload


def _load_quota() -> None:
    """Load today's request count from .quota (resets when the UTC date rolls over)."""
    global _prior, _today
    from datetime import datetime, timezone
    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        d, n = QUOTA_FILE.read_text().split()
        _prior = int(n) if d == _today else 0
    except (FileNotFoundError, ValueError):
        _prior = 0


def _save_quota() -> None:
    try:
        QUOTA_FILE.write_text(f"{_today} {_prior + _used}")
    except OSError:
        pass


def _load_ids(path: Path, col: str) -> set:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {row[col] for row in csv.DictReader(f)}


def _append(path: Path, rows: list, header: list) -> None:
    if not rows:
        return
    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if fresh:
            w.writeheader()
        w.writerows(rows)


EV_HEADER = ["match_id", "minute", "team", "type", "player", "player_id", "assist", "out", "out_pid"]


def _ev_rows(mid, evs) -> list:
    """Event rows for one match. out_pid (assistingPlayerId) = the player going off on a sub."""
    return [{"match_id": mid, "minute": e.get("time"), "team": (e.get("team") or {}).get("name"),
             "type": e.get("type"), "player": e.get("player"), "player_id": e.get("playerId"),
             "assist": e.get("assist"), "out": e.get("substituted"),
             "out_pid": e.get("assistingPlayerId")} for e in (evs or [])]


def fetch_matches() -> list:
    """All World Cup matches, following the limit/offset pagination."""
    out, offset = [], 0
    while True:
        resp = _get("matches", leagueId=LEAGUE, season=SEASON, limit=100, offset=offset)
        data = resp.get("data", []) if isinstance(resp, dict) else []
        out.extend(data)
        total = (resp.get("pagination") or {}).get("totalCount", len(out)) if isinstance(resp, dict) else len(out)
        offset += 100
        if not data or offset >= total:
            break
    return out


def main() -> None:
    if not KEY:
        sys.exit('[x] Set HIGHLIGHTLY_KEY first ($env:HIGHLIGHTLY_KEY = "...").')
    CACHE.mkdir(exist_ok=True)
    _load_quota()
    print(f"quota: {_prior}/{DAILY_CAP} requests already used today (UTC)")
    m_csv, ln_csv, ev_csv = CACHE / "wc_matches.csv", CACHE / "wc_lineups.csv", CACHE / "wc_events.csv"
    st_csv = CACHE / "wc_stats.csv"

    # 1) matches -- rewritten every run (cheap, keeps scores fresh)
    rows = []
    for m in fetch_matches():
        score = ((m.get("state") or {}).get("score") or {}).get("current")
        rows.append({"match_id": m["id"], "date": (m.get("date") or "")[:10], "round": m.get("round"),
                     "status": (m.get("state") or {}).get("description"),
                     "home": m["homeTeam"]["name"], "home_id": m["homeTeam"]["id"],
                     "away": m["awayTeam"]["name"], "away_id": m["awayTeam"]["id"], "score": score})
    if rows:                              # only overwrite the cache when the fetch returned matches --
        if m_csv.exists():                # never wipe it on a quota/Cloudflare miss (that blanks the whole site)
            m_csv.unlink()
        _append(m_csv, rows, ["match_id", "date", "round", "status",
                              "home", "home_id", "away", "away_id", "score"])
    else:
        print("matches fetch returned nothing (quota/Cloudflare) -> keeping the existing cache")
    print(f"matches: {len(rows)} saved ({sum(1 for r in rows if r['score'])} played)")

    # 2) line-ups + events for FINISHED matches, one request each, cached
    finished = [r for r in rows if r["score"]]
    # one-time: drop a pre-out_pid events cache so every game re-fetches with the sub-on id.
    # Gated on `rows` (the matches fetch worked) so we never wipe events when the API is
    # unreachable (Cloudflare) and couldn't refill them.
    # one-time backfill: if the cached events predate out_pid, re-fetch them all WITH the new
    # field -- atomically: build the full set in memory and only swap the file in if EVERY game
    # succeeded, so a low-quota / Cloudflare-blocked run never leaves a half-empty events cache.
    if rows and ev_csv.exists():
        with ev_csv.open(encoding="utf-8") as f:
            stale = "out_pid" not in f.readline()
        if stale:
            print("events cache predates out_pid -> re-fetching all events (atomic)")
            fresh, ok = [], True
            for r in finished:
                evs = _get(f"events/{r['match_id']}")
                if evs is None:                  # API error / quota -> abort, keep the old cache intact
                    ok = False
                    break
                fresh += _ev_rows(r["match_id"], evs)
            if ok:
                ev_csv.unlink()
                _append(ev_csv, fresh, EV_HEADER)
                print(f"  re-fetched events for {len(finished)} games")
            else:
                print("  re-fetch incomplete (quota/API) -> kept the existing events cache")
    ln_todo = [r for r in finished if str(r["match_id"]) not in _load_ids(ln_csv, "match_id")]
    ev_todo = [r for r in finished if str(r["match_id"]) not in _load_ids(ev_csv, "match_id")]
    print(f"line-ups: fetching {len(ln_todo)} new | events: fetching {len(ev_todo)} new")

    for r in ln_todo:
        mid = r["match_id"]
        ln = _get(f"lineups/{mid}")
        out = []
        for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
            t = (ln or {}).get(key) or {}
            starters = [p for line in (t.get("initialLineup") or []) for p in line]
            for tag, group in (("yes", starters), ("no", t.get("substitutes") or [])):
                out += [{"match_id": mid, "side": side, "team": r[side], "formation": t.get("formation"),
                         "starter": tag, "player_id": p.get("id"), "player": p.get("name"),
                         "number": p.get("number"), "position": p.get("position")} for p in group]
        _append(ln_csv, out, ["match_id", "side", "team", "formation", "starter",
                              "player_id", "player", "number", "position"])

    for r in ev_todo:
        _append(ev_csv, _ev_rows(r["match_id"], _get(f"events/{r['match_id']}")), EV_HEADER)

    # 3) match statistics (possession, shots, ...) for finished matches, cached one each
    st_todo = [r for r in finished if str(r["match_id"]) not in _load_ids(st_csv, "match_id")]
    print(f"stats: fetching {len(st_todo)} new")
    for r in st_todo:
        mid = r["match_id"]
        stt = _get(f"statistics/{mid}")
        out = []
        for t in (stt or []):
            tname = (t.get("team") or {}).get("name")
            side = "home" if tname == r["home"] else "away" if tname == r["away"] else None
            if side is None:
                continue
            out += [{"match_id": mid, "side": side, "team": r[side],
                     "stat": s.get("displayName"), "value": s.get("value")}
                    for s in (t.get("statistics") or [])]
        _append(st_csv, out, ["match_id", "side", "team", "stat", "value"])

    _save_quota()
    print(f"\n[ok] Done. {_used} requests this run, {_prior + _used}/{DAILY_CAP} today. Cached under {CACHE.name}/.")


if __name__ == "__main__":
    main()
