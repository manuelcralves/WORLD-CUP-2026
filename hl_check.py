"""Probe: does the Highlightly FREE plan reach the 2026 World Cup?

This spends ~2 requests and dumps the raw responses so we can (a) see whether the
free tier is allowed to touch season 2026, and (b) map Highlightly's JSON schema
before building the full fetcher.

    # PowerShell:
    $env:HIGHLIGHTLY_KEY = "your-key-here"
    python hl_check.py

If you registered through RapidAPI (instead of directly on highlightly.net):
    $env:HIGHLIGHTLY_RAPIDAPI = "1"
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

KEY = os.environ.get("HIGHLIGHTLY_KEY")
RAPID = os.environ.get("HIGHLIGHTLY_RAPIDAPI") == "1"
BASE = ("https://football-highlights-api.p.rapidapi.com" if RAPID
        else "https://soccer.highlightly.net")
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")   # Cloudflare 1010-blocks the default python UA
# The direct API at soccer.highlightly.net authenticates with x-rapidapi-key too (confirmed) -- NOT x-api-key.
HEADERS = ({"x-rapidapi-key": KEY or "", "x-rapidapi-host": "football-highlights-api.p.rapidapi.com", "User-Agent": _UA}
           if RAPID else {"x-rapidapi-key": KEY or "", "User-Agent": _UA})


def get(path, **params):
    url = f"{BASE}/{path}" + (("?" + urllib.parse.urlencode(params)) if params else "")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
    except Exception as e:  # noqa
        return None, str(e)


def _name(it):
    return str(it.get("name") or (it.get("league") or {}).get("name") or "")


def _id(it):
    return it.get("id") or it.get("leagueId") or (it.get("league") or {}).get("id")


if not KEY:
    sys.exit('[x] Set HIGHLIGHTLY_KEY first  ($env:HIGHLIGHTLY_KEY = "...").')

print(f"Base: {BASE}\n")
print("== STEP 1: find the World Cup league ==")
data, err = get("leagues", leagueName="World Cup")
if err:
    sys.exit(f"[x] /leagues failed: {err}")
print(json.dumps(data, ensure_ascii=False, indent=1)[:2000])

items = data.get("data") if isinstance(data, dict) else data
wc_id = None
for it in (items or []):
    n = _name(it).lower()
    if n == "world cup" or ("world cup" in n and not any(
            x in n for x in ("women", "qual", "u20", "u17", "club"))):
        wc_id = _id(it)
        break
print(f"\n--> picked World Cup leagueId = {wc_id}")

if wc_id is not None:
    print("\n== STEP 2: 2026 World Cup matches ==")
    m, err = get("matches", leagueId=wc_id, season=2026, limit=100)
    if err:
        print("[x] /matches season=2026 failed:", err)
    else:
        rows = m.get("data") if isinstance(m, dict) else m
        page = m.get("pagination") if isinstance(m, dict) else None
        print(f"[ok] 2026 matches: {len(rows or [])}  pagination={page}")
        # STEP 3: grab a FINISHED match and dump its line-up + events (schema mapping)
        finished = [r for r in (rows or []) if ((r.get("state") or {}).get("score") or {}).get("current")]
        print(f"\n== STEP 3: finished matches = {len(finished)} ==")
        if finished:
            fm = finished[0]
            fid = fm["id"]
            print(f"sampling {fid}: {fm['homeTeam']['name']} {fm['state']['score']['current']} {fm['awayTeam']['name']}")
            ln, e1 = get(f"lineups/{fid}")
            print("\n-- LINEUP --", e1 or "")
            print(json.dumps(ln, ensure_ascii=False, indent=1)[:1700])
            ev, e2 = get(f"events/{fid}")
            print("\n-- EVENTS --", e2 or "")
            print(json.dumps(ev, ensure_ascii=False, indent=1)[:1700])

print("\n>>> Paste me this WHOLE output (there's no key in it) and I'll take it from here.")
