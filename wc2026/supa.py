"""Push the World Cup match table to Supabase (for the prediction leaderboard).

The daily pipeline upserts every group fixture — its kickoff, the model's
most-likely scoreline and the result once played — into the `matches` table,
using the service-role key. A safe no-op when the SUPABASE_* env vars are absent,
so it never breaks the build (e.g. on a fork or before the secrets are set).
"""
from __future__ import annotations

import json
import os
import urllib.request


def _ko_iso(kickoffs: dict, home: str, away: str):
    k = kickoffs.get("|".join(sorted([home, away])))
    return f"{k['date']}T{k['hm']}:00+01:00" if k else None     # WEST = UTC+1


def push_matches(data: dict) -> None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return
    ko, rows = data.get("kickoffs", {}), {}
    for m in data.get("matches") or []:              # upcoming: model pick = top[0]
        h, a = m["home"], m["away"]
        mh, ma = (m["top"][0]["score"].split("-") if m.get("top") else ("1", "1"))
        rows[f"{h}|{a}"] = {"match_id": f"{h}|{a}", "home": h, "away": a,
                            "kickoff": _ko_iso(ko, h, a),
                            "home_score": None, "away_score": None,   # uniform keys
                            "model_home": int(mh), "model_away": int(ma),
                            "stage": "group", "played": False}
    for m in data.get("played_review") or []:        # played: model pick + result
        h, a = m["home"], m["away"]
        mh, ma = m["ml_score"].split("-")
        rows[f"{h}|{a}"] = {"match_id": f"{h}|{a}", "home": h, "away": a,
                            "kickoff": _ko_iso(ko, h, a), "home_score": int(m["hs"]),
                            "away_score": int(m["as"]), "model_home": int(mh),
                            "model_away": int(ma), "stage": "group", "played": True}
    body = list(rows.values())
    if not body:
        return
    req = urllib.request.Request(
        url.rstrip("/") + "/rest/v1/matches",
        data=json.dumps(body).encode("utf-8"),
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"},
        method="POST")
    try:
        urllib.request.urlopen(req, timeout=30)
        print(f"Supabase: upserted {len(body)} matches.")
    except Exception as e:                            # never break the pipeline
        print("Supabase push failed (continuing):", type(e).__name__, e)
