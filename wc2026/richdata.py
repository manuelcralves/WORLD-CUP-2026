"""Load the Highlightly rich-data CSVs (api_cache/) into structures for the
dashboard: per-match detail (timeline + line-ups), per-team squads, a card
ranking, and FIFA fair-play points per team.

All team names are normalised to the martj42 spellings the rest of the site uses.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

# Highlightly spells a few teams differently from the martj42 dataset.
HL_TO_MARTJ42 = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "USA": "United States",
}


def _norm(name: str) -> str:
    return HL_TO_MARTJ42.get(name, name)


def _num(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 99


def load_rich(cache_dir) -> dict:
    """Returns {matchDetail, squads, cards, fairplay} or {} if the cache is absent."""
    cache = Path(cache_dir)
    mfile = cache / "wc_matches.csv"
    if not mfile.exists():
        return {}

    mid_teams, scores = {}, {}
    with mfile.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mid_teams[r["match_id"]] = (_norm(r["home"]), _norm(r["away"]))
            scores[r["match_id"]] = r["score"]

    # line-ups: match_id -> {home/away: {formation, xi, bench}}
    lineups: dict = {}
    lf = cache / "wc_lineups.csv"
    if lf.exists():
        with lf.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = lineups.setdefault(r["match_id"], {
                    "home": {"formation": None, "xi": [], "bench": []},
                    "away": {"formation": None, "xi": [], "bench": []}})
                side = d.get(r["side"])
                if side is None:
                    continue
                side["formation"] = r["formation"]
                (side["xi"] if r["starter"] == "yes" else side["bench"]).append(
                    {"player": r["player"], "number": r["number"], "position": r["position"]})

    # events: match_id -> [ {minute, team, type, player, assist, out} ]
    events: dict = defaultdict(list)
    ef = cache / "wc_events.csv"
    if ef.exists():
        with ef.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                events[r["match_id"]].append({
                    "minute": r["minute"], "team": _norm(r["team"]), "type": r["type"],
                    "player": r["player"], "assist": r["assist"], "out": r["out"]})

    # per-match detail, keyed by the site's "home|away" (martj42 names)
    detail = {}
    blank = {"formation": None, "xi": [], "bench": []}
    for mid, (home, away) in mid_teams.items():
        ln = lineups.get(mid, {})
        detail[f"{home}|{away}"] = {
            "score": scores.get(mid),
            "timeline": events.get(mid, []),
            "home": {"team": home, **ln.get("home", dict(blank))},
            "away": {"team": away, **ln.get("away", dict(blank))}}

    # squads: per team, deduped across matches, ordered by shirt number
    sq: dict = defaultdict(dict)
    for mid, (home, away) in mid_teams.items():
        for sidekey, team in (("home", home), ("away", away)):
            side = lineups.get(mid, {}).get(sidekey, {})
            for p in side.get("xi", []) + side.get("bench", []):
                sq[team][p["player"]] = {"number": p["number"], "position": p["position"]}
    squads = {t: [{"player": pl, **info} for pl, info in
                  sorted(ps.items(), key=lambda kv: _num(kv[1]["number"]))]
              for t, ps in sq.items()}

    # cards (ranking) + fair-play points (tie-break): yellow -1, red -4
    cp: dict = defaultdict(lambda: {"team": None, "Y": 0, "R": 0})
    ct: dict = defaultdict(lambda: {"Y": 0, "R": 0})
    for evs in events.values():
        for e in evs:
            k = "Y" if e["type"] == "Yellow Card" else "R" if e["type"] == "Red Card" else None
            if not k:
                continue
            ct[e["team"]][k] += 1
            if e["player"]:
                cp[e["player"]]["team"] = e["team"]
                cp[e["player"]][k] += 1
    players = sorted(({"player": pl, **info} for pl, info in cp.items()),
                     key=lambda c: (-(c["R"] * 4 + c["Y"]), c["player"]))
    teams = sorted(({"team": t, **info} for t, info in ct.items()),
                   key=lambda c: (-(c["R"] * 4 + c["Y"]), c["team"]))
    fairplay = {t: -(c["Y"] + 4 * c["R"]) for t, c in ct.items()}

    return {"matchDetail": detail, "squads": squads,
            "cards": {"players": players, "teams": teams}, "fairplay": fairplay}
