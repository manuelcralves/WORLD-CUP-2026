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


def assist_counts(cache) -> dict:
    """WC assists tallied by the assister's lower-cased surname, from the Highlightly goal
    events (which give only an abbreviated name and no id, so we match on surname). Used to
    show assists + break Golden Boot ties (goals -> assists). Empty if the cache is absent."""
    out: dict = defaultdict(int)
    ef = Path(cache) / "wc_events.csv"
    if ef.exists():
        with ef.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                a = (r.get("assist") or "").strip()
                if a:
                    out[a.split()[-1].lower()] += 1
    return dict(out)


def _num(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 99


# Highlightly returns ~40 stats per match; show only these classics, in this order.
STAT_ORDER = ["Possession", "Expected Goals", "Shots on target", "Shots off target",
              "Corners", "Offsides", "Fouls", "Yellow cards", "Red cards", "Successful passes"]


def _pct(v):
    try:
        return f"{round(float(v) * 100)}%"      # possession 0.51 -> "51%"
    except (TypeError, ValueError):
        return v


def _zero(v) -> bool:
    return v in ("", "0", "0.0", None)


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
                    {"player": r["player"], "number": r["number"], "position": r["position"],
                     "id": r.get("player_id", "")})

    # events: match_id -> [ {minute, team, type, player, assist, out} ]
    events: dict = defaultdict(list)
    ef = cache / "wc_events.csv"
    if ef.exists():
        with ef.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                events[r["match_id"]].append({
                    "minute": r["minute"], "team": _norm(r["team"]), "type": r["type"],
                    "player": r["player"], "assist": r["assist"], "out": r["out"],
                    "pid": r.get("player_id", ""), "out_pid": r.get("out_pid", "")})

    # match statistics: match_id -> {home: {stat: value}, away: {stat: value}}
    stats: dict = {}
    sf = cache / "wc_stats.csv"
    if sf.exists():
        with sf.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = stats.setdefault(r["match_id"], {"home": {}, "away": {}})
                side = d.get(r["side"])
                if side is not None:
                    side[r["stat"]] = r["value"]

    # per-match detail, keyed by the site's "home|away" (martj42 names)
    detail = {}
    blank = {"formation": None, "xi": [], "bench": []}
    for mid, (home, away) in mid_teams.items():
        ln = lineups.get(mid, {})
        st = stats.get(mid, {"home": {}, "away": {}})
        srows = []
        for k in STAT_ORDER:
            hv, av = st["home"].get(k, ""), st["away"].get(k, "")
            if k == "Possession":
                hv, av = _pct(hv), _pct(av)          # 0.51 -> 51%
            if not _zero(hv) or not _zero(av):       # skip stats both teams have at 0
                srows.append({"stat": k, "home": hv, "away": av})
        detail[f"{home}|{away}"] = {
            "score": scores.get(mid),
            "timeline": events.get(mid, []),
            "stats": srows,
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
