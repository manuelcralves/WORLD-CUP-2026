"""Auto-feed Highlightly goal events into goalscorers.csv (full names via player_id).

Fills the scorers of any played 2026 World Cup match NOT already in goalscorers.csv,
so the marcadores / Golden Boot stay current without manual entry. The abbreviated
event name ("M. Galarza") is mapped to the full line-up name ("Matias Galarza") by
player_id. Games already in goalscorers.csv (manual / martj42) are left untouched.
Own goals are credited to the opponent (own_goal=TRUE); penalties flagged. The date
and home/away order are taken from results.csv so the rows join the dataset cleanly.

Any scorer whose player_id is not in the line-ups is printed as UNMAPPED so you can
check the spelling (the only Golden-Boot fragmentation risk).

    python feed_goals_highlightly.py [REPO_DIR]   # default: .
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from wc2026.richdata import HL_TO_MARTJ42

GOAL_TYPES = {"Goal", "Penalty", "Own Goal"}
FIELDS = ["date", "home_team", "away_team", "team", "scorer", "minute", "own_goal", "penalty"]


def _norm(name: str) -> str:
    return HL_TO_MARTJ42.get(name, name)


def feed(repo: Path) -> int:
    cache = repo / "api_cache"
    mfile, lfile, efile = cache / "wc_matches.csv", cache / "wc_lineups.csv", cache / "wc_events.csv"
    gfile, rfile = repo / "goalscorers.csv", repo / "results.csv"
    if not (mfile.exists() and efile.exists() and gfile.exists() and rfile.exists()):
        print("  missing cache / goalscorers.csv / results.csv -- nothing to feed")
        return 0

    # match_id -> (home, away) in martj42 spelling
    matches = {}
    with mfile.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            matches[r["match_id"]] = (_norm(r["home"]), _norm(r["away"]))

    # player_id -> (full name, team) from the line-ups
    pmap = {}
    if lfile.exists():
        with lfile.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("player_id"):
                    pmap[r["player_id"]] = (r["player"], _norm(r["team"]))

    # canonical (date, home, away) from results.csv, keyed by the unordered team pair
    res = {}
    with rfile.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["tournament"] == "FIFA World Cup" and r["date"].startswith("2026-"):
                res[frozenset((r["home_team"], r["away_team"]))] = (r["date"], r["home_team"], r["away_team"])

    # (date, pair) that already has a scorer row -> never overwritten. Keyed on the
    # DATE too, not just the pair: the same two teams meet again and again across
    # history (and across rounds), so a pair-only check skipped every new game
    # between teams that have ever played — i.e. essentially all knockout ties.
    have = set()
    with gfile.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            have.add((r["date"], frozenset((r["home_team"], r["away_team"]))))

    # goal events grouped by match
    goals_by = {}
    with efile.open(encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e["type"] in GOAL_TYPES:
                goals_by.setdefault(e["match_id"], []).append(e)

    new_rows, filled, skipped, no_fixture, unmapped = [], 0, 0, 0, 0
    for mid, evs in goals_by.items():
        if mid not in matches:
            continue
        pair = frozenset(matches[mid])
        if pair not in res:
            no_fixture += 1
            continue
        gdate, ghome, gaway = res[pair]
        if (gdate, pair) in have:                       # this exact game already has scorers
            skipped += 1
            continue
        rows_here = []
        for e in evs:
            if e["type"] == "Penalty" and (e.get("minute") or "").startswith("120+"):
                continue   # penalty-SHOOTOUT kick (after extra time) -> not a match goal; feeding it would inflate the score + Golden Boot
            full, scteam = pmap.get(e.get("player_id"), (None, None))
            if full is None:                              # scorer not in the line-ups
                unmapped += 1
                full, scteam = e.get("player"), _norm(e.get("team") or "")
                print(f"  UNMAPPED scorer '{full}' in {ghome} v {gaway} -- check spelling")
            og = e["type"] == "Own Goal"
            credited = (gaway if scteam == ghome else ghome) if og else scteam
            mn = (e.get("minute") or "").split("+")[0]
            rows_here.append({"date": gdate, "home_team": ghome, "away_team": gaway,
                              "team": credited, "scorer": full,
                              "minute": mn if mn.isdigit() else "",
                              "own_goal": "TRUE" if og else "FALSE",
                              "penalty": "TRUE" if e["type"] == "Penalty" else "FALSE"})
        if rows_here:
            new_rows += rows_here
            filled += 1
            print(f"  + {ghome} v {gaway} ({gdate}): {len(rows_here)} goal(s)")

    if new_rows:
        raw = gfile.read_bytes()
        nl = "\r\n" if b"\r\n" in raw else "\n"
        if raw and not raw.endswith(nl.encode()):
            gfile.write_bytes(raw + nl.encode())
        with gfile.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS, lineterminator=nl).writerows(new_rows)
    print(f"  ({filled} games filled, {skipped} already had goals, "
          f"{no_fixture} no fixture in results.csv, {unmapped} names unmapped)")
    return filled


def main() -> None:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    print("Feeding Highlightly goals into goalscorers.csv ...")
    n = feed(repo)
    print(f"\nDone: {n} game(s) of scorers filled from Highlightly.")


if __name__ == "__main__":
    main()
