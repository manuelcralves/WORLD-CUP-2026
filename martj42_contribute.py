"""Prepare a clean contribution to martj42/international_results.

Fills in the score of matches already listed in results.csv (NA,NA -> real
score) and appends their goalscorers, preserving the exact CSV byte format so
the pull-request diff stays minimal (only the touched lines change).

Usage:
    python martj42_contribute.py [PATH_TO_MARTJ42_CLONE]   # default: .

Workflow: fork martj42/international_results, clone it, edit the RESULTS and
GOALS blocks below with real, sourced results, run this script pointing at the
clone, then commit and open a pull request. Re-running is safe: matches that
already have a score (or goals already present) are skipped.

Minute convention (matches the dataset): stoppage-time goals are recorded at
the base minute of the half -- 1st-half 45+x -> 45, 2nd-half 90+x -> 90. The
dataset does not use 90+2 / 91-99 for these.
"""
from __future__ import annotations

import sys
from pathlib import Path

# (date, home_team, away_team, home_score, away_score)
RESULTS = [
    ("2026-06-17", "Portugal",   "DR Congo", 1, 1),
    ("2026-06-17", "England",    "Croatia",  4, 2),
    ("2026-06-17", "Ghana",      "Panama",   1, 0),
    ("2026-06-17", "Uzbekistan", "Colombia", 1, 3),
    ("2026-06-18", "Czech Republic", "South Africa",           1, 1),
    ("2026-06-18", "Switzerland",    "Bosnia and Herzegovina", 4, 1),
    ("2026-06-18", "Canada",         "Qatar",                  6, 0),
    ("2026-06-18", "Mexico",         "South Korea",            1, 0),
]

# (date, home_team, away_team, team, scorer, minute, own_goal, penalty)
GOALS = [
    ("2026-06-17", "Portugal",   "DR Congo", "Portugal",   "João Neves",           6, False, False),
    ("2026-06-17", "Portugal",   "DR Congo", "DR Congo",   "Yoane Wissa",          45, False, False),  # 45+5
    ("2026-06-17", "England",    "Croatia",  "England",    "Harry Kane",           12, False, True),   # penalty
    ("2026-06-17", "England",    "Croatia",  "Croatia",    "Martin Baturina",      36, False, False),
    ("2026-06-17", "England",    "Croatia",  "England",    "Harry Kane",           42, False, False),
    ("2026-06-17", "England",    "Croatia",  "Croatia",    "Petar Musa",           45, False, False),  # 45+5
    ("2026-06-17", "England",    "Croatia",  "England",    "Jude Bellingham",      47, False, False),
    ("2026-06-17", "England",    "Croatia",  "England",    "Marcus Rashford",      85, False, False),
    ("2026-06-17", "Ghana",      "Panama",   "Ghana",      "Caleb Yirenkyi",       90, False, False),  # 90+5 -> 90
    ("2026-06-17", "Uzbekistan", "Colombia", "Colombia",   "Daniel Muñoz",         40, False, False),
    ("2026-06-17", "Uzbekistan", "Colombia", "Uzbekistan", "Abbosbek Fayzullaev",  60, False, False),
    ("2026-06-17", "Uzbekistan", "Colombia", "Colombia",   "Luis Díaz",            65, False, False),
    ("2026-06-17", "Uzbekistan", "Colombia", "Colombia",   "Jaminton Campaz",      90, False, False),  # 90+9 -> 90
    ("2026-06-18", "Czech Republic", "South Africa", "Czech Republic", "Michal Sadilek",  6,  False, False),
    ("2026-06-18", "Czech Republic", "South Africa", "South Africa",   "Teboho Mokoena",  83, False, True),   # penalty
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina", "Switzerland",            "Johan Manzambi", 74, False, False),
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina", "Switzerland",            "Ruben Vargas",   84, False, False),
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina", "Switzerland",            "Johan Manzambi", 90, False, False),  # 90+3 -> 90
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina", "Bosnia and Herzegovina", "Ermin Mahmic",   90, False, False),  # 90+3 -> 90
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina", "Switzerland",            "Granit Xhaka",   90, False, True),   # penalty, 90+7 -> 90
    ("2026-06-18", "Canada", "Qatar", "Canada", "Cyle Larin",         16, False, False),
    ("2026-06-18", "Canada", "Qatar", "Canada", "Jonathan David",     29, False, False),
    ("2026-06-18", "Canada", "Qatar", "Canada", "Jonathan David",     45, False, False),  # 45+3 -> 45
    ("2026-06-18", "Canada", "Qatar", "Canada", "Nathan Saliba",      64, False, False),
    ("2026-06-18", "Canada", "Qatar", "Canada", "Mohammad Al Mannai", 75, True,  False),  # own goal: Qatar player, counts for Canada
    ("2026-06-18", "Canada", "Qatar", "Canada", "Jonathan David",     90, False, False),  # 90+2 -> 90
    ("2026-06-18", "Mexico", "South Korea", "Mexico", "Luis Romo",    50, False, False),
]

# (date, home_team, away_team, winner, first_shooter) — penalty-shootout outcomes.
# Highlightly's free tier doesn't expose the shootout WINNER (only the status
# 'Finished after penalties' + the level score), so we source these by hand here.
# Re-applied every build AFTER update_data.py pulls martj42's copy (which would
# otherwise wipe a manual shootouts.csv row), so the model sends the real winner
# through. Drop a row once martj42 merges it upstream.
SHOOTOUTS = [
    ("2026-06-29", "Germany",     "Paraguay", "Paraguay", "Germany"),      # Paraguay won the shootout
    ("2026-06-29", "Netherlands", "Morocco",  "Morocco",  "Netherlands"),  # Morocco won the shootout 3-2
    ("2026-07-03", "Australia",   "Egypt",    "Egypt",    "Australia"),     # Egypt won the shootout (R32)
    ("2026-07-07", "Switzerland", "Colombia", "Switzerland", "Colombia"),   # Switzerland won the shootout (R16)
]


def _b(v: bool) -> str:
    return "TRUE" if v else "FALSE"


def _newline(data: bytes) -> bytes:
    return b"\r\n" if b"\r\n" in data else b"\n"


def fill_results(path: Path) -> int:
    data = path.read_bytes()
    nl = _newline(data)
    lines = data.split(nl)
    changed = 0
    for i, raw in enumerate(lines):
        line = raw.decode("utf-8")
        for d, h, a, hs, as_ in RESULTS:
            prefix = f"{d},{h},{a},"
            if line.startswith(prefix):
                rest = line[len(prefix):]
                if rest.startswith("NA,NA,"):
                    lines[i] = (prefix + rest.replace("NA,NA", f"{hs},{as_}", 1)).encode("utf-8")
                    changed += 1
                    print(f"  results : {h} {hs}-{as_} {a}")
                else:
                    print(f"  results : SKIP {h} vs {a} (already has a score)")
    if changed:
        path.write_bytes(nl.join(lines))
    return changed


def add_goals(path: Path) -> int:
    data = path.read_bytes()
    nl = _newline(data)
    if not data.endswith(nl):
        data += nl
    new = []
    for d, h, a, team, scorer, minute, og, pen in GOALS:
        row = f"{d},{h},{a},{team},{scorer},{minute},{_b(og)},{_b(pen)}".encode("utf-8")
        if row + nl in data:
            print(f"  goals   : SKIP {scorer} {minute}' (already present)")
        else:
            new.append(row)
            print(f"  goals   : {minute}' {scorer} ({team})")
    if new:
        data += nl.join(new) + nl
        path.write_bytes(data)
    return len(new)


def fill_shootouts(path: Path) -> int:
    if not path.exists() or not SHOOTOUTS:
        return 0
    data = path.read_bytes()
    nl = _newline(data)
    if not data.endswith(nl):
        data += nl
    lines = data.split(nl)
    new = []
    for d, h, a, winner, first in SHOOTOUTS:
        pres = [f"{d},{h},{a},".encode("utf-8"), f"{d},{a},{h},".encode("utf-8")]   # either home/away order
        if any(ln.startswith(p) for ln in lines for p in pres):
            print(f"  shootout: SKIP {h} v {a} (already present)")
        else:
            new.append(f"{d},{h},{a},{winner},{first}".encode("utf-8"))
            print(f"  shootout: {h} v {a} -> {winner} won on penalties")
    if new:
        data += nl.join(new) + nl
        path.write_bytes(data)
    return len(new)


def main() -> None:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    res, gls, sho = repo / "results.csv", repo / "goalscorers.csv", repo / "shootouts.csv"
    if not res.exists() or not gls.exists():
        sys.exit(f"results.csv / goalscorers.csv not found in {repo.resolve()}")
    print(f"Updating dataset in {repo.resolve()} ...")
    r = fill_results(res)
    g = add_goals(gls)
    s = fill_shootouts(sho)
    print(f"\nDone: {r} result(s) filled, {g} goal(s) added, {s} shootout(s) added.")


if __name__ == "__main__":
    main()
