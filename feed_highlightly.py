"""Auto-feed the Highlightly World Cup results into results.csv (the model's dataset).

Reads api_cache/wc_matches.csv (fetched by fetch_wc_data.py) and fills the score of
any FINISHED 2026 World Cup match still listed as `NA,NA` in results.csv — so the model has
the latest results within ~2h of a game ending, without waiting for the daily
martj42 pull or manual entry.

Byte-preserving like martj42_contribute.py (only touched lines change). Team names
are mapped to the martj42 spellings; the home/away order is matched either way.
Re-running is safe: rows that already have a score are skipped. Goal scorers are
NOT fed (Highlightly abbreviates player names, which would fragment the Golden
Boot) — those stay on the manual / martj42 rhythm.

    python feed_highlightly.py [REPO_DIR]   # default: .
"""
from __future__ import annotations

import csv
import sys
from datetime import date, timedelta
from pathlib import Path

from wc2026.richdata import HL_TO_MARTJ42


def _norm(name: str) -> str:
    return HL_TO_MARTJ42.get(name, name)


def feed(results_csv: Path, matches_csv: Path) -> int:
    if not matches_csv.exists():
        print(f"  no {matches_csv} — nothing to feed")
        return 0
    played = []                                   # (date, home, away, home_score, away_score)
    with matches_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not (r.get("status") or "").startswith("Finished"):   # FINISHED / after extra time / after penalties — never a live/mid-game score (mirrors fetch_wc_data.py)
                continue
            sc = (r.get("score") or "").replace(" ", "")
            if "-" not in sc:
                continue
            try:
                hs, a_s = (int(x) for x in sc.split("-"))
            except ValueError:
                continue
            played.append((r["date"], _norm(r["home"]), _norm(r["away"]), hs, a_s))
    if not played:
        print("  no played matches in the Highlightly cache")
        return 0

    data = results_csv.read_bytes()
    nl = b"\r\n" if b"\r\n" in data else b"\n"
    lines = data.split(nl)
    # index the 2026 World Cup fixture rows by (date, home, away) -> line number
    rowidx = {}
    for i, raw in enumerate(lines):
        line = raw.decode("utf-8")
        if line.startswith("2026-") and ",FIFA World Cup," in line:
            p = line.split(",", 3)
            if len(p) >= 3:
                rowidx[(p[0], p[1], p[2])] = i

    def _shift(d: str, n: int) -> str:
        y, m, dd = (int(x) for x in d.split("-"))
        return (date(y, m, dd) + timedelta(days=n)).isoformat()

    filled = skipped = notfound = 0
    for d, h, a, hs, a_s in played:
        hit = None                                    # Highlightly date is UTC; martj42 is local -> allow ±1 day
        for cd in (d, _shift(d, -1), _shift(d, 1)):
            if (cd, h, a) in rowidx:
                hit = (rowidx[(cd, h, a)], hs, a_s)
                break
            if (cd, a, h) in rowidx:                  # reversed home/away -> flip the score
                hit = (rowidx[(cd, a, h)], a_s, hs)
                break
        if hit is None:
            notfound += 1
            print(f"  NOT FOUND: {h} vs {a} ({d})")
            continue
        li, x, y = hit
        line = lines[li].decode("utf-8")
        p = line.split(",", 3)
        prefix = f"{p[0]},{p[1]},{p[2]},"
        rest = line[len(prefix):]
        if rest.startswith("NA,NA,"):
            lines[li] = (prefix + rest.replace("NA,NA", f"{x},{y}", 1)).encode("utf-8")
            filled += 1
            print(f"  filled: {h} {hs}-{a_s} {a}  (results.csv {p[0]})")
        else:
            skipped += 1
    if filled:
        results_csv.write_bytes(nl.join(lines))
    print(f"  ({filled} filled, {skipped} already scored, {notfound} not found in results.csv)")
    return filled


def main() -> None:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    res = repo / "results.csv"
    if not res.exists():
        sys.exit(f"results.csv not found in {repo.resolve()}")
    print(f"Feeding Highlightly results into {res} ...")
    n = feed(res, repo / "api_cache" / "wc_matches.csv")
    print(f"\nDone: {n} result(s) filled from Highlightly.")


if __name__ == "__main__":
    main()
