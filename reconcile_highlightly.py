"""Daily reconcile: does results.csv agree with the Highlightly cache?

Reads results.csv (the model's dataset) and api_cache/wc_matches.csv (Highlightly)
and reports, for every played 2026 World Cup match, whether the dataset score
matches Highlightly. Flags:
  - DISAGREE : both have a score but they differ (a real discrepancy to look at)
  - MISSING  : Highlightly has the game, results.csv has no fixture row for it
  - PENDING  : the fixture is in results.csv but still NA (feed hasn't filled it yet)

Read-only -- never writes anything. Team names are mapped to the martj42 spellings
and the home/away order is matched either way, within a +/-1 day window (Highlightly
dates are UTC, martj42 uses the venue's local date). Runs in the daily build so you
can confirm at a glance that the dataset is in sync with Highlightly.

    python reconcile_highlightly.py [REPO_DIR]   # default: .
"""
from __future__ import annotations

import csv
import sys
from datetime import date, timedelta
from pathlib import Path

from wc2026.richdata import HL_TO_MARTJ42


def _norm(name: str) -> str:
    return HL_TO_MARTJ42.get(name, name)


def _shift(d: str, n: int) -> str:
    y, m, dd = (int(x) for x in d.split("-"))
    return (date(y, m, dd) + timedelta(days=n)).isoformat()


def reconcile(results_csv: Path, matches_csv: Path) -> dict:
    # Highlightly played matches -> (date, home, away, home_score, away_score)
    hl = []
    with matches_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sc = (r.get("score") or "").replace(" ", "")
            if "-" not in sc:
                continue
            try:
                hs, a_s = (int(x) for x in sc.split("-"))
            except ValueError:
                continue
            hl.append((r["date"], _norm(r["home"]), _norm(r["away"]), hs, a_s))

    # results.csv WC-2026 rows indexed by (date, home, away) -> (hs, as) or None when NA
    res = {}
    with results_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["tournament"] != "FIFA World Cup" or not r["date"].startswith("2026-"):
                continue
            hs = r["home_score"]
            score = None if hs in ("", "NA") else (int(hs), int(r["away_score"]))
            res[(r["date"], r["home_team"], r["away_team"])] = score

    agree, disagree, missing, pending = [], [], [], []
    for d, h, a, hs, a_s in hl:
        found = None                                   # (dataset_score, reversed?)
        for cd in (d, _shift(d, -1), _shift(d, 1)):    # +/-1 day: UTC vs local
            if (cd, h, a) in res:
                found = (res[(cd, h, a)], False); break
            if (cd, a, h) in res:                      # fixture stored reversed
                found = (res[(cd, a, h)], True); break
        label = f"{h} {hs}-{a_s} {a} ({d})"
        if found is None:
            missing.append(label)
        elif found[0] is None:
            pending.append(label)
        else:
            ds = found[0]
            want = (a_s, hs) if found[1] else (hs, a_s)   # match Highlightly in the row's order
            if ds == want:
                agree.append(label)
            else:
                shown = f"{ds[1]}-{ds[0]}" if found[1] else f"{ds[0]}-{ds[1]}"
                disagree.append(f"{h} vs {a} ({d}): dataset {shown}, Highlightly {hs}-{a_s}")
    return {"agree": agree, "disagree": disagree, "missing": missing, "pending": pending}


def main() -> None:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    res = repo / "results.csv"
    mc = repo / "api_cache" / "wc_matches.csv"
    if not res.exists():
        sys.exit(f"results.csv not found in {repo.resolve()}")
    if not mc.exists():
        print("  no Highlightly cache -- nothing to reconcile")
        return
    print("Reconciling results.csv vs Highlightly ...")
    rep = reconcile(res, mc)
    n = sum(len(rep[k]) for k in rep)
    print(f"  {len(rep['agree'])}/{n} played WC matches agree with Highlightly")
    if rep["disagree"]:
        print(f"\n  /!\\ {len(rep['disagree'])} DISAGREE -- look at these:")
        for x in rep["disagree"]:
            print(f"     - {x}")
    if rep["missing"]:
        print(f"\n  /!\\ {len(rep['missing'])} in Highlightly but missing from results.csv:")
        for x in rep["missing"]:
            print(f"     - {x}")
    if rep["pending"]:
        print(f"\n  ... {len(rep['pending'])} not yet scored in results.csv (feed pending):")
        for x in rep["pending"]:
            print(f"     - {x}")
    if not (rep["disagree"] or rep["missing"]):
        print("\n  OK -- dataset is in sync with Highlightly.")


if __name__ == "__main__":
    main()
