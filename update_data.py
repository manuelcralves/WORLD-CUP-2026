"""Updates the dataset from the live source (martj42 on GitHub).

Backs up the current CSVs into _backup/ and downloads the latest version of the 4
files. After that, just re-run the pipeline to refresh the forecasts:

    python update_data.py
    python run_pipeline.py both
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv"]
HERE = Path(__file__).resolve().parent


def main():
    backup = HERE / "_backup"
    backup.mkdir(exist_ok=True)
    print("Updating from github.com/martj42/international_results ...")
    for f in FILES:
        if (HERE / f).exists():
            shutil.copy(HERE / f, backup / f)
        urllib.request.urlretrieve(f"{BASE_URL}/{f}", HERE / f)
        print(f"  updated: {f}")

    r = pd.read_csv(HERE / "results.csv", parse_dates=["date"])
    wc = r[(r["tournament"] == "FIFA World Cup") & (r["date"].dt.year == 2026)]
    print(f"\nLatest date: {r['date'].max().date()} | "
          f"2026 World Cup: {int(wc['home_score'].notna().sum())}/72 matches played")
    print("Backup of the previous files in _backup/.")
    print("Now run:  python run_pipeline.py both")


if __name__ == "__main__":
    main()
