"""Data loading and preparation.

Reads the CSVs from the "International football results from 1872 to 2026"
dataset, normalizes team names (former_names), and isolates the 2026 World Cup:
reconstructs the 12 groups from the schedule and separates the matches already
played from those still to be played.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

# Directory where the CSVs live (the project root, one level above this package).
DATA_DIR = Path(__file__).resolve().parent.parent

# The three 2026 World Cup hosts play "at home".
HOSTS = {"United States", "Mexico", "Canada"}


def load_results(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load results.csv with correct types (dates and the `neutral` boolean)."""
    df = pd.read_csv(Path(data_dir) / "results.csv", parse_dates=["date"])
    # The CSV stores TRUE/FALSE as text.
    df["neutral"] = df["neutral"].astype(str).str.strip().str.upper().eq("TRUE")
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


def load_former_names(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(
        Path(data_dir) / "former_names.csv", parse_dates=["start_date", "end_date"]
    )


def name_mapping(former: pd.DataFrame) -> dict[str, str]:
    """Map former_name -> current_name (e.g. 'Dahomey' -> 'Benin').

    We give national teams historical continuity: the matches of a team under
    an old name now count towards the current team (improves Elo and form).
    """
    return dict(zip(former["former"], former["current"]))


def normalize_names(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(mapping)
    return df


def is_played(df: pd.DataFrame) -> pd.Series:
    """Mask of matches with a known result (non-null scores)."""
    return df["home_score"].notna() & df["away_score"].notna()


def world_cup_2026(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the 2026 World Cup GROUP-STAGE matches (the 72 earliest).

    Once the bracket is drawn, the knockout fixtures also land in the dataset as
    "FIFA World Cup" games — but the model builds the knockouts from the official
    bracket + simulated standings, NOT from dataset fixtures, and a cross-group
    knockout game would break the group reconstruction (it would merge two
    groups). The group stage is exactly 12*6 = 72 games and is fully played
    before any knockout, so the 72 earliest WC-2026 matches ARE the group stage.
    """
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    return wc.sort_values("date", kind="stable").head(72).copy()


def knockout_2026(df: pd.DataFrame) -> pd.DataFrame:
    """The 2026 World Cup KNOCKOUT fixtures — everything after the 72 group games
    (the cross-group ties the dataset adds once the bracket is drawn). Empty until
    the draw lands. For the Beat-the-Machine predict list, NOT the group sim."""
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    return wc.sort_values("date", kind="stable").iloc[72:].copy()


def reconstruct_groups(wc: pd.DataFrame) -> dict[str, list[str]]:
    """Reconstruct the 12 groups from who plays against whom.

    In a group stage each team faces the other 3 in its group, so the groups
    are the connected components of the matchups graph.
    The groups are labeled A..L in a stable way (by the alphabetically smallest
    team in each group); the official label is assigned later, if needed.
    """
    adj: dict[str, set[str]] = defaultdict(set)
    for m in wc.itertuples(index=False):
        adj[m.home_team].add(m.away_team)
        adj[m.away_team].add(m.home_team)

    seen: set[str] = set()
    comps: list[list[str]] = []
    for team in adj:
        if team in seen:
            continue
        stack, comp = [team], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        comps.append(sorted(comp))

    comps.sort(key=lambda c: c[0])  # stable order
    return {chr(ord("A") + i): comp for i, comp in enumerate(comps)}


def knockout_winners(knockout, mapping, data_dir: Path | str = DATA_DIR) -> dict:
    """Played knockout ties -> {frozenset({home, away}): winner}.

    Lets the simulator FIX games already played in the knockouts instead of
    re-simulating them, so the bracket/odds reflect reality and eliminated teams
    fall to 0%. Decisive games -> the higher score; draws settled on penalties ->
    the winner from shootouts.csv (names normalized to match the results frame).
    """
    shoot = {}
    sp = Path(data_dir) / "shootouts.csv"
    if sp.exists():
        try:
            for r in pd.read_csv(sp).itertuples(index=False):
                h = mapping.get(r.home_team, r.home_team)
                a = mapping.get(r.away_team, r.away_team)
                shoot[frozenset({h, a})] = mapping.get(r.winner, r.winner)
        except Exception:
            pass
    out = {}
    for g in knockout.itertuples(index=False):
        if pd.isna(g.home_score) or pd.isna(g.away_score):
            continue
        pair = frozenset({g.home_team, g.away_team})
        if g.home_score > g.away_score:
            out[pair] = g.home_team
        elif g.away_score > g.home_score:
            out[pair] = g.away_team
        elif pair in shoot:
            out[pair] = shoot[pair]
    return out


def load_all(data_dir: Path | str = DATA_DIR, cutoff=None, asof=None) -> dict:
    """Shortcut that returns everything the rest of the pipeline needs.

    With `cutoff` (e.g. "2026-06-11") you enter **pre-tournament** mode: training
    uses only matches BEFORE the date, and ALL 72 World Cup matches become
    simulated (no result is known). Without `cutoff` it is the snapshot mode:
    the matches already played stay fixed.

    Returns a dict with:
      - `matches`     : matches used in training (all, or only those before the cutoff)
      - `played`      : those in `matches` that have a result
      - `wc`          : the 72 matches of the 2026 World Cup (full schedule)
      - `wc_played`   : World Cup matches with a fixed result (empty in pre-tournament mode)
      - `wc_remaining`: World Cup matches to simulate
      - `groups`      : dict label -> list of teams
      - `cutoff`      : the cutoff date (or None)
    """
    df = load_results(data_dir)
    mapping = name_mapping(load_former_names(data_dir))
    df = normalize_names(df, mapping)

    wc = world_cup_2026(df)
    groups = reconstruct_groups(wc)

    if asof is not None:
        # "live as of date D": train on everything up to D, fix WC games played
        # by then, simulate the rest. (Used to backfill the odds-over-time chart.)
        cut = pd.Timestamp(asof)
        train = df[df["date"] <= cut].copy()
        done = wc[(wc["date"] <= cut) & is_played(wc)]
        return {
            "matches": train,
            "played": train[is_played(train)].copy(),
            "wc": wc,
            "wc_played": done.copy(),
            "wc_remaining": wc[~wc.index.isin(done.index)].copy(),
            "groups": groups,
            "cutoff": cut,
        }

    if cutoff is not None:
        cut = pd.Timestamp(cutoff)
        train = df[df["date"] < cut].copy()
        return {
            "matches": train,
            "played": train[is_played(train)].copy(),
            "wc": wc,
            "wc_played": wc.iloc[0:0].copy(),   # none fixed
            "wc_remaining": wc.copy(),           # all 72 to simulate
            "groups": groups,
            "cutoff": cut,
        }

    played_mask = is_played(df)
    ko = knockout_2026(df)
    return {
        "matches": df,
        "played": df[played_mask].copy(),
        "wc": wc,
        "wc_played": wc[is_played(wc)].copy(),
        "wc_remaining": wc[~is_played(wc)].copy(),
        "knockout": ko,
        "ko_results": knockout_winners(ko, mapping, data_dir),
        "groups": groups,
        "cutoff": None,
    }


if __name__ == "__main__":
    d = load_all()
    print(f"Total matches: {len(d['matches'])} | with result: {len(d['played'])}")
    print(f"2026 World Cup: {len(d['wc'])} matches "
          f"({len(d['wc_played'])} played, {len(d['wc_remaining'])} to play)")
    print(f"\nReconstructed groups ({len(d['groups'])}):")
    for label, teams in d["groups"].items():
        print(f"  {label}: {', '.join(teams)}")
