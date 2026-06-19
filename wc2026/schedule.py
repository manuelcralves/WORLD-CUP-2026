"""Official kickoff times for the 2026 World Cup group stage.

Times come from the published FIFA schedule (kickoffs given in GMT/UTC). Portugal
is on Western European Summer Time (WEST = UTC+1) throughout June-July 2026, so
the Lisbon kickoff is simply the UTC time + 1 hour. Matches are keyed by the
unordered pair of teams (each group pairing is unique), using the team names as
they appear in the dataset.
"""
from __future__ import annotations

import pandas as pd

# Source spelling -> dataset spelling.
_FIX = {"Czechia": "Czech Republic", "USA": "United States",
        "Turkiye": "Turkey", "Curacao": "Curaçao"}

# date(UTC) time(UTC) Home vs Away  — full 72-match group stage, in UTC.
_RAW = """
2026-06-11 19:00 Mexico vs South Africa
2026-06-12 02:00 South Korea vs Czechia
2026-06-12 19:00 Canada vs Bosnia and Herzegovina
2026-06-13 01:00 USA vs Paraguay
2026-06-13 19:00 Qatar vs Switzerland
2026-06-13 22:00 Brazil vs Morocco
2026-06-14 01:00 Haiti vs Scotland
2026-06-14 04:00 Australia vs Turkiye
2026-06-14 17:00 Germany vs Curacao
2026-06-14 20:00 Netherlands vs Japan
2026-06-14 23:00 Ivory Coast vs Ecuador
2026-06-15 02:00 Sweden vs Tunisia
2026-06-15 16:00 Spain vs Cape Verde
2026-06-15 19:00 Belgium vs Egypt
2026-06-15 22:00 Saudi Arabia vs Uruguay
2026-06-16 01:00 Iran vs New Zealand
2026-06-16 19:00 France vs Senegal
2026-06-16 22:00 Iraq vs Norway
2026-06-17 01:00 Argentina vs Algeria
2026-06-17 04:00 Austria vs Jordan
2026-06-17 17:00 Portugal vs DR Congo
2026-06-17 20:00 England vs Croatia
2026-06-17 23:00 Ghana vs Panama
2026-06-18 02:00 Uzbekistan vs Colombia
2026-06-18 16:00 Czechia vs South Africa
2026-06-18 19:00 Switzerland vs Bosnia and Herzegovina
2026-06-18 22:00 Canada vs Qatar
2026-06-19 01:00 Mexico vs South Korea
2026-06-19 22:00 Scotland vs Morocco
2026-06-19 19:00 USA vs Australia
2026-06-20 00:30 Brazil vs Haiti
2026-06-20 03:00 Turkiye vs Paraguay
2026-06-20 17:00 Netherlands vs Sweden
2026-06-20 20:00 Germany vs Ivory Coast
2026-06-21 00:00 Ecuador vs Curacao
2026-06-21 04:00 Tunisia vs Japan
2026-06-21 16:00 Spain vs Saudi Arabia
2026-06-21 19:00 Belgium vs Iran
2026-06-21 22:00 Uruguay vs Cape Verde
2026-06-22 01:00 New Zealand vs Egypt
2026-06-22 17:00 Argentina vs Austria
2026-06-22 21:00 France vs Iraq
2026-06-23 00:00 Norway vs Senegal
2026-06-23 03:00 Jordan vs Algeria
2026-06-23 17:00 Portugal vs Uzbekistan
2026-06-23 20:00 England vs Ghana
2026-06-23 23:00 Panama vs Croatia
2026-06-24 02:00 Colombia vs DR Congo
2026-06-24 19:00 Switzerland vs Canada
2026-06-24 19:00 Bosnia and Herzegovina vs Qatar
2026-06-24 22:00 Scotland vs Brazil
2026-06-24 22:00 Morocco vs Haiti
2026-06-25 01:00 Czechia vs Mexico
2026-06-25 01:00 South Africa vs South Korea
2026-06-25 20:00 Ecuador vs Germany
2026-06-25 20:00 Curacao vs Ivory Coast
2026-06-25 23:00 Japan vs Sweden
2026-06-25 23:00 Tunisia vs Netherlands
2026-06-26 02:00 Turkiye vs USA
2026-06-26 02:00 Paraguay vs Australia
2026-06-26 19:00 Norway vs France
2026-06-26 19:00 Senegal vs Iraq
2026-06-27 00:00 Cape Verde vs Saudi Arabia
2026-06-27 00:00 Uruguay vs Spain
2026-06-27 03:00 Egypt vs Iran
2026-06-27 03:00 New Zealand vs Belgium
2026-06-27 21:00 Panama vs England
2026-06-27 21:00 Croatia vs Ghana
2026-06-27 23:30 Colombia vs Portugal
2026-06-27 23:30 DR Congo vs Uzbekistan
2026-06-28 02:00 Algeria vs Austria
2026-06-28 02:00 Jordan vs Argentina
"""

_WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MO = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
       "Nov", "Dec"]


def _norm(t: str) -> str:
    return _FIX.get(t.strip(), t.strip())


def _key(a: str, b: str) -> str:
    return "|".join(sorted([_norm(a), _norm(b)]))


KICKOFFS_UTC: dict[str, str] = {}
for _ln in _RAW.strip().splitlines():
    _d, _t, _rest = _ln.split(" ", 2)
    _a, _b = _rest.split(" vs ")
    KICKOFFS_UTC[_key(_a, _b)] = f"{_d}T{_t}"


def lisbon(home: str, away: str) -> dict | None:
    """Lisbon-time kickoff info for a fixture, or None if not in the schedule."""
    iso = KICKOFFS_UTC.get("|".join(sorted([home, away])))
    if not iso:
        return None
    t = pd.Timestamp(iso) + pd.Timedelta(hours=1)  # WEST = UTC+1
    return {"hm": t.strftime("%H:%M"), "date": t.strftime("%Y-%m-%d"),
            "label": f"{_WD[t.weekday()]} {t.day} {_MO[t.month]} · {t.strftime('%H:%M')}"}


def all_lisbon() -> dict:
    """{sorted-pair-key: {hm, label}} for every scheduled fixture (for the dashboard)."""
    out = {}
    for key in KICKOFFS_UTC:
        a, b = key.split("|")
        out[key] = lisbon(a, b)
    return out


if __name__ == "__main__":
    from . import data as D
    b = D.load_all()
    miss = []
    for g in b["wc"].itertuples(index=False):
        if lisbon(g.home_team, g.away_team) is None:
            miss.append((g.home_team, g.away_team))
    print(f"{len(KICKOFFS_UTC)} kickoffs loaded; {len(miss)} dataset fixtures unmatched")
    for m in miss:
        print("  MISSING:", m)
    print("Opener (Mexico vs South Africa):", lisbon("Mexico", "South Africa"))
    print("Portugal vs DR Congo:", lisbon("Portugal", "DR Congo"))
