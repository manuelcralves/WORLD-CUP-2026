"""Monte Carlo simulation of the 2026 World Cup.

Official structure (draw of 5 Dec 2025, matches 73-104):
 - 12 groups of 4 -> top 2 qualify + the 8 best third-placed teams (32 teams)
 - Round of 32 with the official eligibility lists for the third-placed teams
 - Bracket R32 -> R16 -> Quarter-finals -> Semi-finals -> Final

In each simulation:
 - the matches already played enter with their real result (fixed)
 - the remaining group-stage matches are sampled from the Poisson model
 - teams are ranked by the FIFA criteria (points; head-to-head pts/GD/goals among
   the tied teams; overall GD; overall goals; then the FIFA world ranking) and the
   knockout rounds are simulated (with extra time and penalties).

Everything is vectorised in numpy: the N simulations run in parallel round by round.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import data as D
from . import fifa as FIFA

HOSTS = D.HOSTS

# Official groups (with the names exactly as they appear in the dataset). Validated
# at runtime against the reconstruction built from the schedule.
OFFICIAL_GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Round of 32 (matches 73-88). Each side is ("W", group), ("RU", group) or
# ("3", match number) for one of the 8 best third-placed teams.
R32 = {
    73: (("RU", "A"), ("RU", "B")),
    74: (("W", "E"), ("3", 74)),
    75: (("W", "F"), ("RU", "C")),
    76: (("W", "C"), ("RU", "F")),
    77: (("W", "I"), ("3", 77)),
    78: (("RU", "E"), ("RU", "I")),
    79: (("W", "A"), ("3", 79)),
    80: (("W", "L"), ("3", 80)),
    81: (("W", "D"), ("3", 81)),
    82: (("W", "G"), ("3", 82)),
    83: (("RU", "K"), ("RU", "L")),
    84: (("W", "H"), ("RU", "J")),
    85: (("W", "B"), ("3", 85)),
    86: (("W", "J"), ("RU", "H")),
    87: (("W", "K"), ("3", 87)),
    88: (("RU", "D"), ("RU", "G")),
}

# Groups eligible to supply the third-placed team for each Round of 32 match.
THIRD_ELIGIBLE = {
    74: "ABCDF", 77: "CDFGH", 79: "CEFHI", 80: "EHIJK",
    81: "BEFIJ", 82: "AEHIJ", 85: "EFGIJ", 87: "DEIJL",
}

# Several eligibility-valid assignments can exist, so the bipartite matching in
# _match_thirds doesn't always reproduce FIFA's official allocation. The actual
# tournament's third-place combination is pinned here to the real R32 draw; any
# other combination (pre-tournament / hypothetical) falls back to the matching.
THIRD_ASSIGNMENT = {
    frozenset("BDEFIJKL"): {74: "D", 77: "F", 79: "E", 80: "K",
                            81: "B", 82: "I", 85: "J", 87: "L"},
}

# Bracket from the Round of 16 onwards (match -> (source_match_1, source_match_2)).
LATER = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100),
    104: (101, 102),
}

STAGES = ["p_ko", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
STAGE_LABELS = {
    "p_ko": "Last 32", "p_r16": "Round of 16", "p_qf": "Quarter-finals",
    "p_sf": "Semi-finals", "p_final": "Final", "p_champion": "Champion",
}


# --------------------------------------------------------------------------- #
def _lambda_matrices(teams, model, state, default, host_adv=0.5):
    """LH[i,j], LA[i,j] = expected goals of i and of j (i treated as "home").

    Used only in the knockout rounds. Home advantage applies if exactly one of
    the teams is a host, but reduced (`host_adv`, half by default): the knockout
    rounds are played in more neutral stadiums than the group-stage home games.
    """
    n = len(teams)
    LH = np.zeros((n, n))
    LA = np.zeros((n, n))
    for i, a in enumerate(teams):
        sa = state.get(a, default)
        for j, b in enumerate(teams):
            if i == j:
                continue
            sb = state.get(b, default)
            if (a in HOSTS) and (b not in HOSTS):
                va = host_adv
            elif (b in HOSTS) and (a not in HOSTS):
                va = -host_adv
            else:
                va = 0.0
            rows = pd.DataFrame([
                {"elo_self": sa["elo"], "elo_opp": sb["elo"], "venue": va,
                 "is_friendly": 0.0, "self_gf": sa["gf_form"], "opp_ga": sb["ga_form"]},
                {"elo_self": sb["elo"], "elo_opp": sa["elo"], "venue": -va,
                 "is_friendly": 0.0, "self_gf": sb["gf_form"], "opp_ga": sa["ga_form"]},
            ])
            lam = model.predict_lambda(rows)
            LH[i, j], LA[i, j] = lam[0], lam[1]
    return LH, LA


def _match_thirds(letters_q, slots):
    """Assigns each qualified group (third place) to an eligible R32 match.

    The eligibility lists alone don't uniquely determine FIFA's choice (several
    assignments can be valid), so a known official combination is pinned in
    THIRD_ASSIGNMENT; otherwise we solve the bipartite matching respecting the
    eligibility lists (avoids group rematches). Returns {match_number: group}.
    """
    pin = THIRD_ASSIGNMENT.get(frozenset(letters_q))
    if pin is not None:
        return {s: pin[s] for s in slots}
    C = np.ones((8, 8)) * 1000.0
    for i, s in enumerate(slots):
        for j, L in enumerate(letters_q):
            if L in THIRD_ELIGIBLE[s]:
                C[i, j] = 0.0
    ri, ci = linear_sum_assignment(C)
    return {slots[ri[k]]: letters_q[ci[k]] for k in range(len(ri))}


def _assign_thirds(qual, thirds, letters, N):
    """For each simulation, maps the 8 third-place slots -> concrete team."""
    slots = [74, 77, 79, 80, 81, 82, 85, 87]
    slot_team = {s: np.zeros(N, dtype=int) for s in slots}
    bits = (1 << np.arange(12))
    masks = (qual.T * bits).sum(1)  # bitmask of the set of qualified groups

    cache = {}
    for mk in np.unique(masks):
        letters_q = [letters[i] for i in range(12) if (mk >> i) & 1]
        cache[mk] = _match_thirds(letters_q, slots)

    for mk in np.unique(masks):
        sims = np.where(masks == mk)[0]
        for s, L in cache[mk].items():
            slot_team[s][sims] = thirds[L][sims]
    return slot_team


def simulate(bundle: dict, trained: dict, n_sims: int = 20000, seed: int = 42,
             fairplay: dict = None):
    """Runs N simulations and returns the probability table per team.

    Also records, for every team, the distribution of its knockout opponents
    per round (in table.attrs["opp_matrix"]).
    """
    rng = np.random.default_rng(seed)
    model, state, default = trained["model"], trained["state"], trained["default"]
    # penalty shootout model (data-driven): P(A wins) = sigmoid(b1 * Elo_diff)
    shoot_b1 = trained.get("shootout", {}).get("b1", 0.0)
    groups = OFFICIAL_GROUPS
    N = n_sims

    # validate the official groups against the reconstruction from the data
    recon = {frozenset(v) for v in bundle["groups"].values()}
    for L, ts in groups.items():
        assert frozenset(ts) in recon, f"Group {L} does not match the schedule"

    teams = sorted({t for ts in groups.values() for t in ts})
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    elo_arr = np.array([state.get(t, default)["elo"] for t in teams])
    fifa_rk_arr = np.array([FIFA.rank_of(t) or 999 for t in teams], dtype=float)   # FIFA world rank — final group tiebreaker
    fp_arr = np.array([(fairplay or {}).get(t, 0) for t in teams], dtype=float)   # FIFA fair-play points (cards), static from real results
    LH, LA = _lambda_matrices(teams, model, state, default)
    team_letter = {t: L for L, ts in groups.items() for t in ts}

    # ---- group-stage matches (goals per simulation) --------------------- #
    group_games = {L: [] for L in groups}
    for r in bundle["wc_played"].itertuples(index=False):
        gh = np.full(N, int(r.home_score)); ga = np.full(N, int(r.away_score))
        group_games[team_letter[r.home_team]].append((r.home_team, r.away_team, gh, ga))
    for r in bundle["wc_remaining"].itertuples(index=False):
        lh, la = model.lambdas_for(state, default, r.home_team, r.away_team, bool(r.neutral))
        gh = rng.poisson(lh, N); ga = rng.poisson(la, N)
        group_games[team_letter[r.home_team]].append((r.home_team, r.away_team, gh, ga))

    # ---- standings of each group ---------------------------------------- #
    winners, runners, thirds = {}, {}, {}
    third_pts, third_gd, third_gf, third_fr, third_fp = {}, {}, {}, {}, {}
    pos = np.zeros((n, 4))   # count of 1st/2nd/3rd/4th per team
    exp_pts = np.zeros(n)    # accumulated points (for the expected average)
    col = np.arange(N)
    for L, ts in groups.items():
        gidx = {t: k for k, t in enumerate(ts)}
        gti = np.array([ti[t] for t in ts])
        pts = np.zeros((4, N)); gd = np.zeros((4, N)); gf = np.zeros((4, N))
        for h, a, gh, ga in group_games[L]:
            hi, ai = gidx[h], gidx[a]
            hw = gh > ga; dr = gh == ga; aw = gh < ga
            pts[hi] += 3 * hw + dr; pts[ai] += 3 * aw + dr
            gd[hi] += gh - ga; gd[ai] += ga - gh
            gf[hi] += gh; gf[ai] += ga
        # head-to-head (FIFA criterion 2): pts/GD/goals counting ONLY the matches
        # between teams level on points (single pass over each tie set).
        h2p = np.zeros((4, N)); h2d = np.zeros((4, N)); h2f = np.zeros((4, N))
        for h, a, gh, ga in group_games[L]:
            hi, ai = gidx[h], gidx[a]
            tied = pts[hi] == pts[ai]
            hw = gh > ga; dr = gh == ga; aw = gh < ga
            h2p[hi] += (3 * hw + dr) * tied; h2p[ai] += (3 * aw + dr) * tied
            h2d[hi] += (gh - ga) * tied;     h2d[ai] += (ga - gh) * tied
            h2f[hi] += gh * tied;            h2f[ai] += ga * tied
        # FIFA order: points > head-to-head (pts, GD, goals) > overall GD >
        # overall goals > FIFA world ranking (replaces fair play + drawing of lots).
        fr = np.broadcast_to(fifa_rk_arr[gti].reshape(4, 1), (4, N))
        fp = np.broadcast_to(fp_arr[gti].reshape(4, 1), (4, N))   # fair-play (cards): fewer = higher
        order = np.lexsort((fr, -fp, -gf, -gd, -h2f, -h2d, -h2p, -pts), axis=0)
        winners[L] = gti[order[0]]
        runners[L] = gti[order[1]]
        thirds[L] = gti[order[2]]
        third_pts[L] = pts[order[2], col]
        third_gd[L] = gd[order[2], col]
        third_gf[L] = gf[order[2], col]
        third_fr[L] = fifa_rk_arr[gti[order[2]]]
        third_fp[L] = fp_arr[gti[order[2]]]
        for p_ in range(4):
            np.add.at(pos[:, p_], gti[order[p_]], 1)
        for k in range(4):
            exp_pts[gti[k]] += pts[k].sum()

    # ---- 8 best third-placed teams -------------------------------------- #
    letters = list(groups.keys())
    TP = np.vstack([third_pts[L] for L in letters])
    TG = np.vstack([third_gd[L] for L in letters])
    TF = np.vstack([third_gf[L] for L in letters])
    TR = np.vstack([third_fr[L] for L in letters])
    TFP = np.vstack([third_fp[L] for L in letters])
    # best thirds: points > overall GD > overall goals > fair-play (cards) > FIFA rank
    # (no head-to-head — third-placed teams come from different groups and never met).
    torder = np.lexsort((TR, -TFP, -TF, -TG, -TP), axis=0)
    qual = np.zeros((12, N), dtype=bool)
    for r in range(8):
        qual[torder[r], col] = True
    slot_team = _assign_thirds(qual, thirds, letters, N)

    # ---- knockout rounds ------------------------------------------------- #
    def resolve(spec):
        typ, key = spec
        if typ == "W":
            return winners[key]
        if typ == "RU":
            return runners[key]
        return slot_team[key]

    def play(A, B):
        """Returns the winner (indices) of each simulation of the A vs B tie."""
        lamA, lamB = LH[A, B], LA[A, B]
        ga = rng.poisson(lamA); gb = rng.poisson(lamB)
        winA = ga > gb
        tie = ga == gb
        if tie.any():
            idx = np.where(tie)[0]
            # extra time (30 min ~ 1/3 of regulation time)
            ea = rng.poisson(lamA[idx] * (30 / 90))
            eb = rng.poisson(lamB[idx] * (30 / 90))
            a2, b2 = ga[idx] + ea, gb[idx] + eb
            wt = a2 > b2
            still = a2 == b2
            # penalties: model fitted on the 677 shootouts (almost 50/50)
            z = shoot_b1 * (elo_arr[A[idx]] - elo_arr[B[idx]])
            pa = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            pen = rng.random(idx.size) < pa
            wt = np.where(still, pen, wt)
            winA[idx] = wt
        return np.where(winA, A, B)

    results = {}
    for mno, (sa, sb) in R32.items():
        results[mno] = play(resolve(sa), resolve(sb))
    for mno, (m1, m2) in LATER.items():
        results[mno] = play(results[m1], results[m2])

    def participants(mno):
        if mno in R32:
            sa, sb = R32[mno]
            return resolve(sa), resolve(sb)
        m1, m2 = LATER[mno]
        return results[m1], results[m2]

    rounds = {"R32": list(R32), "R16": list(range(89, 97)),
              "QF": list(range(97, 101)), "SF": [101, 102], "F": [104]}
    reach = {r: np.zeros(n) for r in rounds}
    # opp_mat[round][i, j] = number of sims team i faced team j in that round
    opp_mat = {r: np.zeros((n, n)) for r in rounds}
    for rname, mnos in rounds.items():
        for mno in mnos:
            A, B = participants(mno)
            np.add.at(reach[rname], A, 1)
            np.add.at(reach[rname], B, 1)
            np.add.at(opp_mat[rname], (A, B), 1)
            np.add.at(opp_mat[rname], (B, A), 1)
    champ = np.zeros(n)
    np.add.at(champ, results[104], 1)

    table = pd.DataFrame({
        "team": teams,
        "group": [team_letter[t] for t in teams],
        "elo": elo_arr.round(0),
        "p_win_group": pos[:, 0] / N,
        "p_1st": pos[:, 0] / N, "p_2nd": pos[:, 1] / N,
        "p_3rd": pos[:, 2] / N, "p_4th": pos[:, 3] / N,
        "exp_points": (exp_pts / N).round(2),
        "p_ko": reach["R32"] / N,
        "p_r16": reach["R16"] / N,
        "p_qf": reach["QF"] / N,
        "p_sf": reach["SF"] / N,
        "p_final": reach["F"] / N,
        "p_champion": champ / N,
    }).sort_values("p_champion", ascending=False).reset_index(drop=True)
    table.attrs["n_sims"] = N
    table.attrs["seed"] = seed
    table.attrs["teams"] = teams
    table.attrs["opp_matrix"] = opp_mat
    table.attrs["reach_counts"] = {r: pd.Series(reach[r], index=teams)
                                   for r in rounds}
    return table


if __name__ == "__main__":
    from . import model as M

    b = D.load_all()
    trained = M.train_full(b)
    tab = simulate(b, trained, n_sims=10000, seed=42)
    pd.set_option("display.width", 150)
    show = tab.copy()
    for c in ["p_win_group", "p_ko", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]:
        show[c] = (show[c] * 100).round(1)
    print("=== Top 16 title contenders ===")
    print(show.head(16).to_string(index=False))
    print("\n=== Favourite ===")
    print(show.head(1).to_string(index=False))
