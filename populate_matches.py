"""Push the match table to Supabase, without running the full deploy (no sims,
no tweet, no site rebuild). Used by the manual 'Populate matches' GitHub Action;
needs SUPABASE_URL + SUPABASE_SERVICE_KEY in the environment (a no-op without them).
"""
from wc2026 import data as D, model as M, predictions as PR, schedule as SCH, supa as SUPA


def main():
    live = D.load_all()
    trained = M.train_full(live)
    pre = M.train_full(D.load_all(cutoff="2026-06-11"))      # blind model for played games
    data = {
        "matches": PR.match_predictions(live, trained, topn=3),
        "played_review": PR.played_review(live, pre),
        "kickoffs": SCH.all_lisbon(),
    }
    SUPA.push_matches(data)


if __name__ == "__main__":
    main()
