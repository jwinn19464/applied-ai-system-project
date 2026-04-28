"""
Evaluation / Test Harness

Runs the music recommender on a fixed set of predefined test cases and
prints a structured summary of pass/fail results, confidence ratings,
fallback rates, and specialist style divergence scores.

Run with:
    python eval.py                  # full evaluation (loads all models)
    python eval.py --fast           # skip specialist model, faster run
    python eval.py --verbose        # show agent traces for each case

Exit code 0 if all required cases pass, 1 otherwise.
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from recommender import load_songs, Recommender, Song, UserProfile
from rag import build_default_store
from agent import RecommenderAgent, print_agent_trace

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "starter_pop",
        "desc": "Classic pop/happy profile — clear catalog match",
        "user": UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False, target_danceability=0.7,
        ),
        "expect_genre": "pop",
        "expect_min_confidence": 0.55,
        "expect_fallback": False,
    },
    {
        "name": "lofi_study",
        "desc": "Lofi/chill profile — niche genre, should lock in",
        "user": UserProfile(
            favorite_genre="lofi", favorite_mood="chill",
            target_energy=0.4, likes_acoustic=True,
        ),
        "expect_genre": "lofi",
        "expect_min_confidence": 0.40,
        "expect_fallback": False,
    },
    {
        "name": "high_energy_metal",
        "desc": "Extreme energy, metal genre",
        "user": UserProfile(
            favorite_genre="metal", favorite_mood="raging",
            target_energy=0.95, likes_acoustic=False,
        ),
        "expect_genre": "metal",
        "expect_min_confidence": 0.30,
        "expect_fallback": None,    # don't assert fallback either way
    },
    {
        "name": "classical_peaceful",
        "desc": "Classical / peaceful — acoustic seeker, very low energy",
        "user": UserProfile(
            favorite_genre="classical", favorite_mood="peaceful",
            target_energy=0.2, likes_acoustic=True,
        ),
        "expect_genre": "classical",
        "expect_min_confidence": 0.30,
        "expect_fallback": None,
    },
    {
        "name": "acoustic_dance_conflict",
        "desc": "Contradictory prefs: acoustic + very high danceability — should not crash",
        "user": UserProfile(
            favorite_genre="electronic", favorite_mood="happy",
            target_energy=0.15, likes_acoustic=True, target_danceability=0.95,
        ),
        "expect_genre": None,       # don't assert genre — conflict makes it unpredictable
        "expect_min_confidence": 0.0,
        "expect_fallback": None,
    },
    {
        "name": "high_energy_sad",
        "desc": "Unusual combo: pop + sad + high energy",
        "user": UserProfile(
            favorite_genre="pop", favorite_mood="sad",
            target_energy=0.9, likes_acoustic=False,
        ),
        "expect_genre": "pop",
        "expect_min_confidence": 0.50,
        "expect_fallback": False,
    },
    {
        "name": "rap_zero_energy",
        "desc": "Rare mood + zero energy — likely triggers fallback",
        "user": UserProfile(
            favorite_genre="rap", favorite_mood="hateful",
            target_energy=0.0, likes_acoustic=False,
        ),
        "expect_genre": None,
        "expect_min_confidence": 0.0,
        "expect_fallback": None,
    },
    {
        "name": "jazz_romantic",
        "desc": "Jazz + romantic — niche genre seeker",
        "user": UserProfile(
            favorite_genre="jazz", favorite_mood="romantic",
            target_energy=0.45, likes_acoustic=True,
        ),
        "expect_genre": "jazz",
        "expect_min_confidence": 0.20,
        "expect_fallback": None,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_case(case: dict, recommender: Recommender, agent: RecommenderAgent, verbose: bool) -> dict:
    user = case["user"]
    songs = recommender.songs

    t0 = time.time()
    agent_result = agent.run(user, songs, k=5)
    elapsed = time.time() - t0

    if verbose:
        print(f"\n{'='*60}\nCase: {case['name']}")
        print_agent_trace(agent_result)

    checks = {}
    top_song = agent_result.songs[0] if agent_result.songs else None

    # Check 1: top genre matches expectation
    if case["expect_genre"] is not None and top_song is not None:
        checks["genre_match"] = top_song.genre.lower() == case["expect_genre"].lower()
    else:
        checks["genre_match"] = None   # not asserted

    # Check 2: confidence meets threshold
    checks["confidence_ok"] = agent_result.confidence >= case["expect_min_confidence"]

    # Check 3: fallback expectation (if specified)
    if case["expect_fallback"] is not None:
        checks["fallback_ok"] = agent_result.used_fallback == case["expect_fallback"]
    else:
        checks["fallback_ok"] = None   # not asserted

    # Check 4: always — at least k results returned, explanation non-empty
    checks["has_results"] = len(agent_result.songs) > 0
    checks["has_explanation"] = bool(agent_result.explanation.strip())

    required = ["confidence_ok", "has_results", "has_explanation"]
    optional = ["genre_match", "fallback_ok"]
    all_required_pass = all(checks[k] for k in required)
    all_asserted_pass = all_required_pass and all(
        checks[k] for k in optional if checks[k] is not None
    )

    return {
        "name": case["name"],
        "desc": case["desc"],
        "pass": all_asserted_pass,
        "required_pass": all_required_pass,
        "checks": checks,
        "confidence": agent_result.confidence,
        "used_fallback": agent_result.used_fallback,
        "top_song": top_song.title if top_song else "—",
        "top_genre": top_song.genre if top_song else "—",
        "elapsed_s": round(elapsed, 3),
    }


def run_specialist_eval(recommender: Recommender) -> dict:
    """
    Compare baseline vs. specialist explanations on 3 songs.
    Returns style divergence scores to demonstrate measurable difference.
    """
    from specialist import FewShotExplainer, style_score

    explainer = FewShotExplainer()
    songs_to_test = recommender.songs[:3]
    dummy_reasons = ["genre match", "energy close"]

    rows = []
    for song in songs_to_test:
        baseline = explainer.baseline(song, dummy_reasons)
        hype_out  = explainer.explain(song, dummy_reasons, style="hype")
        study_out = explainer.explain(song, dummy_reasons, style="study")
        rows.append({
            "song": song.title,
            "baseline_hype_score":  style_score(baseline, "hype"),
            "specialist_hype_score": style_score(hype_out, "hype"),
            "baseline_study_score":  style_score(baseline, "study"),
            "specialist_study_score": style_score(study_out, "study"),
            "hype_output": hype_out[:80],
            "study_output": study_out[:80],
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",    action="store_true", help="Skip specialist evaluation")
    parser.add_argument("--verbose", action="store_true", help="Print agent trace per case")
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    songs_path = os.path.join(base, "data", "songs.csv")
    docs_dir   = os.path.join(base, "data", "docs")

    print("=" * 65)
    print("  MUSIC RECOMMENDER — EVALUATION HARNESS")
    print("=" * 65)

    # Setup
    raw_songs = load_songs(songs_path)
    songs = [Song(**s) for s in raw_songs]
    recommender = Recommender(songs)

    store = build_default_store(docs_dir)
    agent = RecommenderAgent(store=store)

    # ── Recommender + Agent evaluation ─────────────────────────────────────
    print(f"\nRunning {len(TEST_CASES)} test cases...\n")
    results = [run_case(c, recommender, agent, verbose=args.verbose) for c in TEST_CASES]

    # Summary table
    col = "{:<25} {:<8} {:<10} {:<10} {:<8} {:<25}"
    print(col.format("CASE", "PASS", "CONFIDENCE", "FALLBACK", "TIME(s)", "TOP SONG"))
    print("-" * 90)
    for r in results:
        status = "PASS" if r["pass"] else ("WARN" if r["required_pass"] else "FAIL")
        print(col.format(
            r["name"][:24],
            status,
            f"{r['confidence']:.3f}",
            str(r["used_fallback"]),
            str(r["elapsed_s"]),
            r["top_song"][:24],
        ))

    # Aggregate metrics
    n_pass     = sum(1 for r in results if r["pass"])
    n_required = sum(1 for r in results if r["required_pass"])
    avg_conf   = sum(r["confidence"] for r in results) / len(results)
    fallback_rate = sum(1 for r in results if r["used_fallback"]) / len(results)

    print("-" * 90)
    print(f"\n  Total cases      : {len(results)}")
    print(f"  Full pass        : {n_pass}/{len(results)}")
    print(f"  Required pass    : {n_required}/{len(results)}")
    print(f"  Avg confidence   : {avg_conf:.3f}")
    print(f"  Fallback rate    : {fallback_rate:.1%}")

    # ── Specialist evaluation ───────────────────────────────────────────────
    if not args.fast:
        print("\n" + "=" * 65)
        print("  SPECIALIST STYLE EVALUATION (flan-t5-base few-shot)")
        print("=" * 65)
        spec_rows = run_specialist_eval(recommender)

        scol = "{:<28} {:<16} {:<16} {:<16} {:<16}"
        print("\n" + scol.format("SONG", "BASE→HYPE", "SPEC→HYPE", "BASE→STUDY", "SPEC→STUDY"))
        print("-" * 80)
        for row in spec_rows:
            print(scol.format(
                row["song"][:27],
                f"{row['baseline_hype_score']:.2f}",
                f"{row['specialist_hype_score']:.2f}",
                f"{row['baseline_study_score']:.2f}",
                f"{row['specialist_study_score']:.2f}",
            ))
        print()
        for row in spec_rows:
            print(f"  [{row['song']}]")
            print(f"    Hype  : {row['hype_output']}")
            print(f"    Study : {row['study_output']}")

        avg_hype_lift  = sum(r["specialist_hype_score"]  - r["baseline_hype_score"]  for r in spec_rows) / len(spec_rows)
        avg_study_lift = sum(r["specialist_study_score"] - r["baseline_study_score"] for r in spec_rows) / len(spec_rows)
        print(f"\n  Avg hype style lift  : +{avg_hype_lift:.3f}")
        print(f"  Avg study style lift : +{avg_study_lift:.3f}")
        print("  (Positive = specialist output is more on-style than baseline)")

    # Exit code
    all_required_pass = all(r["required_pass"] for r in results)
    print("\n" + ("EVALUATION PASSED" if all_required_pass else "EVALUATION FAILED"))
    sys.exit(0 if all_required_pass else 1)


if __name__ == "__main__":
    main()
