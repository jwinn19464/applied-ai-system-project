"""
Command line runner — demonstrates all four system features.

Modes
-----
(default)   Run preset profiles through the agent + show reasoning trace
--nlp       Accept natural language input and run through agent pipeline
--rag-demo  Show RAG retrieval results for sample queries
--fast      Skip model-loading steps (uses plain Recommender only)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from recommender import load_songs, Recommender, Song, UserProfile
from rag import build_default_store
from agent import RecommenderAgent, print_agent_trace

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
SONGS_PATH = os.path.join(BASE_DIR, "data", "songs.csv")
DOCS_DIR   = os.path.join(BASE_DIR, "data", "docs")

PRESET_PROFILES = [
    {
        "name": "starter",
        "user": UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False, target_danceability=0.7,
        ),
    },
    {
        "name": "high_energy_sad",
        "user": UserProfile(
            favorite_genre="pop", favorite_mood="sad",
            target_energy=0.9, likes_acoustic=False, target_danceability=0.2,
        ),
    },
    {
        "name": "acoustic_dance_conflict",
        "user": UserProfile(
            favorite_genre="electronic", favorite_mood="happy",
            target_energy=0.15, likes_acoustic=True, target_danceability=0.95,
        ),
    },
    {
        "name": "low_energy_hateful",
        "user": UserProfile(
            favorite_genre="rap", favorite_mood="hateful",
            target_energy=0.0, likes_acoustic=False,
        ),
    },
]


def run_preset_profiles(agent: RecommenderAgent, songs: list) -> None:
    print("\n=== PRESET PROFILES — Agent + RAG Pipeline ===\n")
    for variant in PRESET_PROFILES:
        print(f"\n{'='*60}\nProfile: {variant['name']}")
        u = variant["user"]
        print(f"  genre={u.favorite_genre}, mood={u.favorite_mood}, "
              f"energy={u.target_energy}, acoustic={u.likes_acoustic}")
        result = agent.run(u, songs, k=5)
        print_agent_trace(result)


def run_nlp_mode(agent: RecommenderAgent, songs: list) -> None:
    from nlp_parser import NLPProfileParser
    parser = NLPProfileParser()

    print("\nNatural language mode — describe the music you want.")
    print("Type 'quit' to exit.\n")

    while True:
        text = input("Your request: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        user = parser.parse(text)
        print(f"\nParsed: genre={user.favorite_genre}, mood={user.favorite_mood}, "
              f"energy={user.target_energy:.2f}, acoustic={user.likes_acoustic}")

        result = agent.run(user, songs, k=5)
        print_agent_trace(result)


def run_rag_demo(store) -> None:
    print("\n=== RAG RETRIEVAL DEMO ===\n")
    queries = [
        "upbeat pop for a workout",
        "something quiet and acoustic for studying",
        "aggressive metal with extreme energy",
        "nostalgic electronic night drive",
    ]
    for q in queries:
        print(f"Query: '{q}'")
        hits = store.retrieve(q, top_k=2)
        for score, chunk in hits:
            print(f"  [{score:.3f}] {chunk.label} ({chunk.source}) — {chunk.text[:100]}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp",      action="store_true")
    parser.add_argument("--rag-demo", action="store_true")
    parser.add_argument("--fast",     action="store_true", help="Skip model loading")
    args = parser.parse_args()

    raw_songs = load_songs(SONGS_PATH)
    songs = [Song(**s) for s in raw_songs]
    recommender = Recommender(songs)

    if args.fast:
        # Plain recommender, no models
        print("\n=== FAST MODE — plain Recommender (no models) ===")
        for v in PRESET_PROFILES:
            result = recommender.recommend(v["user"], k=3)
            print(f"\n{v['name']}: {result.quality_note}")
            for s in result.songs:
                print(f"  {s.title} [{s.genre}/{s.mood}]")
        return

    store = build_default_store(DOCS_DIR)
    agent = RecommenderAgent(store=store)

    if args.rag_demo:
        run_rag_demo(store)
    elif args.nlp:
        run_nlp_mode(agent, songs)
    else:
        run_preset_profiles(agent, songs)


if __name__ == "__main__":
    main()
