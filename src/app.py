"""
Streamlit web app — Music Recommender AI Demo

Tabs
----
1  Try It          End-to-end: enter a profile, get recommendations + agent trace
2  RAG Retrieval   See what genre/mood knowledge is retrieved for any query
3  Reliability     Run the eval harness (8 test cases) and show pass/fail table
4  Specialist      Few-shot style demo: baseline vs hype vs study explanation
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from recommender import load_songs, Song, UserProfile, Recommender
from agent import RecommenderAgent, print_agent_trace
from rag import build_default_store

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
SONGS_PATH = os.path.join(BASE_DIR, "data", "songs.csv")
DOCS_DIR   = os.path.join(BASE_DIR, "data", "docs")

# ---------------------------------------------------------------------------
# Cached resource loading (only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading song catalog…")
def get_songs():
    raw = load_songs(SONGS_PATH)
    return [Song(**s) for s in raw]

@st.cache_resource(show_spinner="Loading embedding model & documents…")
def get_store():
    return build_default_store(DOCS_DIR)

@st.cache_resource(show_spinner="Initialising agent…")
def get_agent():
    return RecommenderAgent(store=get_store())

@st.cache_resource(show_spinner="Loading specialist model (flan-t5-base)…")
def get_explainer():
    from specialist import FewShotExplainer
    return FewShotExplainer()


def format_genres(genre: str) -> str:
    if not genre:
        return ""
    if "," in genre:
        return ", ".join(g.strip() for g in genre.split(","))
    if "/" in genre:
        return ", ".join(g.strip() for g in genre.split("/"))
    return genre


def build_song_document(songs: list[Song]) -> str:
    lines = [
        "# Song Catalog",
        "",
        "Detailed song metadata for every track in the catalog.",
        "",
    ]
    for song in songs:
        lines.extend([
            f"## {song.title} — {song.artist}",
            "",
            f"- **ID:** {song.id}",
            f"- **Genres:** {format_genres(song.genre)}",
            f"- **Mood:** {song.mood}",
            f"- **Energy:** {song.energy:.2f}",
            f"- **Tempo (BPM):** {song.tempo_bpm:.0f}",
            f"- **Valence:** {song.valence:.2f}",
            f"- **Danceability:** {song.danceability:.2f}",
            f"- **Acousticness:** {song.acousticness:.2f}",
            "",
        ])
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Music Recommender AI", page_icon="🎵", layout="wide")
st.title("🎵 Music Recommender AI")
st.caption("Agentic pipeline · RAG · Few-shot specialist · Reliability eval")

tab_try, tab_library, tab_eval = st.tabs(
    ["🎯 Try It", "📖 Song Library", "✅ Reliability"]
)

# ===========================================================================
# Tab 1 — Try It
# ===========================================================================

with tab_try:
    st.header("End-to-End Recommendation")
    st.write(
        "Fill in a listener profile, choose a retrieval mode, then click **Recommend**. "
        "The agent runs its six-step reasoning chain and shows you exactly why each song was chosen."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        genre = st.selectbox("Favorite genre", ["pop", "rock", "lofi", "electronic", "rap", "jazz", "classical", "country", "metal", "ambient", "synthwave", "J-pop"])
        mood  = st.selectbox("Favorite mood",  ["happy", "sad", "chill", "intense", "romantic", "energetic", "melancholy", "peaceful", "hateful", "raging"])
        energy = st.slider("Target energy (0 = calm, 1 = extreme)", 0.0, 1.0, 0.7, 0.05)
        likes_acoustic = st.checkbox("Prefers acoustic music")
        use_danceability = st.checkbox("Set danceability target")
        danceability = st.slider("Target danceability", 0.0, 1.0, 0.6, 0.05) if use_danceability else None
        k = st.slider("Number of recommendations", 1, 10, 5)

        st.divider()
        rag_mode = st.radio(
            "Retrieval mode",
            options=["No RAG", "Simple RAG", "Enhanced RAG"],
            index=0,
            help=(
                "**No RAG** — agent scores songs using attribute weights only, no external knowledge.\n\n"
                "**Simple RAG** — retrieves 1 document chunk to enrich the explanation.\n\n"
                "**Enhanced RAG** — retrieves 3 chunks from multiple sources (genres + moods) for a richer explanation."
            ),
        )

        style_mode = st.radio(
            "Explanation style",
            options=["Default", "🔥 Hype", "📋 Formal"],
            index=0,
            help=(
                "**Default** — score-based explanation from the agent.\n\n"
                "**Hype** — few-shot specialist rewrites the explanation in an energetic DJ voice.\n\n"
                "**Formal** — few-shot specialist rewrites it in a professional, factual tone."
            ),
        )

        run_btn = st.button("Recommend", type="primary")

    with col_right:
        if run_btn:
            songs = get_songs()
            user  = UserProfile(
                favorite_genre=genre,
                favorite_mood=mood,
                target_energy=energy,
                likes_acoustic=likes_acoustic,
                target_danceability=danceability,
            )

            if rag_mode == "No RAG":
                from agent import RecommenderAgent as _Agent
                run_agent = _Agent(store=None)
                rag_top_k = 0
            elif rag_mode == "Simple RAG":
                run_agent = get_agent()
                rag_top_k = 1
            else:  # Enhanced RAG
                run_agent = get_agent()
                rag_top_k = 3

            mode_label = {"No RAG": "⬜ No RAG", "Simple RAG": "🔍 Simple RAG", "Enhanced RAG": "🚀 Enhanced RAG"}[rag_mode]
            st.caption(f"Running with **{mode_label}**")

            with st.spinner("Agent reasoning…"):
                result = run_agent.run(user, songs, k=k, rag_top_k=rag_top_k)

            # ── Confidence badge ────────────────────────────────────────────
            conf_color = "green" if result.confidence >= 0.5 else ("orange" if result.confidence >= 0.25 else "red")
            st.markdown(
                f"**Confidence:** :{conf_color}[{result.confidence:.2f}]"
                + ("  ⚠️ *fallback used*" if result.used_fallback else "")
            )

            # ── Top recommendations ─────────────────────────────────────────
            st.subheader("Top Recommendations")
            for i, song in enumerate(result.songs, 1):
                st.markdown(f"**{i}. {song.title}** — {song.artist}  \n"
                            f"`{song.genre}` / `{song.mood}` · energy {song.energy:.2f} · BPM {song.tempo_bpm:.0f}")

            # ── Explanation ─────────────────────────────────────────────────
            st.subheader("Why the top pick?")

            top_song = result.songs[0]
            top_reasons = result.steps[2].detail if len(result.steps) > 2 else ""

            if style_mode != "Default":
                from specialist import style_score
                style_key = "hype" if "Hype" in style_mode else "formal"
                with st.spinner(f"Generating {style_mode} explanation…"):
                    explainer = get_explainer()
                    # extract reason strings from the scoring step detail
                    raw_reasons = [r.strip() for r in top_reasons.replace("Top 3:", "").split(",") if "(" in r]
                    reason_labels = [r.split("(")[0].strip().strip("'") for r in raw_reasons] or ["genre match", "energy close"]
                    styled = explainer.explain(top_song, reason_labels, style=style_key)
                    baseline = explainer.baseline(top_song, reason_labels)

                st.info(styled)

                b_score = style_score(baseline, style_key)
                s_score = style_score(styled, style_key)
                delta = s_score - b_score
                st.caption(
                    f"Style score vs baseline — {style_mode}: **{s_score:.2f}** "
                    f"({'+'if delta >= 0 else ''}{delta:.2f} vs default {b_score:.2f}). "
                    "Higher = more on-style than the plain score-based explanation."
                )
            else:
                st.info(result.explanation)

            # ── RAG context callout ─────────────────────────────────────────
            if rag_mode != "No RAG":
                retrieval_step = next(
                    (s for s in result.steps if s.name == "Context Retrieval"), None
                )
                if retrieval_step and retrieval_step.detail:
                    with st.expander(f"📚 Retrieved context ({rag_mode})"):
                        st.markdown(retrieval_step.detail)

            # ── Agent reasoning trace ───────────────────────────────────────
            with st.expander("🔬 Agent Reasoning Trace (all 6 steps)"):
                for i, step in enumerate(result.steps, 1):
                    st.markdown(f"**Step {i} — {step.name}**")
                    st.markdown(f"- Decision: `{step.decision}`")
                    if step.detail:
                        st.markdown(f"- Detail: {step.detail[:300]}")
        else:
            st.info("Set a profile on the left, choose a retrieval mode, and click **Recommend**.")

with tab_library:
    st.header("Song Library")
    st.write(
        "Browse the full song catalog and download a Markdown document containing metadata for every track. "
        "This is useful when songs belong to multiple genres or when you want a complete catalog reference."
    )
    songs = get_songs()
    document = build_song_document(songs)
    st.download_button(
        "Download song info document",
        document,
        file_name="song_catalog.md",
        mime="text/markdown",
    )

    for song in songs:
        with st.expander(f"{song.title} — {song.artist}"):
            st.markdown(
                f"**Genres:** {format_genres(song.genre)}  \n"
                f"- **Mood:** {song.mood}  \n"
                f"- **Energy:** {song.energy:.2f}  \n"
                f"- **Tempo (BPM):** {song.tempo_bpm:.0f}  \n"
                f"- **Valence:** {song.valence:.2f}  \n"
                f"- **Danceability:** {song.danceability:.2f}  \n"
                f"- **Acousticness:** {song.acousticness:.2f}"
            )


# ===========================================================================
# Tab 3 — Reliability / Eval
# ===========================================================================

with tab_eval:
    st.header("Reliability Evaluation")
    st.write(
        "Runs the built-in test suite (8 fixed profiles) against the full agent pipeline. "
        "Each case checks: correct top genre, minimum confidence, fallback behaviour, "
        "non-empty results, and non-empty explanation."
    )

    if st.button("Run Evaluation", type="primary"):
        songs = get_songs()
        agent = get_agent()

        # Import inline so the heavy eval import only happens when needed
        import time
        sys.path.insert(0, os.path.join(BASE_DIR))
        from eval import TEST_CASES, run_case

        recommender = Recommender(songs)

        progress = st.progress(0, text="Running test cases…")
        rows = []
        for idx, case in enumerate(TEST_CASES):
            row = run_case(case, recommender, agent, verbose=False)
            rows.append(row)
            progress.progress((idx + 1) / len(TEST_CASES), text=f"Case {idx+1}/{len(TEST_CASES)}: {case['name']}")

        progress.empty()

        # ── Summary metrics ─────────────────────────────────────────────────
        n_pass     = sum(1 for r in rows if r["pass"])
        n_required = sum(1 for r in rows if r["required_pass"])
        avg_conf   = sum(r["confidence"] for r in rows) / len(rows)
        fallback_rate = sum(1 for r in rows if r["used_fallback"]) / len(rows)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Full pass",      f"{n_pass}/{len(rows)}")
        m2.metric("Required pass",  f"{n_required}/{len(rows)}")
        m3.metric("Avg confidence", f"{avg_conf:.2f}")
        m4.metric("Fallback rate",  f"{fallback_rate:.0%}")

        # ── Result table ────────────────────────────────────────────────────
        st.subheader("Case-by-case results")
        for r in rows:
            status = "✅ PASS" if r["pass"] else ("⚠️ WARN" if r["required_pass"] else "❌ FAIL")
            with st.expander(f"{status}  **{r['name']}** — {r['desc']}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Confidence", f"{r['confidence']:.3f}")
                c2.metric("Fallback",   str(r["used_fallback"]))
                c3.metric("Time (s)",   r["elapsed_s"])
                c4.metric("Top song",   r["top_song"][:20])

                checks = r["checks"]
                for k_name, v in checks.items():
                    if v is None:
                        icon = "⬜"
                    elif v:
                        icon = "✅"
                    else:
                        icon = "❌"
                    st.markdown(f"{icon} `{k_name}`")

        # ── Narrative summary ────────────────────────────────────────────────
        st.divider()
        note = (
            f"**{n_required} of {len(rows)} test cases passed all required checks.** "
            f"Average confidence: {avg_conf:.2f}. "
            f"Fallback triggered in {fallback_rate:.0%} of cases. "
        )
        if n_required == len(rows):
            note += "All required checks passed — the system is reliable across the test suite."
        else:
            note += f"{len(rows) - n_required} case(s) failed required checks; review the ❌ rows above."
        st.info(note)

