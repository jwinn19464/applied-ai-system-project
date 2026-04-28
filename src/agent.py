"""
Agentic Workflow module.

RecommenderAgent executes a six-step reasoning chain. Every step is logged to
an AgentStep record so the caller can inspect exactly what the agent decided
and why — making the process fully observable.

Steps
-----
1  Profile Analysis   Classify the user profile into a type (energy-driven,
                      genre-specific, acoustic-seeker, or balanced).
2  Strategy Planning  Derive custom attribute weights for this profile type.
3  Candidate Scoring  Score all songs using the adapted weights.
4  Quality Check      Compute confidence; decide whether to run fallback.
5  Context Retrieval  RAG: fetch relevant genre/mood knowledge for the top song.
6  Explanation        Synthesize scoring reasons + retrieved context into a
                      richer, human-readable explanation.

The adapted weights from Step 2 are what make the agent's output measurably
different from the plain Recommender — for example, an energy-driven profile
will weight energy 2× higher than the default.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from recommender import Song, UserProfile, RecommendationResult, score_song, _compute_confidence, DEFAULT_WEIGHTS
from rag import DocumentStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    name: str
    decision: str       # what the agent decided
    detail: str = ""    # supporting information


@dataclass
class AgentResult:
    songs: List[Song]
    confidence: float
    used_fallback: bool
    steps: List[AgentStep]
    explanation: str    # enriched explanation for the top recommendation


# ---------------------------------------------------------------------------
# Profile types
# ---------------------------------------------------------------------------

PROFILE_TYPES = {
    "energy_driven":    "User's target_energy is extreme (< 0.25 or > 0.75). Energy matching is the primary axis.",
    "genre_specific":   "User's genre is a niche category (lofi, jazz, classical, metal, ambient, synthwave). Genre lock-in is critical.",
    "acoustic_seeker":  "User strongly prefers acoustic music. Acousticness weight elevated.",
    "balanced":         "No single attribute dominates. Use default weights.",
}

NICHE_GENRES = {"lofi", "jazz", "classical", "metal", "ambient", "synthwave", "country"}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RecommenderAgent:
    """
    Multi-step recommendation agent with a RAG-enriched explanation chain.

    Args:
        store: A loaded DocumentStore used for context retrieval (Step 5).
               If None, Step 5 is skipped and explanations are score-only.
    """

    def __init__(self, store: Optional[DocumentStore] = None):
        self._store = store

    def run(self, user: UserProfile, songs: List[Song], k: int = 5, rag_top_k: int = 2) -> AgentResult:
        steps: List[AgentStep] = []

        # ── Step 1: Profile Analysis ────────────────────────────────────────
        profile_type = self._analyze_profile(user)
        steps.append(AgentStep(
            name="Profile Analysis",
            decision=f"type={profile_type}",
            detail=PROFILE_TYPES[profile_type],
        ))

        # ── Step 2: Strategy Planning ───────────────────────────────────────
        weights = self._plan_weights(user, profile_type)
        weight_summary = ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
        steps.append(AgentStep(
            name="Strategy Planning",
            decision=f"custom weights derived for '{profile_type}'",
            detail=weight_summary,
        ))

        # ── Step 3: Candidate Scoring ───────────────────────────────────────
        user_prefs = self._profile_to_prefs(user)
        scored = self._score_all(user_prefs, songs, weights)
        top3 = ", ".join(f"'{s.title}' ({sc:.2f})" for sc, s, _ in scored[:3])
        steps.append(AgentStep(
            name="Candidate Scoring",
            decision=f"scored {len(songs)} songs with adapted weights",
            detail=f"Top 3: {top3}",
        ))

        # ── Step 4: Quality Check ───────────────────────────────────────────
        all_scores = [sc for sc, _, _r in scored]
        confidence = _compute_confidence(all_scores)
        used_fallback = False

        if confidence < 0.35:
            relaxed_prefs = {**user_prefs, "_relaxed": True}
            scored = self._score_all(relaxed_prefs, songs, weights)
            confidence = _compute_confidence([sc for sc, _, _r in scored])
            used_fallback = True
            steps.append(AgentStep(
                name="Quality Check",
                decision="confidence too low — fallback triggered",
                detail=f"original confidence={(all_scores[0] if all_scores else 0):.3f}, "
                       f"post-fallback confidence={confidence:.3f}",
            ))
        else:
            steps.append(AgentStep(
                name="Quality Check",
                decision="confidence acceptable — proceeding",
                detail=f"confidence={confidence:.3f}",
            ))

        top_song: Song = scored[0][1]
        top_reasons: List[str] = scored[0][2] if len(scored[0]) > 2 else []

        # ── Step 5: Context Retrieval (RAG) ─────────────────────────────────
        context_text = ""
        if self._store is not None:
            hits = self._store.retrieve_for_song(top_song.genre, top_song.mood, top_k=rag_top_k)
            if hits:
                context_text = " ".join(chunk.text[:200] for _, chunk in hits)
                sources = [chunk.label for _, chunk in hits]
                steps.append(AgentStep(
                    name="Context Retrieval",
                    decision=f"retrieved {len(hits)} chunks from document store",
                    detail=f"sources: {sources} | preview: {context_text[:120]}...",
                ))
            else:
                steps.append(AgentStep(
                    name="Context Retrieval",
                    decision="no relevant documents found",
                    detail="",
                ))
        else:
            steps.append(AgentStep(
                name="Context Retrieval",
                decision="skipped — no DocumentStore provided",
                detail="",
            ))

        # ── Step 6: Explanation Assembly ────────────────────────────────────
        explanation = self._assemble_explanation(top_song, top_reasons, context_text, user)
        steps.append(AgentStep(
            name="Explanation Assembly",
            decision="explanation generated",
            detail=explanation,
        ))

        return AgentResult(
            songs=[s for _, s, _r in scored[:k]],
            confidence=confidence,
            used_fallback=used_fallback,
            steps=steps,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyze_profile(self, user: UserProfile) -> str:
        if user.target_energy < 0.25 or user.target_energy > 0.75:
            return "energy_driven"
        if user.favorite_genre.lower() in NICHE_GENRES:
            return "genre_specific"
        if user.likes_acoustic:
            return "acoustic_seeker"
        return "balanced"

    def _plan_weights(self, user: UserProfile, profile_type: str) -> Dict[str, float]:
        """Return attribute weights tuned for the detected profile type."""
        w = dict(DEFAULT_WEIGHTS)
        if profile_type == "energy_driven":
            w["energy"] = w["energy"] * 2.0       # double energy weight
            w["genre"]  = w["genre"]  * 0.75      # relax genre slightly
        elif profile_type == "genre_specific":
            w["genre"]  = w["genre"]  * 3.0       # lock onto niche genre
            w["mood"]   = w["mood"]   * 1.5
        elif profile_type == "acoustic_seeker":
            w["acousticness"] = w["acousticness"] * 2.5
            w["energy"]       = w["energy"]       * 0.75
        # "balanced" → default weights unchanged
        return w

    def _score_all(
        self,
        user_prefs: Dict,
        songs: List[Song],
        weights: Dict,
    ) -> List[Tuple]:
        """Score all songs, return sorted list of (score, song, reasons)."""
        results = []
        for song in songs:
            score, reasons = score_song(user_prefs, song.__dict__, custom_weights=weights)
            results.append((score, song, reasons))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def _profile_to_prefs(self, user: UserProfile) -> Dict:
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        if user.target_danceability is not None:
            prefs["danceability"] = user.target_danceability
        return prefs

    def _assemble_explanation(
        self,
        song: Song,
        reasons: List[str],
        context: str,
        user: UserProfile,
    ) -> str:
        """Combine scoring reasons with RAG-retrieved context into a richer explanation."""
        parts = [f"'{song.title}' by {song.artist} ({song.genre} / {song.mood})"]

        if reasons:
            parts.append("Matched on: " + "; ".join(reasons) + ".")
        else:
            parts.append("No strong attribute matches — shown as best available.")

        if context:
            # Pick the most informative sentence (longest, skipping very short fragments)
            sentences = [s.strip() for s in context.split(".") if len(s.strip()) > 40]
            if sentences:
                best = max(sentences, key=len)
                # Tie the context sentence back to this specific song and user preference
                acoustic_note = "its acoustic texture" if song.acousticness >= 0.5 else "its produced sound"
                energy_note = (
                    "high energy" if song.energy >= 0.75
                    else "low energy" if song.energy <= 0.35
                    else "moderate energy"
                )
                parts.append(
                    f"For a {user.favorite_mood} {user.favorite_genre} listener: {best.rstrip('.')},"
                    f" which aligns with this track's {energy_note} and {acoustic_note}."
                )

        return " ".join(parts)


def print_agent_trace(result: AgentResult) -> None:
    """Pretty-print all steps in an AgentResult for inspection."""
    print("\n── Agent Reasoning Trace ──────────────────────────────────────")
    for i, step in enumerate(result.steps, 1):
        print(f"  Step {i} [{step.name}]")
        print(f"    Decision : {step.decision}")
        if step.detail:
            print(f"    Detail   : {step.detail[:120]}")
    print(f"\n── Final Explanation ──────────────────────────────────────────")
    print(f"  {result.explanation}")
    fallback_tag = "  [fallback was used]" if result.used_fallback else ""
    print(f"  Confidence: {result.confidence:.2f}{fallback_tag}")
    print("───────────────────────────────────────────────────────────────")
