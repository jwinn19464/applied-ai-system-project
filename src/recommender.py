from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class RecommendationResult:
    """
    Wraps the output of Recommender.recommend() with quality metadata.
    confidence: 0.0–1.0. High = top song clearly stands out. Low = results are similarly scored.
    used_fallback: True if scoring criteria were relaxed due to low confidence.
    quality_note: Human-readable explanation of confidence level.
    """
    songs: List["Song"]
    confidence: float
    used_fallback: bool
    quality_note: str


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_danceability: Optional[float] = None


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Initialize the Recommender with a list of songs."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> RecommendationResult:
        """
        Return the top-k songs for the given user profile.
        Computes a confidence score from the result distribution. If confidence is
        low (top song does not clearly stand out), scoring tolerances are relaxed
        and recommendations are recalculated before returning.
        """
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        if user.target_danceability is not None:
            user_prefs["danceability"] = user.target_danceability

        songs, scores = self._score_all(user_prefs)
        confidence = _compute_confidence(scores)

        if confidence < 0.35:
            relaxed_prefs = {**user_prefs, "_relaxed": True}
            songs, scores = self._score_all(relaxed_prefs)
            new_confidence = _compute_confidence(scores)
            return RecommendationResult(
                songs=songs[:k],
                confidence=new_confidence,
                used_fallback=True,
                quality_note=(
                    f"Low confidence ({confidence:.2f}) — your preferences are uncommon "
                    f"in this catalog. Scoring tolerances were relaxed to surface better matches."
                ),
            )

        return RecommendationResult(
            songs=songs[:k],
            confidence=confidence,
            used_fallback=False,
            quality_note=f"Confidence: {confidence:.2f} — strong matches found.",
        )

    def _score_all(self, user_prefs: Dict) -> Tuple[List[Song], List[float]]:
        """Score all songs and return them sorted by score, highest first."""
        scored = []
        for song in self.songs:
            score, _ = score_song(user_prefs, song.__dict__)
            scored.append((score, song))
        scored.sort(key=lambda item: item[0], reverse=True)
        songs = [s for _, s in scored]
        scores = [sc for sc, _ in scored]
        return songs, scores

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Generate an explanation for why a song is recommended."""
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        if user.target_danceability is not None:
            user_prefs["danceability"] = user.target_danceability

        score, reasons = score_song(user_prefs, song.__dict__)
        if not reasons:
            return f"No strong attribute matches found. Score: {score:.2f}."
        return f"Score: {score:.2f}. " + "; ".join(reasons)

def _compute_confidence(scores: List[float]) -> float:
    """
    Returns a 0.0–1.0 confidence score based on how clearly the top result
    stands out from the median. High = strong match exists. Low = results are
    all similarly scored, meaning the catalog may not suit the user's preferences.
    """
    if len(scores) < 2:
        return 1.0
    top = scores[0]
    if top == 0.0:
        return 0.0
    median = scores[len(scores) // 2]
    return round(min((top - median) / top, 1.0), 3)


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return a list of dictionaries."""
    songs: List[Dict] = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    print(f"Loading songs from {csv_path}...")
    print(f"Loaded {len(songs)} songs.")
    return songs

DEFAULT_WEIGHTS = {
    "genre": 0.25 / 2,
    "mood": 0.20,
    "energy": 0.20 * 2,
    "danceability": 0.15,
    "acousticness": 0.20,
}


def score_song(user_prefs: Dict, song: Dict, custom_weights: Optional[Dict] = None) -> Tuple[float, List[str]]:
    """
    Score a song against user preferences and return (score, reasons).

    Args:
        user_prefs:     User preference dict. May include '_relaxed': True to
                        widen energy/danceability tolerances.
        song:           Song attribute dict.
        custom_weights: Optional weight overrides. Falls back to DEFAULT_WEIGHTS
                        for any key not provided. Used by RecommenderAgent to
                        apply profile-specific scoring strategies.
    """
    relaxed = user_prefs.get("_relaxed", False)
    energy_tolerance = 0.40 if relaxed else 0.20
    dance_tolerance = 0.40 if relaxed else 0.20

    weights = {**DEFAULT_WEIGHTS, **(custom_weights or {})}

    score = 0.0
    reasons: List[str] = []

    if "genre" in user_prefs and user_prefs["genre"]:
        if song.get("genre", "").strip().lower() == str(user_prefs["genre"]).strip().lower():
            score += weights["genre"] * 3
            reasons.append(f"genre match (+{weights['genre'] * 3:.2f})")

    if "mood" in user_prefs and user_prefs["mood"]:
        if song.get("mood", "").strip().lower() == str(user_prefs["mood"]).strip().lower():
            score += weights["mood"] * 1.5
            reasons.append(f"mood match (+{weights['mood'] * 1.5:.2f})")

    if "energy" in user_prefs and isinstance(user_prefs["energy"], (int, float)):
        energy_diff = abs(song.get("energy", 0.0) - float(user_prefs["energy"]))
        if energy_diff <= energy_tolerance:
            score += weights["energy"] * 2
            label = "energy close (relaxed)" if relaxed else "energy close"
            reasons.append(f"{label} (+{weights['energy'] * 2:.2f})")

    if "danceability" in user_prefs and isinstance(user_prefs["danceability"], (int, float)):
        dance_diff = abs(song.get("danceability", 0.0) - float(user_prefs["danceability"]))
        if dance_diff <= dance_tolerance:
            score += weights["danceability"]
            label = "danceability close (relaxed)" if relaxed else "danceability close"
            reasons.append(f"{label} (+{weights['danceability']:.2f})")

    if "likes_acoustic" in user_prefs:
        likes_acoustic = bool(user_prefs["likes_acoustic"])
        acousticness = float(song.get("acousticness", 0.0))
        if likes_acoustic and acousticness >= 0.50:
            score += weights["acousticness"]
            reasons.append(f"acousticness match (+{weights['acousticness']:.2f})")
        elif not likes_acoustic and acousticness <= 0.50:
            score += weights["acousticness"]
            reasons.append(f"acousticness match (+{weights['acousticness']:.2f})")

    return score, reasons



def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Return the top-k recommended songs for a user based on their preferences."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons) if reasons else f"No preferred attribute matches. Score: {score:.2f}."
        scored.append((song, score, explanation))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
