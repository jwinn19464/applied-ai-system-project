"""
NLP-based UserProfile parser using a HuggingFace zero-shot classification model.

The model (facebook/bart-large-mnli) is pre-trained on natural language inference
and requires no task-specific fine-tuning. We configure it with our domain label
sets (genres, moods, energy levels, acoustic preference) so it classifies free-text
descriptions into structured UserProfile fields.

Usage:
    parser = NLPProfileParser()
    profile = parser.parse("Something chill and acoustic for studying")
"""

from transformers import pipeline
from recommender import UserProfile

GENRES = ["pop", "rock", "lofi", "electronic", "rap", "jazz", "classical", "country", "J-pop"]
MOODS = ["happy", "sad", "chill", "intense", "romantic", "energetic", "melancholy"]
ENERGY_LABELS = ["high energy", "medium energy", "low energy"]
ACOUSTIC_LABELS = ["acoustic", "electronic or produced"]

_ENERGY_MAP = {"high energy": 0.85, "medium energy": 0.55, "low energy": 0.20}


class NLPProfileParser:
    """
    Converts a natural language description into a structured UserProfile
    by running zero-shot classification against domain-specific label sets.

    Each attribute (genre, mood, energy, acoustic preference) is classified
    independently so that the highest-confidence label wins per field.
    """

    def __init__(self, model: str = "facebook/bart-large-mnli"):
        print(f"Loading NLP model: {model} ...")
        self._classifier = pipeline("zero-shot-classification", model=model)
        print("Model ready.")

    def parse(self, text: str) -> UserProfile:
        """
        Parse a free-text music description into a UserProfile.

        Args:
            text: Natural language input, e.g. "upbeat pop for a workout".

        Returns:
            A UserProfile with fields inferred from the text.
        """
        genre = self._top_label(text, GENRES)
        mood = self._top_label(text, MOODS)
        energy_label = self._top_label(text, ENERGY_LABELS)
        energy = _ENERGY_MAP[energy_label]
        acoustic_label = self._top_label(text, ACOUSTIC_LABELS)
        likes_acoustic = acoustic_label == "acoustic"

        return UserProfile(
            favorite_genre=genre,
            favorite_mood=mood,
            target_energy=energy,
            likes_acoustic=likes_acoustic,
        )

    def _top_label(self, text: str, labels: list) -> str:
        result = self._classifier(text, candidate_labels=labels)
        return result["labels"][0]
