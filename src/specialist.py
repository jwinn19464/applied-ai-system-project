"""
Few-Shot Specialization module.

Uses google/flan-t5-base with explicit few-shot examples embedded in the
prompt to generate recommendation explanations in two distinct styles:

  hype  — energetic DJ voice; short, punchy, exclamation-heavy
  study — calm academic voice; measured, descriptive, factual

The baseline is the raw score_song() output (attribute match labels +
a numeric score). The specialist output is measurably different:
  - Longer sentences
  - Style-specific vocabulary (measured by style_score())
  - No raw numeric scores visible to the user

Why flan-t5-base?
  flan-T5 was instruction-fine-tuned on hundreds of NLP tasks. It follows
  a "Task: ... Input: ... Output:" pattern reliably for short generation
  tasks without needing any additional training on our data.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Literal

StyleName = Literal["hype", "formal"]

# ---------------------------------------------------------------------------
# Few-shot examples per style
# They teach the model the expected tone and structure.
# ---------------------------------------------------------------------------

_HYPE_EXAMPLES = """\
Example 1
Song: Gym Hero by Max Pulse | genre=pop | mood=intense | energy=0.93
Matched: genre match, energy close
Output: YO this is THE one — Gym Hero hits hard, pure pop energy cranked to 0.93. Built for moving. Do NOT skip this.

Example 2
Song: Storm Runner by Voltline | genre=rock | mood=intense | energy=0.91
Matched: energy close, acousticness match
Output: STORM RUNNER goes OFF. Rock intensity, energy off the charts. This track RUNS you, not the other way around.

Example 3
Song: Neon Pulse by Synth Master | genre=electronic | mood=euphoric | energy=0.88
Matched: genre match, mood match
Output: Neon Pulse is EUPHORIC electronic bliss — genre locked, mood locked, vibes absolutely immaculate. Play it loud.\
"""

_FORMAL_EXAMPLES = """\
Example 1
Song: Library Rain by Paper Lanterns | genre=lofi | mood=chill | energy=0.35
Matched: genre match, mood match, acousticness match
Output: Library Rain aligns well with your preferences. The lofi genre, chill mood, and gentle acousticness collectively create a low-stimulation soundscape suited to sustained focus.

Example 2
Song: Spacewalk Thoughts by Orbit Bloom | genre=ambient | mood=chill | energy=0.28
Matched: mood match, energy close, acousticness match
Output: Spacewalk Thoughts exhibits a high degree of compatibility across the mood, energy, and acousticness dimensions. The minimal energy level and atmospheric texture support extended listening without distraction.

Example 3
Song: Moonlight Sonata by Ludwig Echo | genre=classical | mood=peaceful | energy=0.20
Matched: energy close, acousticness match
Output: Moonlight Sonata closely matches your energy target and acoustic preference. Classical instrumentation and a peaceful mood profile make this a suitable candidate for reflective or low-activity listening contexts.\
"""

_PROMPT_TEMPLATE = """\
Task: Write a music recommendation explanation in the following style: {style_description}.

{examples}

Now write an explanation for the following. Match the style exactly.
Song: {title} by {artist} | genre={genre} | mood={mood} | energy={energy:.2f}
Matched: {reasons}
Output:\
"""

_STYLE_META = {
    "hype": {
        "description": "energetic DJ hype voice — short punchy sentences, exclamations, all-caps words for emphasis",
        "examples": _HYPE_EXAMPLES,
        "vocab": ["yo", "fire", "slaps", "goes off", "drop", "do not skip", "built for", "hits hard", "cranked", "bliss"],
    },
    "formal": {
        "description": "formal, professional tone — complete sentences, factual observations, no exclamations, no slang",
        "examples": _FORMAL_EXAMPLES,
        "vocab": ["aligns", "exhibits", "compatibility", "dimension", "profile", "acoustic", "suitable", "context", "sustained", "collectively"],
    },
}


class FewShotExplainer:
    """
    Generates recommendation explanations in a specified style using
    few-shot prompting with flan-t5-base.

    Usage:
        explainer = FewShotExplainer()
        text = explainer.explain(song, reasons=["genre match", "energy close"], style="hype")
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"Loading specialist model: {model_name} ...")
        self._tokenizer = T5Tokenizer.from_pretrained(model_name)
        self._model     = T5ForConditionalGeneration.from_pretrained(model_name)
        print("Specialist model ready.")

    def explain(self, song, reasons: list, style: StyleName = "hype") -> str:
        """
        Generate a styled explanation for a recommended song.

        Args:
            song:    A Song dataclass instance.
            reasons: List of match reason strings from score_song().
            style:   'hype' or 'study'.

        Returns:
            A generated explanation string.
        """
        meta = _STYLE_META[style]
        prompt = _PROMPT_TEMPLATE.format(
            style_description=meta["description"],
            examples=meta["examples"],
            title=song.title,
            artist=song.artist,
            genre=song.genre,
            mood=song.mood,
            energy=song.energy,
            reasons=", ".join(reasons) if reasons else "no strong matches",
        )
        inputs  = self._tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self._model.generate(**inputs, max_new_tokens=120)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def baseline(self, song, reasons: list) -> str:
        """Return the raw rule-based baseline explanation (no model)."""
        if not reasons:
            return f"No strong attribute matches for '{song.title}'. Score: 0.00."
        return f"'{song.title}' by {song.artist}. " + "; ".join(reasons) + "."


def style_score(text: str, style: StyleName) -> float:
    """
    Measure how strongly a text exhibits vocabulary from the target style.
    Returns the fraction of style vocabulary words found in the text.
    Used by eval.py to verify specialist output measurably differs from baseline.
    """
    vocab = _STYLE_META[style]["vocab"]
    text_lower = text.lower()
    hits = sum(1 for word in vocab if word in text_lower)
    return hits / len(vocab)
