"""
RAG (Retrieval-Augmented Generation) module.

Loads knowledge-base documents about genres and moods, embeds them using
sentence-transformers, and retrieves the most relevant chunks for a given
query (typically a song + user profile description).

The retrieved context is used downstream to enrich recommendation explanations
with real domain knowledge rather than just attribute match scores.

Two data sources:
  - data/docs/genres.txt  — genre characteristics and use-cases
  - data/docs/moods.txt   — mood descriptions and emotional context
"""

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, util
import torch


@dataclass
class DocumentChunk:
    source: str    # e.g. "genres.txt"
    label: str     # e.g. "[pop]" or "[chill]"
    text: str      # the chunk body


class DocumentStore:
    """
    Embeds and retrieves document chunks using cosine similarity.

    Usage:
        store = DocumentStore()
        store.load_file("data/docs/genres.txt", source="genres")
        store.load_file("data/docs/moods.txt",  source="moods")
        results = store.retrieve("upbeat pop for a workout", top_k=2)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name} ...")
        self._model = SentenceTransformer(model_name)
        self._chunks: List[DocumentChunk] = []
        self._embeddings = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_file(self, path: str, source: str) -> int:
        """
        Parse a labeled-section document and add all sections as chunks.
        Section headers are lines like '[pop]' or '[chill]'.
        Returns the number of chunks added.
        """
        with open(path, encoding="utf-8") as f:
            raw = f.read()

        sections = re.split(r"(\[[^\]]+\])", raw)
        added = 0
        label = ""
        for part in sections:
            part = part.strip()
            if re.fullmatch(r"\[[^\]]+\]", part):
                label = part
            elif part and label:
                self._chunks.append(DocumentChunk(source=source, label=label, text=part))
                added += 1

        self._recompute_embeddings()
        return added

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[float, DocumentChunk]]:
        """
        Return the top_k most relevant chunks for a query string.
        Scores are cosine similarities in [0, 1].
        """
        if not self._chunks or self._embeddings is None:
            return []

        query_emb = self._model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self._embeddings)[0]
        top_indices = torch.topk(scores, min(top_k, len(self._chunks))).indices
        return [(float(scores[i]), self._chunks[i]) for i in top_indices]

    def retrieve_for_song(self, genre: str, mood: str, top_k: int = 2) -> List[Tuple[float, DocumentChunk]]:
        """Convenience method: retrieve context relevant to a specific song's genre + mood."""
        query = f"{genre} music with a {mood} mood"
        return self.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute_embeddings(self) -> None:
        texts = [c.text for c in self._chunks]
        self._embeddings = self._model.encode(texts, convert_to_tensor=True)

    def __len__(self) -> int:
        return len(self._chunks)


def build_default_store(docs_dir: str) -> DocumentStore:
    """
    Build and return a DocumentStore loaded with genres.txt and moods.txt
    from the given docs directory.
    """
    store = DocumentStore()
    genres_path = os.path.join(docs_dir, "genres.txt")
    moods_path  = os.path.join(docs_dir, "moods.txt")
    n_genres = store.load_file(genres_path, source="genres")
    n_moods  = store.load_file(moods_path,  source="moods")
    print(f"DocumentStore ready: {n_genres} genre chunks + {n_moods} mood chunks loaded.")
    return store
