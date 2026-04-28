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

    def load_song_metadata(self, path: str, source: str = "song_metadata") -> int:
        """
        Parse a song metadata document and add one chunk per song.
        Supports both the old and new metadata formats.
        """
        with open(path, encoding='utf-8') as f:
            raw = f.read()

        blocks = []
        current = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("SONG_ID:") and current:
                blocks.append("\n".join(current))
                current = [stripped]
            elif stripped.startswith("SONG ID:") and current:
                blocks.append("\n".join(current))
                current = [stripped]
            else:
                current.append(stripped)
        if current:
            blocks.append("\n".join(current))

        added = 0
        for block in blocks:
            meta = {}
            for line in block.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                meta[key.strip().upper()] = value.strip()

            song_id = meta.get("SONG_ID") or meta.get("SONG ID")
            if not song_id:
                continue

            title = meta.get("TITLE", "")
            artist = meta.get("ARTIST", "")
            genre = meta.get("PRIMARY GENRE") or meta.get("GENRE", "")
            mood = meta.get("PRIMARY MOOD") or meta.get("MOOD", "")
            additional = meta.get("ADDITIONAL GENRES") or meta.get("ADDITIONAL_GENRES", "")
            description = meta.get("DETAILED DESCRIPTION", "")
            nuance = meta.get("NUANCE/CONTEXT", "")
            rag_note = meta.get("RAG CONTEXTUAL NOTE", "")
            attributes = meta.get("TECHNICAL METRICS") or meta.get("ATTRIBUTES", "")

            text_parts = [
                title,
                artist,
                genre,
                mood,
                additional,
                description,
                nuance,
                rag_note,
                attributes,
            ]
            text = " ".join(part for part in text_parts if part).strip()
            if not text:
                continue

            label = f"[song:{song_id}] {title}"
            self._chunks.append(DocumentChunk(source=source, label=label, text=text))
            added += 1

        self._recompute_embeddings()
        return added

    def _parse_song_id(self, label: str) -> int | None:
        if label.startswith("[song:"):
            try:
                return int(label.split("]", 1)[0][6:])
            except ValueError:
                return None
        return None

    def retrieve_song_metadata(self, query: str, top_k: int = 5) -> List[Tuple[int, float, DocumentChunk]]:
        hits = self.retrieve(query, top_k=top_k)
        results = []
        for score, chunk in hits:
            song_id = self._parse_song_id(chunk.label)
            if song_id is not None and chunk.source == "song_metadata":
                results.append((song_id, score, chunk))
        return results

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
    Build and return a DocumentStore loaded with genres.txt, moods.txt,
    and song metadata for richer RAG retrieval.
    """
    store = DocumentStore()
    genres_path = os.path.join(docs_dir, "genres.txt")
    moods_path = os.path.join(docs_dir, "moods.txt")
    metadata_candidates = [
        "Enriched_Song_Metadata_RAG.txt",
        "Music_Catalog_RAG_Metadata.txt",
    ]

    metadata_path = None
    for filename in metadata_candidates:
        candidate = os.path.join(docs_dir, filename)
        if os.path.exists(candidate):
            metadata_path = candidate
            break

    n_genres = store.load_file(genres_path, source="genres")
    n_moods = store.load_file(moods_path, source="moods")
    n_meta = store.load_song_metadata(metadata_path, source="song_metadata") if metadata_path else 0

    print(
        f"DocumentStore ready: {n_genres} genre chunks + {n_moods} mood chunks "
        f"+ {n_meta} song metadata chunks loaded."
    )
    return store
