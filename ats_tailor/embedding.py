"""The ``ats_match.Embedder`` adapter around ``sentence-transformers``.

The matching core takes models as injected objects, so everything model-shaped
lives here: loading (cached per process, because ``eval`` runs the pipeline a
dozen times), row normalisation, the query/passage prefixes asymmetric models
need, and the on-disk vector cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import EMBED_MODEL, HF_TOKEN
from .loaders import (
    _text_hash,
    build_cache_key,
    load_cached_embeddings,
    save_cached_embeddings,
)

#: Model-family → (query prefix, passage prefix). Matched as a substring of the
#: model name, since the same family ships under many repo paths.
PREFIXES: dict[str, tuple[str, str]] = {
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
    "gte": ("", ""),
}


@lru_cache(maxsize=4)
def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, token=HF_TOKEN)


def prefixes_for(model_name: str) -> tuple[str, str]:
    lowered = str(model_name).lower()
    for family, pair in PREFIXES.items():
        if family in lowered:
            return pair
    return ("", "")


class SentenceTransformerEmbedder:
    """Row-normalised encoding, with the prefixes the model expects."""

    def __init__(self, model_name: str = EMBED_MODEL, model=None):
        self.model_name = model_name
        self._model = model
        self.query_prefix, self.passage_prefix = prefixes_for(model_name)

    @property
    def model(self):
        if self._model is None:
            self._model = load_sentence_transformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype=float)
        vectors = np.asarray(self.model.encode(texts), dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


def encode_cached(
    embedder: SentenceTransformerEmbedder,
    texts: Sequence[str],
    cache_dir: str | Path,
    kind: str,
    prefix: str = "",
    schema_version: str = "v2m1",
) -> np.ndarray:
    """Encode ``texts`` once per (model, content) pair, caching the matrix on disk.

    The key hashes the prefixed texts, so switching a model's prefixes or
    editing a single bullet invalidates the entry rather than silently
    returning vectors for the previous wording.
    """
    cache_dir = Path(cache_dir)
    payload = [f"{prefix}{text}" for text in texts]
    key = build_cache_key(kind, embedder.model_name, _text_hash("\n".join(payload)),
                          schema_version)
    cached = load_cached_embeddings(
        cache_dir, key, expected_ndim=2, expected_rows=len(payload)
    )
    if cached is not None:
        return np.asarray(cached, dtype=float)
    vectors = embedder.encode(payload)
    if len(payload):
        save_cached_embeddings(cache_dir, key, vectors)
    return vectors
