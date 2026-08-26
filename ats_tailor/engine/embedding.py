"""The ``ats_match.Embedder`` adapter around ``sentence-transformers``.

The matching core takes models as injected objects, so everything model-shaped
lives here: loading (cached per process, because ``eval`` runs the pipeline a
dozen times), row normalisation, the query/passage prefixes asymmetric models
need, and the vector cache in front of them.

The cache is the runtime store (:func:`encode_store`); the ``.npy`` directory
cache this module also carried through Phase 3a is gone — nothing called it
once the store landed, and a second cache with its own key scheme is a second
way for two runs to disagree.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from ..config import EMBED_MODEL, HF_TOKEN
from ..index.loaders import _text_hash

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


def text_hash(text: str, prefix: str = "") -> str:
    """The embedding cache key for one string: sha256 of the *prefixed* text.

    The prefix is part of the key because it is part of what the model was
    asked to encode: switching a model's query/passage prefixes has to
    invalidate the entry rather than silently return vectors for the other
    prompt.
    """
    return _text_hash(f"{prefix}{text}")


def encode_store(
    embedder: SentenceTransformerEmbedder,
    texts: Sequence[str],
    store,
    kind: str | Sequence[str],
    prefix: str = "",
) -> np.ndarray:
    """Encode ``texts``, reading and writing the runtime store one text at a time.

    ``kind`` is one label for every text, or one label per text when a batch
    mixes them (the pipeline embeds bullets and skills in a single call).

    Where the ``.npy`` cache v2 used to keep keyed
    on a hash of the *whole list*, so editing one bullet re-encoded every
    claim, the store keys on each text: a run after an edit pays for the edit
    only, and ``sync --embed`` can fill the cache ahead of any run.

    v1 is untouched: it caches through :func:`loaders.load_cached_embeddings`
    and keeps writing ``.emb_cache/*.npy``.
    """
    texts = list(texts)
    if not texts:
        return np.zeros((0, embedder.dimension), dtype=float)
    kinds = [kind] * len(texts) if isinstance(kind, str) else list(kind)
    if len(kinds) != len(texts):
        raise ValueError("kind must be a string or one label per text")
    hashes = [text_hash(text, prefix) for text in texts]
    cached = store.get_embeddings(hashes, embedder.model_name)

    missing = list(dict.fromkeys(h for h in hashes if h not in cached))
    if missing:
        by_hash = {h: t for h, t in zip(hashes, texts)}
        kind_of = {h: k for h, k in zip(hashes, kinds)}
        payload = [f"{prefix}{by_hash[h]}" for h in missing]
        vectors = embedder.encode(payload)
        store.put_embeddings(
            (h, embedder.model_name, kind_of[h], vectors[i])
            for i, h in enumerate(missing)
        )
        # Read back rather than reuse what was just encoded: the store keeps
        # float32, so a freshly computed float64 row would make a cold run
        # differ from a warm one in the last bits — and the whole point of the
        # cache is that it changes nothing but the time taken.
        cached.update(store.get_embeddings(missing, embedder.model_name))

    return np.vstack([np.asarray(cached[h], dtype=float) for h in hashes])
