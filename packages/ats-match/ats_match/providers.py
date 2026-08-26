"""Protocols for the models this core refuses to depend on.

The matching core never imports a model library. An :class:`Embedder` and an
optional :class:`Reranker` are handed in by the caller, which is also where the
weights, the device and the caching live.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns text into row-normalised vectors, so a dot product is a cosine.

    An asymmetric model may expose ``query_prefix`` / ``passage_prefix``
    attributes; :func:`ats_match.fusion.build_matrix` reads them and prepends
    the right one to each side. Symmetric models simply leave them empty.
    """

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """``[len(texts), dim]``, each row unit length."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Scores ``(query, passage)`` pairs directly; higher is more relevant."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        ...


def prefix_of(embedder: object, which: str) -> str:
    """``query_prefix`` / ``passage_prefix`` of an embedder, or ``""``."""
    return str(getattr(embedder, f"{which}_prefix", "") or "")
