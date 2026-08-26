"""Rerankers: the third opinion in the fusion, and its price.

``ats_match.build_matrix`` takes an optional :class:`ats_match.Reranker` and
asks it to re-order the top ``k`` claims for each requirement. Two are
available, both off by default:

- ``ATS_RERANK=true`` — a sentence-transformers cross-encoder
  (``ATS_RERANK_MODEL``, default ``cross-encoder/ms-marco-MiniLM-L-6-v2``;
  ``BAAI/bge-reranker-base`` also works).
- ``ATS_RERANK=llm`` — the configured provider, prompted listwise.

Off by default *on evidence*, not caution: measured over the fourteen labelled
postings in Phase 3a the cross-encoder moved must-coverage by 0.00 points,
nice by +0.6, and cost 3 points of context rate at 2.2× the wall clock
(``docs/v2-design.md`` §8). It is kept because the number is worth re-checking
when the embedder or the index changes, not because it is currently worth
paying for.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .. import config


class CrossEncoderReranker:
    """A sentence-transformers ``CrossEncoder`` behind ``Reranker``."""

    def __init__(self, model_name: str | None = None, model=None):
        self.model_name = model_name or config.RERANK_MODEL
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, token=config.HF_TOKEN)
        return self._model

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        pairs = list(pairs)
        if not pairs:
            return []
        return [float(score) for score in self.model.predict(pairs)]


class ProviderReranker:
    """The configured LLM provider, asked listwise.

    ``ats_match`` hands a reranker ``(query, passage)`` pairs, and it calls once
    per requirement with that requirement's top ``k`` claims. Pairs are
    therefore grouped by query and one prompt is sent per group — which for the
    only caller is one prompt per requirement, exactly the batching the plan
    asked for. A provider that answers ``None`` (no model, bad JSON, wrong
    length) scores every pair 0.0, which leaves the fused order untouched
    rather than inventing one.
    """

    def __init__(self, provider):
        self.provider = provider

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        pairs = list(pairs)
        if not pairs:
            return []
        groups: dict[str, list[int]] = {}
        for i, (query, _passage) in enumerate(pairs):
            groups.setdefault(query, []).append(i)

        scores = [0.0] * len(pairs)
        for query, indices in groups.items():
            answer = self.provider.rerank(query, [pairs[i][1] for i in indices])
            if not answer or len(answer) != len(indices):
                continue
            for position, i in enumerate(indices):
                scores[i] = float(answer[position])
        return scores


@lru_cache(maxsize=2)
def _cached_cross_encoder(model_name: str) -> CrossEncoderReranker:
    """One set of weights per process: ``eval`` runs the pipeline a dozen times."""
    return CrossEncoderReranker(model_name)


def load_reranker(mode: str | None = None, provider=None):
    """The reranker ``ATS_RERANK`` asks for, or ``None``."""
    mode = (mode or config.RERANK_MODE or "none").strip().lower()
    if mode in {"", "none", "false"}:
        return None
    if mode == "llm":
        if provider is None or getattr(provider, "name", "none") == "none":
            return None
        return ProviderReranker(provider)
    return _cached_cross_encoder(config.RERANK_MODEL)
