"""Rank fusion: one score matrix from a dense retriever, BM25, and a reranker.

Reciprocal rank fusion is used rather than a weighted sum of raw scores
because the three signals are on incomparable scales — a cosine of 0.42, a
BM25 of 7.3 and a cross-encoder logit of −1.1 cannot be added, but their ranks
can. What comes out is rescaled per requirement into ``[0, 1]``, so the
selection objective can treat every row the same way.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .lexical import BM25, normalise, tokenize
from .providers import prefix_of
from .types import Claim, Requirement

DEFAULT_K = 60
#: dense, lexical, reranker
DEFAULT_WEIGHTS: tuple[float, float, float] = (1.0, 1.0, 1.5)
#: Cosine bounds of the relevance gate — see :func:`build_matrix`. Below the
#: first value a pair is treated as unrelated, above the second as a full
#: match; both sit where a general-purpose sentence encoder puts "unrelated"
#: and "clearly about the same thing".
DEFAULT_RELEVANCE: tuple[float, float] = (0.15, 0.60)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Item indices ordered best first; ties broken by index, so it is stable."""
    scores = np.asarray(scores, dtype=float)
    return np.lexsort((np.arange(scores.size), -scores))


def rrf(
    rank_lists: Sequence[Sequence[int]],
    k: int = DEFAULT_K,
    n: int | None = None,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Reciprocal rank fusion of several orderings over the same items.

    Each entry of ``rank_lists`` is item indices ordered best first. An item
    missing from a list simply earns nothing from it.
    """
    lists = [list(order) for order in rank_lists]
    if n is None:
        n = 1 + max((max(order) for order in lists if order), default=-1)
    fused = np.zeros(int(n), dtype=float)
    weights = list(weights) if weights is not None else [1.0] * len(lists)
    for order, weight in zip(lists, weights):
        for position, item in enumerate(order):
            fused[item] += float(weight) / (k + position + 1)
    return fused


def _unit_scale(fused: np.ndarray) -> np.ndarray:
    """Min-max a fused row into ``[0, 1]``.

    Absolute RRF values say little (they depend on how many candidates there
    are), while the spread within one row is exactly the discrimination the
    objective needs: the best candidate reaches 1, the worst 0, and the decay
    in between is the fused rank decay.
    """
    low, high = float(fused.min()), float(fused.max())
    if high - low <= 1e-12:
        return np.zeros_like(fused) if high <= 0 else np.ones_like(fused)
    return (fused - low) / (high - low)


def _encode(embedder, texts: Sequence[str], prefix: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=float)
    vectors = np.asarray(
        embedder.encode([f"{prefix}{text}" for text in texts]), dtype=float
    )
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def claim_documents(
    claims: Sequence[Claim], alias_map: Mapping[str, str] | None = None
) -> list[list[str]]:
    """Token stream per claim: its text plus the vocabulary terms attached to it."""
    return [
        tokenize(" ".join([claim.text, *claim.terms]), alias_map) for claim in claims
    ]


def requirement_query(
    requirement: Requirement, alias_map: Mapping[str, str] | None = None
) -> list[str]:
    return tokenize(" ".join([requirement.text, *requirement.terms]), alias_map)


def term_overlap(
    requirements: Sequence[Requirement], claims: Sequence[Claim]
) -> np.ndarray:
    """``[R, C]``: the share of each requirement's terms the claim also carries."""
    claim_terms = [{normalise(t) for t in claim.terms if str(t).strip()} for claim in claims]
    overlap = np.zeros((len(requirements), len(claims)), dtype=float)
    for r, requirement in enumerate(requirements):
        wanted = {normalise(t) for t in requirement.terms if str(t).strip()}
        if not wanted:
            continue
        for c, held in enumerate(claim_terms):
            shared = len(wanted & held)
            if shared:
                overlap[r, c] = shared / len(wanted)
    return overlap


def build_matrix(
    requirements: Sequence[Requirement],
    claims: Sequence[Claim],
    embedder,
    alias_map: Mapping[str, str] | None = None,
    reranker=None,
    top_k: int = 5,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    k: int = DEFAULT_K,
    relevance: tuple[float, float] | None = DEFAULT_RELEVANCE,
    claim_vectors: np.ndarray | None = None,
    requirement_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(matrix[R, C] in [0, 1], claim_sim[C, C])``.

    Rank fusion answers "which claim is the best answer to this requirement",
    but a rank says nothing about whether the best answer is any good: the top
    claim for a requirement nothing matches would still come out at 1.0, and a
    requirement no claim can cover would look covered. So the fused rank is
    multiplied by a *relevance gate* — the pair's cosine mapped through
    ``relevance = (unrelated, clearly related)`` — and a boilerplate line in
    the posting ends up with a low row instead of a confident one. Pass
    ``relevance=None`` for pure rank fusion.

    ``claim_vectors`` / ``requirement_vectors`` let a caller supply cached
    embeddings; they must already be row-normalised and in the same order as
    ``claims`` / ``requirements``.
    """
    n_req, n_claim = len(requirements), len(claims)
    if not n_req or not n_claim:
        return np.zeros((n_req, n_claim), dtype=float), np.zeros((n_claim, n_claim))

    passages = (
        claim_vectors
        if claim_vectors is not None
        else _encode(embedder, [c.text for c in claims], prefix_of(embedder, "passage"))
    )
    queries = (
        requirement_vectors
        if requirement_vectors is not None
        else _encode(embedder, [r.text for r in requirements], prefix_of(embedder, "query"))
    )
    passages = np.asarray(passages, dtype=float)
    queries = np.asarray(queries, dtype=float)

    dense = queries @ passages.T
    claim_sim = passages @ passages.T
    if relevance:
        low, high = relevance
        gate = np.clip((dense - low) / max(high - low, 1e-9), 0.0, 1.0)
        # Shared vocabulary opens the gate on its own. An encoder can miss that
        # a claim is about the very thing a requirement names — the words are
        # there, the sentences simply do not look alike — and suppressing that
        # match is the one mistake a keyword-driven matcher must not make.
        gate = np.maximum(gate, term_overlap(requirements, claims))
    else:
        gate = np.ones_like(dense)

    bm25 = BM25(claim_documents(claims, alias_map))
    matrix = np.zeros((n_req, n_claim), dtype=float)
    dense_w, lex_w, rerank_w = (list(weights) + [0.0, 0.0, 0.0])[:3]

    for r, requirement in enumerate(requirements):
        lexical = bm25.score(requirement_query(requirement, alias_map))
        orders = [ranks_from_scores(dense[r]), ranks_from_scores(lexical)]
        used = [dense_w, lex_w]
        if reranker is not None and top_k > 0:
            preliminary = rrf(orders, k=k, n=n_claim, weights=used)
            candidates = [int(i) for i in ranks_from_scores(preliminary)[:top_k]]
            scores = list(
                reranker.score([(requirement.text, claims[i].text) for i in candidates])
            )
            ordered = [
                candidates[i]
                for i in np.lexsort(
                    (np.arange(len(candidates)), -np.asarray(scores, dtype=float))
                )
            ]
            orders.append(ordered)
            used.append(rerank_w)
        matrix[r] = _unit_scale(rrf(orders, k=k, n=n_claim, weights=used)) * gate[r]
    return matrix, claim_sim
