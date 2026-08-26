"""A pure matching core: cover weighted requirements with atomic claims.

See ``README.md``. Nothing in this package imports the application that uses
it, and nothing in it knows what the claims describe.
"""

from .evaluate import coverage_at_budget, kendall_tau, precision_at_k
from .fusion import build_matrix, ranks_from_scores, rrf
from .lexical import BM25, normalise, tokenize
from .providers import Embedder, Reranker
from .select import select
from .types import (
    KIND_WEIGHTS,
    Claim,
    Constraints,
    Params,
    Requirement,
    SectionRule,
    Selection,
    Step,
)

__all__ = [
    "BM25",
    "Claim",
    "Constraints",
    "Embedder",
    "KIND_WEIGHTS",
    "Params",
    "Requirement",
    "Reranker",
    "SectionRule",
    "Selection",
    "Step",
    "build_matrix",
    "coverage_at_budget",
    "kendall_tau",
    "normalise",
    "precision_at_k",
    "ranks_from_scores",
    "rrf",
    "select",
    "tokenize",
]
