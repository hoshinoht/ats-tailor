"""Metrics over abstract selections — no knowledge of what was selected.

These are the numbers an eval harness averages: how much of the requirement
weight a selection actually covered, how many of the caller's pinned ids the
ranking put on top, and how well an ordering agrees with a reference one.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .types import Requirement


def coverage_at_budget(
    coverage: Mapping[str, float],
    requirements: Sequence[Requirement],
    threshold: float = 0.3,
    kind_weights: Mapping[str, float] | None = None,
) -> float | None:
    """Weighted share of requirements whose coverage reached ``threshold``.

    ``None`` when there is nothing to cover — a ratio over an empty set is not
    a number worth averaging.
    """
    if not requirements:
        return None
    total = 0.0
    hit = 0.0
    for requirement in requirements:
        weight = float(requirement.weight)
        if kind_weights:
            weight *= float(kind_weights.get(requirement.kind, 1.0))
        total += weight
        if float(coverage.get(requirement.id, 0.0)) >= threshold:
            hit += weight
    return hit / total if total else None


def precision_at_k(ranked: Sequence[str], pins: Sequence[str]) -> float | None:
    """Share of the top-``len(pins)`` ranked ids that are pinned."""
    if not pins:
        return None
    k = len(pins)
    return len(set(ranked[:k]) & set(pins)) / k


def kendall_tau(order: Sequence[str], reference: Sequence[str]) -> float | None:
    """Kendall τ-a between two strict orders, over the ids they share.

    ``None`` when fewer than two ids overlap — a correlation over one point is
    not a number worth averaging.
    """
    shared = [item for item in reference if item in set(order)]
    if len(shared) < 2:
        return None
    rank = {item: i for i, item in enumerate(order)}
    concordant = discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            delta = rank[shared[i]] - rank[shared[j]]
            if delta < 0:
                concordant += 1
            elif delta > 0:
                discordant += 1
    pairs = len(shared) * (len(shared) - 1) / 2
    return (concordant - discordant) / pairs
