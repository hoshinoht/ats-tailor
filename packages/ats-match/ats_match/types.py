"""The vocabulary the matching core is written in.

Two sides: a :class:`Requirement` is something the target asks for, a
:class:`Claim` is one atomic piece of evidence that can be shown. Everything
else describes how many claims may be shown, what they cost, and how the
objective trades coverage against repetition.

Nothing here knows where claims come from or how they are displayed; the
calling application supplies ``section`` and ``group`` labels and their rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Kind = Literal["must", "nice", "responsibility"]

#: Default relative importance of each requirement kind.
KIND_WEIGHTS: Mapping[str, float] = {"must": 1.0, "responsibility": 0.7, "nice": 0.5}


@dataclass(frozen=True)
class Requirement:
    """One thing the target asks for.

    ``terms`` are canonical vocabulary tokens the requirement names, used by
    the keyword-in-context bonus; ``weight`` scales the kind weight, so an
    extractor confident about a line can push it above its peers.
    """

    id: str
    text: str
    kind: Kind
    weight: float = 1.0
    terms: tuple[str, ...] = ()
    category: str | None = None


@dataclass(frozen=True)
class Claim:
    """One atomic piece of evidence.

    ``group`` is the container a claim belongs to (claims of one group are
    shown together, under one header that costs ``SectionRule.group_cost``);
    ``section`` names the rule set that governs it. ``cost`` is what showing
    this claim costs on its own, in whatever unit ``Constraints.budget`` uses.
    """

    id: str
    text: str
    group: str
    section: str
    cost: float = 1.0
    terms: tuple[str, ...] = ()
    prior: float = 0.0
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SectionRule:
    """How many groups and claims one section may show, and what a group costs.

    ``required_attr`` names an attribute at least one claim of every opened
    group should carry when the group has such a claim available — the repair
    pass enforces it, never at the price of the budget.
    """

    min_groups: int = 0
    max_groups: int = 0
    min_claims: int = 0
    max_claims: int = 0
    group_cost: float = 0.0
    required_attr: str | None = None
    #: Gain a claim of this section must earn to be *chosen*; ``None`` falls
    #: back to ``Params.min_gain``. Free claims otherwise keep being taken for
    #: whatever their prior is worth, which fills a zero-cost section with
    #: things that answer nothing. The repair passes ignore it: reaching
    #: ``min_claims`` is a structural obligation, not a bid.
    min_gain: float | None = None


@dataclass
class Constraints:
    """The budget, the per-section rules, and the caller's hard overrides."""

    budget: float
    sections: dict[str, SectionRule] = field(default_factory=dict)
    pin_claims: set[str] = field(default_factory=set)
    pin_groups: set[str] = field(default_factory=set)
    exclude_claims: set[str] = field(default_factory=set)
    exclude_groups: set[str] = field(default_factory=set)


@dataclass
class Params:
    """Objective knobs.

    ``tau`` is the soft-pool temperature: as it goes to zero a requirement's
    coverage becomes the single best matching claim, and as it grows several
    mediocre claims start to add up. ``mu`` penalises near-duplicate claims
    inside one section, ``beta`` rewards must-terms that appear both in a
    ``skill``-like section and in ordinary prose.
    """

    tau: float = 0.1
    mu: float = 0.3
    #: Measured, not guessed: coverage is a *weighted sum* over every
    #: requirement while ``ctx`` is a single share in [0, 1], so a β of the
    #: order the design first suggested (0.2) never changes a decision. At 3.0
    #: it moves the keyword-in-context metric by ~9 points; past ~5 it starts
    #: buying context at the expense of coverage. See docs/plans/phase-3a.md.
    beta: float = 3.0
    kind_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(KIND_WEIGHTS)
    )
    #: The section whose claims count as the "named it outright" side of the
    #: keyword-in-context bonus; every other section is the prose side.
    context_section: str = "skill"
    #: Attribute holding the terms a claim would actually *show*, when that is
    #: narrower than the terms it is *retrieved* by (a claim can be tagged with
    #: vocabulary its own words never use). ``None`` uses ``Claim.terms``.
    context_terms_attr: str | None = None
    #: Similarity above which two claims of one section count as redundant.
    redundancy_floor: float = 0.8
    #: Whether near-duplicates *inside* the context section are penalised. A
    #: keyword line is meant to be a dense list of related terms, so charging
    #: it for similarity suppresses exactly the vocabulary it exists to carry.
    redundancy_in_context: bool = False
    #: Cost floor used when ranking by gain per unit cost, so a free claim
    #: gets a finite (and comparable) priority instead of an infinite one.
    min_cost: float = 0.25
    #: Default gain floor for choosing a claim; ``SectionRule.min_gain``
    #: overrides it per section.
    min_gain: float = 0.0
    #: Open every section up to ``max_groups`` (best group first, filled to
    #: ``min_claims``) before spending what is left on extra claims. Breadth
    #: is worth reserving budget for: the caller pays ``group_cost`` for a
    #: header it also *shows*, and a greedy that only ever deepens the groups
    #: it already opened ends with a hole too small to open another one.
    breadth_first: bool = True


@dataclass
class Step:
    """One greedy pick, kept so the choice can be explained afterwards."""

    claim_id: str
    gain: float
    opened_group: str | None = None
    covered: list[tuple[str, float]] = field(default_factory=list)
    repair: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "gain": round(float(self.gain), 6),
            "opened_group": self.opened_group,
            "covered": [[rid, round(float(d), 6)] for rid, d in self.covered],
            "repair": self.repair,
        }


@dataclass
class Selection:
    """What :func:`ats_match.select.select` decided, and why."""

    claim_ids: list[str] = field(default_factory=list)
    group_ids_by_section: dict[str, list[str]] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    objective: float = 0.0
    trace: list[Step] = field(default_factory=list)
    budget_used: float = 0.0

    def gain_of(self, claim_id: str) -> float:
        """The gain recorded for ``claim_id`` when it was picked (0 if never)."""
        for step in self.trace:
            if step.claim_id == claim_id:
                return float(step.gain)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": list(self.claim_ids),
            "groups": {k: list(v) for k, v in self.group_ids_by_section.items()},
            "coverage": {k: round(float(v), 6) for k, v in self.coverage.items()},
            "objective": round(float(self.objective), 6),
            "budget_used": round(float(self.budget_used), 4),
            "trace": [step.to_dict() for step in self.trace],
        }
