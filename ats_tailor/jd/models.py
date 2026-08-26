"""What a posting is, once something has read it.

A :class:`JDProfile` is the whole of what the pipeline knows about a job
description: the title it is advertising, the archetype it belongs to, the
gates that decide whether applying is even possible, and the atomic
requirements the selection is scored against.

Every field is validated, because a profile can come from a local model, from
an agent through the ``external`` provider, or from a human editing the JSON —
and none of those three is trusted to produce well-formed output. The posting
itself is data, never instruction (``docs/v2-design.md`` §7).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ats_match import Requirement

#: Verdicts a gate can return. ``flag`` is "look at this yourself": either the
#: posting says something conditional, or a model and the regexes disagree.
Verdict = Literal["pass", "fail", "flag", "unknown"]
#: The gates, in the order they are printed.
GATE_NAMES: tuple[str, ...] = (
    "eligibility", "language", "location", "deadline", "seniority",
)
#: Requirement kinds, matching ``ats_match.types.Kind``.
KINDS: tuple[str, ...] = ("must", "nice", "responsibility")
#: The band an *extracted* weight is clamped into (:func:`extract.parse_profile`).
#: Below 0.5 nothing is being asked; above 1.5 one line of a posting outweighs
#: three others, which no extractor knows. The model is deliberately not the
#: same as the field's own validation: the deterministic parser weights a
#: requirement that names nothing at ``ATS_SOFT_REQ_WEIGHT`` (0.25 by default),
#: and clamping *that* would quietly quadruple the page's boilerplate budget.
WEIGHT_RANGE = (0.5, 1.5)

_WHITESPACE = re.compile(r"\s+")


def squash(text: Any) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


class Gate(BaseModel):
    """One hard filter, with the posting's own words as evidence.

    A gate never blocks a run. It is printed before any scoring so the decision
    to spend an afternoon on an application is made with the citizenship clause
    in view, rather than after the PDF is already attached to an email.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict = "unknown"
    quote: str | None = None
    note: str | None = None
    #: ``deadline`` only: the ISO date the quote was parsed into.
    date: str | None = None

    @field_validator("quote", "note", "date", mode="before")
    @classmethod
    def _clean(cls, value):
        text = squash(value)
        return text or None

    @property
    def blocking(self) -> bool:
        """Whether this gate is something the user has to look at."""
        return self.verdict in {"fail", "flag"}

    def line(self, name: str) -> str:
        """One printable line: ``eligibility  FAIL  "…quote…"``."""
        parts = [f"{name:<12} {self.verdict.upper():<7}"]
        if self.date:
            parts.append(f"[{self.date}]")
        if self.note:
            parts.append(self.note)
        if self.quote:
            parts.append(f'"{_truncate(self.quote)}"')
        return "  ".join(parts)


def _truncate(text: str, limit: int = 120) -> str:
    text = squash(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class Gates(BaseModel):
    """The five gates of ``docs/v2-design.md`` §5."""

    model_config = ConfigDict(extra="forbid")

    eligibility: Gate = Field(default_factory=Gate)
    language: Gate = Field(default_factory=Gate)
    location: Gate = Field(default_factory=Gate)
    deadline: Gate = Field(default_factory=Gate)
    seniority: Gate = Field(default_factory=Gate)

    def items(self) -> list[tuple[str, Gate]]:
        return [(name, getattr(self, name)) for name in GATE_NAMES]

    def blocking(self) -> list[tuple[str, Gate]]:
        return [(name, gate) for name, gate in self.items() if gate.blocking]

    def render(self) -> str:
        """The block printed before scoring, and written into the report."""
        rows = [gate.line(name) for name, gate in self.items()]
        return "\n".join(rows)

    def summary_line(self) -> str:
        blocking = self.blocking()
        if not blocking:
            return "gates: nothing blocking"
        return "gates: " + ", ".join(
            f"{name} {gate.verdict}" for name, gate in blocking
        )


class RequirementSpec(BaseModel):
    """One atomic thing the posting asks for.

    ``terms`` are the posting's own technology words, normalised through the
    index alias map where they are known; an unknown term is *kept*, because a
    requirement the index cannot cover still has to be visible as one.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    text: str
    kind: Literal["must", "nice", "responsibility"] = "must"
    category: str | None = None
    weight: float = 1.0
    terms: list[str] = Field(default_factory=list)

    @field_validator("text", mode="before")
    @classmethod
    def _clean_text(cls, value):
        return squash(value)

    @field_validator("kind", mode="before")
    @classmethod
    def _clean_kind(cls, value):
        kind = squash(value).lower()
        if kind in KINDS:
            return kind
        # A model asked for the posting's own sectioning answers with things
        # like "required" or "responsibilities"; map what is obvious and let
        # the rest be a must-have, which is the safe direction to be wrong in.
        if kind.startswith(("respons", "duty", "duties", "day")):
            return "responsibility"
        if kind.startswith(("nice", "prefer", "bonus", "desir", "plus", "good")):
            return "nice"
        return "must"

    @field_validator("weight", mode="before")
    @classmethod
    def _clean_weight(cls, value):
        """Any non-negative number. See :data:`WEIGHT_RANGE` for why not 0.5."""
        try:
            weight = float(value)
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, weight)

    @field_validator("terms", mode="before")
    @classmethod
    def _clean_terms(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [squash(term) for term in value if squash(term)]

    @field_validator("category", mode="before")
    @classmethod
    def _clean_category(cls, value):
        return squash(value) or None

    def to_requirement(self) -> Requirement:
        """The ``ats_match`` shape the solver consumes."""
        return Requirement(
            id=self.id, text=self.text, kind=self.kind,
            weight=float(self.weight), terms=tuple(self.terms),
            category=self.category,
        )


class JDProfile(BaseModel):
    """Everything read out of one posting, whoever read it."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    archetype: str | None = None
    gates: Gates = Field(default_factory=Gates)
    requirements: list[RequirementSpec] = Field(default_factory=list)
    #: Who produced this — a provider name, or ``fallback`` for the
    #: deterministic parser. Recorded so an LLM profile and a parsed one are
    #: never mistaken for each other in the store or the run record.
    provider: str = "fallback"
    prompt_version: str = ""
    #: Set when the profile was read back from the store rather than extracted.
    cached: bool = False

    @field_validator("title", "archetype", mode="before")
    @classmethod
    def _clean(cls, value):
        return squash(value) or None

    def to_requirements(self) -> list[Requirement]:
        return [spec.to_requirement() for spec in self.requirements]

    def counts(self) -> dict[str, int]:
        return {kind: sum(1 for r in self.requirements if r.kind == kind)
                for kind in KINDS}

    def summary_line(self) -> str:
        counts = self.counts()
        source = f"{self.provider}{' (cached)' if self.cached else ''}"
        return (f"{len(self.requirements)} requirement(s) "
                f"[{counts['must']} must, {counts['nice']} nice, "
                f"{counts['responsibility']} responsibility] from {source}")

    def to_json(self) -> dict[str, Any]:
        """The shape stored in ``jd_profiles.extracted_json`` / ``edited_json``."""
        return self.model_dump(mode="json", exclude={"cached"})

    @classmethod
    def from_json(cls, payload: Any) -> "JDProfile | None":
        """Read a stored profile back, tolerating the Phase 3b row shape.

        3b wrote a bare list of requirement dicts (there was no profile yet).
        Those rows are still in the database, and a list is unambiguous, so it
        is read as a profile with no title and no gates rather than discarded.
        """
        if payload is None:
            return None
        if isinstance(payload, list):
            payload = {"requirements": payload}
        if not isinstance(payload, dict):
            return None
        try:
            return cls.model_validate(payload)
        except Exception:
            return None
