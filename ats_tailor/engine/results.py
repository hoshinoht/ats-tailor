"""What one pipeline run produced, for callers that need it in-process.

Shared by both engines so ``eval``, ``check`` and the CLI never have to ask
which one ran. The v2 fields default to empty, and ``to_dict`` omits them, so a
v1 ``selection.json`` keeps exactly the shape it had.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """One run's outputs and the choices behind them.

    ``role_ids``/``project_ids`` are in *score* order (the order selection
    ranked them), not the chronological order the page renders, so evaluation
    can measure ranking quality. ``entries`` is what ``atscheck.parse_check``
    needs to verify each rendered entry's date line.
    """

    out_dir: Path
    tex_path: Path
    company: str
    role: str
    role_ids: list[str] = field(default_factory=list)
    project_ids: list[str] = field(default_factory=list)
    cert_ids: list[str] = field(default_factory=list)
    skill_lines: list[dict] = field(default_factory=list)
    entries: list[dict] = field(default_factory=list)
    engine: str = "v1"
    #: v2 only — the solver's answer, the parsed posting, and the compile loop.
    selection: dict[str, Any] | None = None
    requirements: list[dict] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    pages: int | None = None
    budget: float | None = None
    iterations: int = 0
    #: v2 only — what reading the posting produced before any scoring: the
    #: hard filters, the title the Summary targets, the archetype the summary
    #: variant was chosen by, and who did the reading.
    gates: dict[str, Any] = field(default_factory=dict)
    jd_title: str | None = None
    archetype: str | None = None
    provider: str | None = None
    #: Whether the profile came out of the store rather than being read again.
    provider_cached: bool = False
    #: v2 only — the accepted phrasing variants this run used, one entry per
    #: bullet that reached the page in wording other than its own, with the
    #: guardrail verdict that let it in. Empty (and omitted from ``to_dict``)
    #: for an index with no variants, which is every index before Phase 5.
    variants: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "company": self.company,
            "role": self.role,
            "out_dir": str(self.out_dir),
            "tex": self.tex_path.name,
            "roles": self.role_ids,
            "projects": self.project_ids,
            "certifications": self.cert_ids,
            "skill_lines": self.skill_lines,
            "entries": self.entries,
        }
        if self.engine != "v1":
            payload["engine"] = self.engine
        if self.selection is not None:
            payload["requirements"] = self.requirements
            payload["coverage"] = self.coverage
            payload["selection"] = self.selection
            payload["trace"] = self.selection.get("trace", [])
            payload["budget"] = self.budget
            payload["pages"] = self.pages
            payload["iterations"] = self.iterations
            payload["gates"] = self.gates
            payload["jd_title"] = self.jd_title
            payload["archetype"] = self.archetype
            payload["provider"] = self.provider
            payload["provider_cached"] = self.provider_cached
            if self.variants:
                payload["variants"] = self.variants
        return payload
