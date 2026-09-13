"""Pydantic v2 models for the v2 index — the git-diffable source of truth.

The v2 index stores one YAML file per entity (see ``docs/v2-design.md`` §3).
These models define the schema, validate cross-entity references, and emit the
JSON Schema consumed by editors via ``ats-tailor schema``.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Date ranges are rendered with an ASCII hyphen (U+002D), never an en dash:
# en-dash ranges are a confirmed silent ATS import failure (Workday drops the
# range). v1's parse_start_date/parse_end_date scan for month+year tokens and
# are separator-agnostic, so the legacy projection is unaffected.
RANGE_SEPARATOR = " - "
MAX_BULLET_CHARS = 180

MONTH_ABBREV = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
MONTH_MAP = {name.lower(): i + 1 for i, name in enumerate(MONTH_ABBREV)}

_MONTH_YEAR_RE = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b")
_YEAR_RE = re.compile(r"\b(\d{4})\b")
_ENDPOINT_RE = re.compile(r"^\d{4}(-\d{2})?$")

Proficiency = Literal["advanced", "intermediate", "familiar"]
BulletKind = Literal["responsibility", "impact"]
RoleType = Literal["internship", "research", "fulltime"]
EvidenceType = Literal["role", "project", "cert"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Period ────────────────────────────────────────────────────────────────────
class Period(_Base):
    """A date range with year or year-month precision.

    ``start``/``end`` are ``YYYY`` or ``YYYY-MM``; ``end`` may also be
    ``present``. Rendering back to the v1 string form is done by :meth:`format`.
    """

    start: str
    end: str

    @field_validator("start", "end", mode="before")
    @classmethod
    def _coerce(cls, value):
        # YAML parses a bare 4-digit year as an int.
        if isinstance(value, int):
            return f"{value:04d}"
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("start")
    @classmethod
    def _check_start(cls, value: str) -> str:
        if not _ENDPOINT_RE.match(value):
            raise ValueError(f"start must be YYYY or YYYY-MM, got {value!r}")
        _check_month(value, "start")
        return value

    @field_validator("end")
    @classmethod
    def _check_end(cls, value: str) -> str:
        if value.lower() == "present":
            return "present"
        if not _ENDPOINT_RE.match(value):
            raise ValueError(f"end must be YYYY, YYYY-MM or 'present', got {value!r}")
        _check_month(value, "end")
        return value

    # ── formatting ──
    @staticmethod
    def _format_endpoint(value: str) -> str:
        if value == "present":
            return "Present"
        if "-" in value:
            year, month = value.split("-")
            return f"{MONTH_ABBREV[int(month) - 1]} {year}"
        return value

    def format(self) -> str:
        """Render as ``Mon YYYY - Mon YYYY`` / ``- Present`` / ``YYYY - YYYY``.

        The separator is an ASCII hyphen with spaces; see
        :data:`RANGE_SEPARATOR` for why.
        """
        return (
            f"{self._format_endpoint(self.start)}{RANGE_SEPARATOR}"
            f"{self._format_endpoint(self.end)}"
        )

    def start_date(self) -> datetime:
        """Earliest instant covered; year-only precision resolves to January."""
        if "-" in self.start:
            year, month = self.start.split("-")
            return datetime(int(year), int(month), 1)
        return datetime(int(self.start), 1, 1)

    def end_date(self) -> datetime:
        """Latest instant covered; year-only resolves to December, present to now."""
        if self.end == "present":
            return datetime.now()
        if "-" in self.end:
            year, month = self.end.split("-")
            return datetime(int(year), int(month), 1)
        return datetime(int(self.end), 12, 1)

    @classmethod
    def parse(cls, text: str) -> "Period":
        """Parse a period string such as ``"Sep 2024 - Apr 2027"`` or ``"2017 - 2018"``.

        Parsing scans for month+year and bare-year tokens rather than splitting
        on a separator, so ``-``, ``–``, ``—`` and ``to`` all work, as do the
        abbreviated, dotted and full month names v1 accepted.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"cannot parse period from {text!r}")
        raw = text.strip()
        month_years = _MONTH_YEAR_RE.findall(raw)
        years = _YEAR_RE.findall(raw)

        start = _endpoint_from(month_years[0]) if month_years else None
        if start is None and years:
            start = years[0]
        if start is None:
            raise ValueError(f"cannot parse period from {text!r}: no year found")

        if "present" in raw.lower():
            end = "present"
        elif month_years:
            end = _endpoint_from(month_years[-1])
        else:
            end = years[-1]

        return cls(start=start, end=end)


def _check_month(value: str, field: str) -> None:
    if "-" in value:
        month = int(value.split("-")[1])
        if not 1 <= month <= 12:
            raise ValueError(f"{field} month must be 01-12, got {value!r}")


def _endpoint_from(match: tuple[str, str]) -> str:
    month_name, year = match
    month = MONTH_MAP.get(month_name.lower()[:3])
    return f"{year}-{month:02d}" if month else year


# ── Content entities ──────────────────────────────────────────────────────────
class Bullet(_Base):
    id: str
    kind: BulletKind
    text: str
    skills: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    variants: dict[str, str] = Field(default_factory=dict)


class Evidence(_Base):
    ref: str
    type: EvidenceType
    detail: str | None = None


class Skill(_Base):
    id: str
    name: str
    proficiency: Proficiency
    aliases: list[str] = Field(default_factory=list)
    acronym: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    def surface_forms(self) -> list[str]:
        """Name, aliases and acronym, deduped case-insensitively, order preserved."""
        return _dedupe([self.name, *self.aliases, *( [self.acronym] if self.acronym else [] )])


class SkillCategory(_Base):
    category: str
    skills: list[Skill] = Field(default_factory=list)


class Project(_Base):
    id: str
    name: str
    type: str | None = None
    repo: str | None = None
    period: Period
    summary: str
    #: Short phrase for the title line ("Name -- tagline"). When absent the
    #: renderer derives one from ``summary`` by cutting at a clause boundary.
    tagline: str | None = None
    tech_stack: dict[str, list[str]] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    bullets: list[Bullet] = Field(default_factory=list)


class Role(_Base):
    id: str
    company: str
    title: str
    type: RoleType
    period: Period
    hidden: bool = False
    skills: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    bullets: list[Bullet] = Field(default_factory=list)


class Certification(_Base):
    id: str
    name: str
    issuer: str
    date: str | None = None
    skills: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    relevance: list[str] = Field(default_factory=list)


class Education(_Base):
    institution: str
    degree: str
    field: str
    period: Period


class Summaries(_Base):
    default: str = ""
    variants: dict[str, str] = Field(default_factory=dict)
    highlights: list[str] = Field(default_factory=list)


# ── File-shaped wrappers (one per top-level YAML file) ────────────────────────
class CertificationsFile(_Base):
    certifications: list[Certification] = Field(default_factory=list)


class EducationFile(_Base):
    education: list[Education] = Field(default_factory=list)


# ── Whole index ───────────────────────────────────────────────────────────────
class Index(_Base):
    skills: list[SkillCategory] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    roles: list[Role] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    summaries: Summaries = Field(default_factory=Summaries)

    # ── lookups ──
    def all_skills(self) -> list[Skill]:
        return [skill for category in self.skills for skill in category.skills]

    def skill_map(self) -> dict[str, Skill]:
        return {skill.id: skill for skill in self.all_skills()}

    def item_kinds(self) -> dict[str, str]:
        """id → ``role`` | ``project`` | ``cert`` for every referenceable item."""
        kinds: dict[str, str] = {}
        for role in self.roles:
            kinds[role.id] = "role"
        for project in self.projects:
            kinds[project.id] = "project"
        for cert in self.certifications:
            kinds[cert.id] = "cert"
        return kinds

    # ── validation ──
    @model_validator(mode="after")
    def _validate_index(self) -> "Index":
        errors: list[str] = []

        seen_items: dict[str, str] = {}
        for kind, items in (
            ("roles", self.roles),
            ("projects", self.projects),
            ("certifications", self.certifications),
        ):
            for i, item in enumerate(items):
                path = f"{kind}[{i}] ({item.id})"
                if item.id in seen_items:
                    errors.append(
                        f"{path}: duplicate id, already used by {seen_items[item.id]}"
                    )
                else:
                    seen_items[item.id] = path

        seen_skills: dict[str, str] = {}
        for ci, category in enumerate(self.skills):
            for si, skill in enumerate(category.skills):
                path = f"skills[{ci}] ({category.category}) skills[{si}] ({skill.id})"
                if skill.id in seen_skills:
                    errors.append(
                        f"{path}: duplicate skill id, already used by {seen_skills[skill.id]}"
                    )
                else:
                    seen_skills[skill.id] = path

        for ci, category in enumerate(self.skills):
            for si, skill in enumerate(category.skills):
                for ei, ev in enumerate(skill.evidence):
                    path = (
                        f"skills[{ci}] ({category.category}) skills[{si}] "
                        f"({skill.id}) evidence[{ei}]"
                    )
                    if ev.ref not in seen_items:
                        errors.append(f"{path}: unknown evidence ref '{ev.ref}'")

        seen_bullets: dict[str, str] = {}
        for kind, items in (("roles", self.roles), ("projects", self.projects)):
            for i, item in enumerate(items):
                path = f"{kind}[{i}] ({item.id})"
                errors.extend(_check_skill_refs(item.skills, seen_skills, path))
                for bi, bullet in enumerate(item.bullets):
                    bpath = f"{path} bullets[{bi}] ({bullet.id})"
                    if bullet.id in seen_bullets:
                        errors.append(
                            f"{bpath}: duplicate bullet id, already used by "
                            f"{seen_bullets[bullet.id]}"
                        )
                    else:
                        seen_bullets[bullet.id] = bpath
                    errors.extend(_check_skill_refs(bullet.skills, seen_skills, bpath))

        for i, cert in enumerate(self.certifications):
            path = f"certifications[{i}] ({cert.id})"
            errors.extend(_check_skill_refs(cert.skills, seen_skills, path))

        for i, role in enumerate(self.roles):
            if not role.hidden and not role.bullets:
                errors.append(
                    f"roles[{i}] ({role.id}): visible role must have at least one "
                    f"bullet (set hidden: true instead)"
                )

        if errors:
            raise ValueError("\n".join(errors))
        return self

    def warnings(self) -> list[str]:
        """Non-fatal issues worth surfacing to the author."""
        notes: list[str] = []

        for kind, items in (("roles", self.roles), ("projects", self.projects)):
            for i, item in enumerate(items):
                path = f"{kind}[{i}] ({item.id})"
                if not item.skills and not item.keywords:
                    notes.append(f"{path}: no skills and no keywords — nothing to match on")
                for bi, bullet in enumerate(item.bullets):
                    if len(bullet.text) > MAX_BULLET_CHARS:
                        notes.append(
                            f"{path} bullets[{bi}] ({bullet.id}): text is "
                            f"{len(bullet.text)} chars (> {MAX_BULLET_CHARS})"
                        )

        for i, project in enumerate(self.projects):
            if not any(b.kind == "impact" for b in project.bullets):
                notes.append(f"projects[{i}] ({project.id}): no impact bullet")

        for i, cert in enumerate(self.certifications):
            if not cert.skills and not cert.keywords:
                notes.append(
                    f"certifications[{i}] ({cert.id}): no skills and no keywords"
                )

        for ci, category in enumerate(self.skills):
            for si, skill in enumerate(category.skills):
                if not skill.evidence:
                    notes.append(
                        f"skills[{ci}] ({category.category}) skills[{si}] "
                        f"({skill.id}): no evidence"
                    )

        return notes


def _check_skill_refs(refs: list[str], known: dict[str, str], path: str) -> list[str]:
    return [f"{path}: unknown skill id '{ref}'" for ref in refs if ref not in known]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        key = value.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
