"""Per-application overrides: the human's half of the loop.

The solver decides what covers the posting best; the person applying knows
things the index cannot encode — that this project is the one the interviewer
will ask about, that this bullet reads badly next to that one, that the posting
means something narrower than it says. An ``overrides.yaml`` next to the run's
output says so, and the next run of the same output directory honours it.

Everything here is *per run*: nothing an override says is written back into the
index. What survives is the record — the file is copied into the run record,
and ``ats-tailor labels from-run`` promotes the pins and exclusions into the
eval labels, so a real decision becomes ground truth the scoring is measured
against.

Overrides are hard: a pin is shown even if it covers nothing, an exclusion is
never shown. The one thing this module will not do is stay quiet when an edited
bullet drops a number the source bullet claimed — see :func:`metric_warnings`.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

from ats_match import Requirement

#: The file the pipeline looks for inside a run's output directory.
OVERRIDES_FILENAME = "overrides.yaml"

#: The claim id prefix ``adapt.skill_claims`` gives a skill.
SKILL_CLAIM_PREFIX = "skill:"

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

HEADER = """\
# Per-run overrides for this output directory — hand-edited, applied on the
# next `ats-tailor tailor --engine v2 --output <this dir>`. Ids are index ids
# (roles/projects), bullet ids, and skill ids. See docs/plans/phase-3b.md §3.
"""


class RequirementAdd(BaseModel):
    """A requirement the extractor missed, in the extractor's own shape."""

    model_config = ConfigDict(extra="forbid")

    text: str
    kind: str = "must"
    weight: float = 1.0
    terms: list[str] = Field(default_factory=list)
    id: str | None = None


class RequirementEdits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    add: list[RequirementAdd] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)
    reweight: dict[str, float] = Field(default_factory=dict)

    def empty(self) -> bool:
        return not (self.add or self.remove or self.reweight)


class Overrides(BaseModel):
    """What one application wants done differently. Every field is optional."""

    model_config = ConfigDict(extra="forbid")

    #: Role/project ids that must be shown, and ids that must never be.
    pin_items: list[str] = Field(default_factory=list)
    exclude_items: list[str] = Field(default_factory=list)
    #: Bullet ids. Pinning a bullet pins its item — a bullet has nowhere to go
    #: without the entry it belongs to.
    pin_bullets: list[str] = Field(default_factory=list)
    exclude_bullets: list[str] = Field(default_factory=list)
    #: bullet id → the phrasing to use this time. The source facts must survive
    #: the edit; :func:`metric_warnings` says so when they do not.
    bullet_text: dict[str, str] = Field(default_factory=dict)
    #: Skill ids for the Skills section.
    skills_pin: list[str] = Field(default_factory=list)
    skills_exclude: list[str] = Field(default_factory=list)
    #: Replacement summary text; ``""`` drops the Summary section entirely.
    summary: str | None = None
    requirements: RequirementEdits = Field(default_factory=RequirementEdits)
    #: Claim-line budget, replacing the one computed from ``ATS_MAX_PAGE_LINES``.
    budget_lines: int | None = None

    def empty(self) -> bool:
        return not (
            self.pin_items or self.exclude_items or self.pin_bullets
            or self.exclude_bullets or self.bullet_text or self.skills_pin
            or self.skills_exclude or self.summary is not None
            or not self.requirements.empty() or self.budget_lines is not None
        )

    def summary_line(self) -> str:
        """One line naming what is set, for logs and the run record."""
        parts = []
        for name, values in (
            ("pin", self.pin_items), ("exclude", self.exclude_items),
            ("pin-bullet", self.pin_bullets), ("exclude-bullet", self.exclude_bullets),
            ("text", list(self.bullet_text)), ("skills-pin", self.skills_pin),
            ("skills-exclude", self.skills_exclude),
        ):
            if values:
                parts.append(f"{name}: {', '.join(values)}")
        if self.summary is not None:
            parts.append("summary: replaced")
        if not self.requirements.empty():
            parts.append(
                f"requirements: +{len(self.requirements.add)} "
                f"-{len(self.requirements.remove)} "
                f"~{len(self.requirements.reweight)}"
            )
        if self.budget_lines is not None:
            parts.append(f"budget_lines: {self.budget_lines}")
        return "; ".join(parts) or "nothing set"


# ── Reading and writing the file ─────────────────────────────────────────────
def _yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def overrides_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / OVERRIDES_FILENAME


def load(path: str | Path) -> Overrides:
    """Validate one overrides file. An empty file is an empty override set."""
    path = Path(path)
    data = _yaml().load(path.read_text())
    if data is None:
        return Overrides()
    if not isinstance(data, dict):
        raise ValueError(f"{path}: file must contain a mapping at the top level")
    return Overrides.model_validate(_plain(data))


def find(out_dir: str | Path, explicit: str | Path | None = None) -> tuple[Overrides, Path | None]:
    """``(overrides, path)`` for a run: ``--overrides`` if given, else the output dir."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"overrides file not found: {path}")
        return load(path), path
    path = overrides_path(out_dir)
    if path.is_file():
        return load(path), path
    return Overrides(), None


def save(path: str | Path, overrides: Overrides) -> Path:
    """Round-trip write: an existing file keeps its comments and key order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = _yaml()
    data = overrides.model_dump(mode="python")
    if path.exists():
        existing = yaml.load(path.read_text())
        if isinstance(existing, dict):
            data = _merge_into(existing, data)
    buffer = io.StringIO()
    yaml.dump(data, buffer)
    text = buffer.getvalue()
    if not text.lstrip().startswith("#"):
        text = f"{HEADER}\n{text}"
    path.write_text(text)
    return path


def _merge_into(target: Any, source: Any) -> Any:
    """Update ``target`` in place from ``source``, keeping ruamel's comments."""
    if isinstance(target, dict) and isinstance(source, dict):
        for key, value in source.items():
            target[key] = _merge_into(target[key], value) if key in target else value
        for key in [k for k in target if k not in source]:
            del target[key]
        return target
    if isinstance(target, list) and isinstance(source, list):
        for i, value in enumerate(source):
            if i < len(target):
                target[i] = _merge_into(target[i], value)
            else:
                target.append(value)
        while len(target) > len(source):
            target.pop()
        return target
    return source


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def edit(
    path: str | Path,
    pin: Sequence[str] = (),
    exclude: Sequence[str] = (),
    pin_bullet: Sequence[str] = (),
    exclude_bullet: Sequence[str] = (),
    text: Mapping[str, str] | None = None,
    summary: str | None = None,
    budget_lines: int | None = None,
) -> tuple[Overrides, list[str]]:
    """Apply CLI edits to (or create) an overrides file; returns what changed."""
    path = Path(path)
    current = load(path) if path.is_file() else Overrides()
    changes: list[str] = []

    def extend(field_name: str, values: Sequence[str]) -> None:
        existing = list(getattr(current, field_name))
        for value in values:
            value = str(value).strip()
            if value and value not in existing:
                existing.append(value)
                changes.append(f"+{field_name}: {value}")
        setattr(current, field_name, existing)

    extend("pin_items", pin)
    extend("exclude_items", exclude)
    extend("pin_bullets", pin_bullet)
    extend("exclude_bullets", exclude_bullet)
    for bullet_id, replacement in (text or {}).items():
        if current.bullet_text.get(bullet_id) != replacement:
            current.bullet_text[bullet_id] = replacement
            changes.append(f"~bullet_text: {bullet_id}")
    if summary is not None and current.summary != summary:
        current.summary = summary
        changes.append("~summary")
    if budget_lines is not None and current.budget_lines != budget_lines:
        current.budget_lines = budget_lines
        changes.append(f"~budget_lines: {budget_lines}")

    save(path, current)
    return current, changes


# ── Mapping onto the solver ──────────────────────────────────────────────────
@dataclass
class ResolvedOverrides:
    """The overrides in the vocabulary ``ats_match.Constraints`` speaks."""

    pin_claims: set[str] = field(default_factory=set)
    pin_groups: set[str] = field(default_factory=set)
    exclude_claims: set[str] = field(default_factory=set)
    exclude_groups: set[str] = field(default_factory=set)
    #: Ids naming nothing in the index — reported, never fatal.
    unknown: list[str] = field(default_factory=list)

    def warnings(self) -> list[str]:
        return [f"overrides: unknown id '{i}' (nothing in the index has it)"
                for i in self.unknown]


def resolve(overrides: Overrides, index) -> ResolvedOverrides:
    """Turn ids into claim/group constraints, collecting the ones that miss.

    Pinning a bullet also pins its item: ``ats_match`` opens a group when one of
    its claims is pinned, and an entry that shows one bullet is an entry that is
    shown. Excluding an *item* excludes its group, which takes its bullets with
    it; excluding a bullet leaves the entry alone.
    """
    item_kinds = index.item_kinds()
    bullet_group = {
        bullet.id: item.id
        for item in [*index.roles, *index.projects]
        for bullet in item.bullets
    }
    skills = index.skill_map()
    resolved = ResolvedOverrides()

    for item_id in overrides.pin_items:
        if item_kinds.get(item_id) in {"role", "project"}:
            resolved.pin_groups.add(item_id)
        else:
            resolved.unknown.append(item_id)
    for item_id in overrides.exclude_items:
        if item_kinds.get(item_id) in {"role", "project"}:
            resolved.exclude_groups.add(item_id)
        else:
            resolved.unknown.append(item_id)

    for bullet_id in overrides.pin_bullets:
        group = bullet_group.get(bullet_id)
        if group is None:
            resolved.unknown.append(bullet_id)
            continue
        resolved.pin_claims.add(bullet_id)
        resolved.pin_groups.add(group)
    for bullet_id in overrides.exclude_bullets:
        if bullet_id in bullet_group:
            resolved.exclude_claims.add(bullet_id)
        else:
            resolved.unknown.append(bullet_id)

    for skill_id in overrides.skills_pin:
        if skill_id in skills:
            resolved.pin_claims.add(f"{SKILL_CLAIM_PREFIX}{skill_id}")
        else:
            resolved.unknown.append(skill_id)
    for skill_id in overrides.skills_exclude:
        if skill_id in skills:
            resolved.exclude_claims.add(f"{SKILL_CLAIM_PREFIX}{skill_id}")
        else:
            resolved.unknown.append(skill_id)

    # An id both pinned and excluded is a contradiction the user has to fix;
    # exclusion wins, because showing something they said to drop is the worse
    # of the two failures.
    resolved.pin_groups -= resolved.exclude_groups
    resolved.pin_claims -= resolved.exclude_claims
    return resolved


# ── Requirement edits ────────────────────────────────────────────────────────
def apply_requirements(
    requirements: Sequence[Requirement], edits: RequirementEdits
) -> list[Requirement]:
    """Remove, reweight, and append requirements after extraction.

    Ids are the extractor's (``req-3``); added requirements get ``req-add-N``
    so an edit never collides with a re-extraction's numbering.
    """
    if edits.empty():
        return list(requirements)
    removed = {str(i).strip() for i in edits.remove}
    reweight = {str(k).strip(): float(v) for k, v in edits.reweight.items()}
    kept = [
        Requirement(
            id=r.id, text=r.text, kind=r.kind,
            weight=reweight.get(r.id, r.weight), terms=r.terms, category=r.category,
        )
        for r in requirements
        if r.id not in removed
    ]
    for i, addition in enumerate(edits.add, start=1):
        kept.append(Requirement(
            id=addition.id or f"req-add-{i}",
            text=addition.text,
            kind=addition.kind if addition.kind in {"must", "nice", "responsibility"}
            else "must",
            weight=float(addition.weight),
            terms=tuple(str(t) for t in addition.terms),
        ))
    return kept


# ── The metric guardrail ─────────────────────────────────────────────────────
def metric_warnings(index, bullet_text: Mapping[str, str]) -> list[str]:
    """Warn when an edited bullet drops a number the source bullet claimed.

    The light version of the grounding check Phase 5 puts in front of the LLM
    rewriter. It never blocks: the user is editing their own resume and may have
    good reason to drop a figure. It does insist on saying so, because a
    rephrase that quietly loses "40%" is the one edit that turns a measured
    achievement into a claim about nothing.
    """
    bullets = {
        bullet.id: bullet
        for item in [*index.roles, *index.projects]
        for bullet in item.bullets
    }
    warnings: list[str] = []
    for bullet_id, replacement in bullet_text.items():
        bullet = bullets.get(bullet_id)
        if bullet is None:
            warnings.append(f"overrides: unknown bullet id '{bullet_id}'")
            continue
        missing = [m for m in bullet.metrics if not _mentions(replacement, m)]
        if missing:
            warnings.append(
                f"overrides: {bullet_id} drops {', '.join(repr(m) for m in missing)} "
                f"from the source bullet's metrics"
            )
    return warnings


def _mentions(text: str, metric: str) -> bool:
    """Whether ``text`` still carries ``metric``, literally or by its numbers."""
    haystack = _squash(text)
    needle = _squash(metric)
    if needle and needle in haystack:
        return True
    numbers = _NUMBER_RE.findall(metric)
    if not numbers:
        return False
    return all(number in haystack for number in numbers)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# ── Applying the text edits ──────────────────────────────────────────────────
def text_map(overrides: Overrides) -> dict[str, str]:
    """The ``bullet_text`` edits, whitespace-normalised, empties dropped."""
    return {
        str(bullet_id): str(replacement).strip()
        for bullet_id, replacement in overrides.bullet_text.items()
        if str(replacement).strip()
    }
