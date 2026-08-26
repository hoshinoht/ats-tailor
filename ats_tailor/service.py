"""One in-process API. Every front-end goes through this file.

The CLI, the MCP server, and (in 4b) the TUI are all *adapters*: they parse
arguments, they format output, and in between they call a :class:`Service`
method. Nothing above this line knows about SQLite, ruamel, latexmk or the
solver, and nothing below it knows whether a human or an agent is asking.

Three rules shape the surface:

1. **Typed in, typed out.** Every method takes and returns dataclass-shaped
   pydantic models, never a dict with ad-hoc keys. An adapter that wants JSON
   calls ``model_dump(mode="json")`` and is done.
2. **One error type.** :class:`ServiceError` carries a code from
   :data:`ERROR_CODES`, so an adapter maps errors once instead of catching
   ``ValueError``, ``KeyError``, ``FileNotFoundError`` and ``ValidationError``
   at every call site.
3. **Silent.** A service method never prints. The pipeline is run with
   ``quiet=True`` and everything it would have said comes back in
   ``RunSummary.warnings`` — which is what an MCP server speaking JSON-RPC
   over stdout requires, and what a library ought to do anyway.

The posting is data, never instruction (``docs/v2-design.md`` §7). This file
enforces the half of that rule which is not about prompts: :meth:`Service.add_jd`
strips markup and invisible characters out of pasted text and *reports* what it
removed, and no method here ever fetches a URL.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import config
from .guardrail import (
    GuardrailReport,
    check_summary,
    check_variant,
    item_terms,
    source_metrics,
    source_technologies,
)
from .index.io import (
    detect_layout,
    load_index_v2,
    save_certifications,
    save_education,
    save_project,
    save_role,
    save_skill_category,
    save_summaries,
    slugify,
)
from .jd.hashing import jd_hash
from .jd.models import JDProfile
from .index.models import (
    Bullet,
    Certification,
    Education,
    Index,
    Project,
    Role,
    Skill,
    SkillCategory,
    Summaries,
)
from .runs.overrides import Overrides, overrides_path
from .runs.store import STATUSES, Run, Store, StatusError

#: The tracker vocabulary, re-exported so an adapter gets its whole surface
#: from this one module rather than reaching into ``runs.store``.
RUN_STATUSES: tuple[str, ...] = STATUSES

#: Codes an adapter has to handle. ``not_implemented`` is the Phase 5 stub.
ERROR_CODES: tuple[str, ...] = (
    "invalid_input",
    "not_found",
    "validation_failed",
    "compile_failed",
    "provider_unavailable",
    "conflict",
    "not_implemented",
)

#: Every id an adapter may pass in. Deliberately narrower than what the index
#: could hold: an id is used to build file paths, and this rules out traversal,
#: absolute paths, spaces and case confusion in one expression.
ID_RE = re.compile(r"^[a-z0-9._-]+$")

#: The kinds :meth:`Service.index_get` / :meth:`Service.index_put` address.
INDEX_KINDS: tuple[str, ...] = (
    "project", "role", "skill", "certification", "summaries", "education",
)

#: How many claims :meth:`Service.fit` names as strengths.
FIT_STRENGTHS = 5

#: The ``bullet_id`` a summary proposal is filed under. Summaries have no
#: bullet and there is exactly one of them, so a sentinel is honest where a
#: second table would be ceremony.
SUMMARY_TARGET = "summaries"

#: How many requirements a variant prompt is given. Three is what fits in a
#: sentence's worth of intent; more turns a rephrase into a rewrite.
PROPOSAL_REQUIREMENTS = 3

#: Longest archetype an adapter may pass. It is a skill-category name, not an
#: id, so it is not matched against :data:`ID_RE` — but it does become a YAML
#: key and a proposal id, and unbounded text in both is somebody's accident.
MAX_ARCHETYPE_CHARS = 60

#: Where :meth:`Service.add_jd` records what it knows about a posting that the
#: posting itself does not say: who it is with, where it came from, and what
#: was stripped out of it. Derived, gitignored, and safe to delete — the JD
#: files are the content.
JD_META_FILENAME = "jd_meta.json"


class ServiceError(Exception):
    """One error type, with a machine-readable code from :data:`ERROR_CODES`."""

    def __init__(self, code: str, message: str, details: Any = None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


# ── Models ───────────────────────────────────────────────────────────────────
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Sanitised(_Model):
    """What :meth:`Service.add_jd` removed from a pasted posting.

    Counts, not a diff: the point is that the user is told the text they pasted
    was not the text that got stored, and roughly how much of it was markup.
    """

    html_comments: int = 0
    script_blocks: int = 0
    style_blocks: int = 0
    html_tags: int = 0
    entities: int = 0
    invisible_chars: int = 0
    blank_lines: int = 0
    chars_removed: int = 0

    def empty(self) -> bool:
        return not any(self.model_dump().values())

    def summary_line(self) -> str:
        parts = [
            f"{value} {name.replace('_', ' ')}"
            for name, value in self.model_dump().items()
            if value and name != "chars_removed"
        ]
        return "; ".join(parts) or "nothing removed"


class JDInfo(_Model):
    """One posting on disk, and what is known about it without reading it."""

    slug: str
    path: str
    company: str | None = None
    role: str | None = None
    jd_hash: str = ""
    chars: int = 0
    has_profile: bool = False
    gates_summary: str | None = None
    deadline: str | None = None
    label_exists: bool = False
    source_url: str | None = None
    runs: int = 0
    sanitised: Sanitised | None = None


class FitReport(_Model):
    """Cheap triage: could this index cover this posting, before tailoring it.

    ``must_upper_bound`` is the *best achievable* coverage, not a prediction:
    it takes each must-have requirement's single best-matching claim and
    averages, ignoring the page budget, the redundancy penalty and every other
    thing that makes a real run worse than its ceiling. A low ceiling means no
    selection can save this application; a high one says nothing about whether
    one page is enough.
    """

    slug: str
    title: str | None = None
    archetype: str | None = None
    gates: dict[str, Any] = Field(default_factory=dict)
    gates_summary: str = ""
    blocking_gates: list[str] = Field(default_factory=list)
    requirements: int = 0
    must_requirements: int = 0
    nice_requirements: int = 0
    must_upper_bound: float | None = None
    nice_upper_bound: float | None = None
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    uncovered: list[str] = Field(default_factory=list)


class RunSummary(_Model):
    """What one tailoring run produced."""

    run_id: str | None = None
    out_dir: str
    company: str
    role: str
    engine: str = "v2"
    tex: str | None = None
    pages: int | None = None
    budget: float | None = None
    iterations: int = 0
    jd_slug: str | None = None
    jd_hash: str | None = None
    title: str | None = None
    archetype: str | None = None
    provider: str | None = None
    coverage: dict[str, float] = Field(default_factory=dict)
    selection: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, Any] = Field(default_factory=dict)
    check: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class RunNote(_Model):
    created_at: str
    note: str


class RunRecord(_Model):
    """A stored run: the row, its parsed payload, and its history."""

    id: str
    created_at: str
    company: str | None = None
    role: str | None = None
    jd_hash: str | None = None
    engine: str | None = None
    config_hash: str | None = None
    out_dir: str | None = None
    status: str = "drafted"
    final: bool = False
    pdf_sha256: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    notes: list[RunNote] = Field(default_factory=list)


class CompileSummary(_Model):
    pdf_path: str
    pages: int
    sha256: str | None = None


class CheckReport(_Model):
    """The ATS parse check and the keyword coverage table for one run."""

    run_id: str | None = None
    out_dir: str
    ok: bool = False
    pages: int | None = None
    parse: dict[str, Any] = Field(default_factory=dict)
    coverage: dict[str, Any] | None = None
    summary: str = ""


class SelectedClaim(_Model):
    bullet_id: str
    item: str
    section: str = ""
    text: str = ""
    covers: list[str] = Field(default_factory=list)


class ReviewPack(_Model):
    """Everything a reviewing agent needs about one run, in one payload.

    ``jd_text`` arrives trust-wrapped by the adapter, not here: the service
    returns the posting as text and the MCP server is the boundary that labels
    it untrusted, because it is the boundary a model reads across.
    """

    run_id: str
    company: str
    role: str
    out_dir: str
    jd_text: str = ""
    gates: dict[str, Any] = Field(default_factory=dict)
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    selected_claims: list[SelectedClaim] = Field(default_factory=list)
    coverage_table: list[dict[str, Any]] = Field(default_factory=list)
    keyword_table: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    skills_lines: list[dict[str, Any]] = Field(default_factory=list)


class EvalSummary(_Model):
    path: str
    config_hash: str
    table: str
    means: dict[str, Any] = Field(default_factory=dict)
    runs: int = 0
    skipped: list[str] = Field(default_factory=list)
    #: The variants section, when the index has variants or the store has
    #: proposals; empty otherwise, which is every sweep before Phase 5.
    variants: dict[str, Any] = Field(default_factory=dict)


class LabelUpdate(_Model):
    path: str
    changes: list[str] = Field(default_factory=list)


class OverridesPatch(_Model):
    """A partial edit of a run's ``overrides.yaml``.

    ``None`` means "leave this alone"; a list is *added* to what is already
    there (the same additive contract ``ats-tailor override`` has, so an agent
    and a human editing the same file never surprise each other), and
    ``summary`` / ``budget_lines`` replace.
    """

    pin_items: list[str] | None = None
    exclude_items: list[str] | None = None
    pin_bullets: list[str] | None = None
    exclude_bullets: list[str] | None = None
    skills_pin: list[str] | None = None
    skills_exclude: list[str] | None = None
    bullet_text: dict[str, str] | None = None
    #: requirement id → the weight this application gives it. A reweight is a
    #: statement about *this* posting ("they say Kubernetes twice, believe
    #: them"), which is why it lives beside the pins and not in the index.
    requirement_reweight: dict[str, float] | None = None
    summary: str | None = None
    budget_lines: int | None = None

    def empty(self) -> bool:
        return all(value is None for value in self.model_dump().values())

    def apply_to(self, current: Overrides) -> tuple[Overrides, list[str]]:
        """Merge this patch into ``current``; returns the result and what changed."""
        updated = current.model_copy(deep=True)
        changes: list[str] = []
        for field_name in ("pin_items", "exclude_items", "pin_bullets",
                           "exclude_bullets", "skills_pin", "skills_exclude"):
            values = getattr(self, field_name)
            if values is None:
                continue
            existing = list(getattr(updated, field_name))
            for value in values:
                value = str(value).strip()
                if value and value not in existing:
                    existing.append(value)
                    changes.append(f"+{field_name}: {value}")
            setattr(updated, field_name, existing)
        if self.bullet_text is not None:
            for bullet_id, text in self.bullet_text.items():
                if updated.bullet_text.get(bullet_id) != text:
                    updated.bullet_text[str(bullet_id)] = str(text)
                    changes.append(f"~bullet_text: {bullet_id}")
        if self.requirement_reweight is not None:
            for requirement_id, weight in self.requirement_reweight.items():
                requirement_id = str(requirement_id).strip()
                if not requirement_id:
                    continue
                weight = float(weight)
                if updated.requirements.reweight.get(requirement_id) != weight:
                    updated.requirements.reweight[requirement_id] = weight
                    changes.append(f"~weight: {requirement_id} = {weight:g}")
        if self.summary is not None and updated.summary != self.summary:
            updated.summary = self.summary
            changes.append("~summary")
        if self.budget_lines is not None and updated.budget_lines != self.budget_lines:
            updated.budget_lines = int(self.budget_lines)
            changes.append(f"~budget_lines: {self.budget_lines}")
        return updated, changes


class OverridesInfo(_Model):
    """A run's ``overrides.yaml`` after an edit that did not re-run anything.

    The TUI pins and excludes one item at a time and re-runs once, so writing
    the file has to be separable from acting on it. ``changes`` is what this
    edit did; ``summary`` is what the file now says altogether.
    """

    run_id: str
    path: str
    changes: list[str] = Field(default_factory=list)
    summary: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict)


class ItemUsage(_Model):
    """What one role or project did in one recorded run.

    ``rank`` is the position of its first selected bullet in the greedy trace —
    1 means the solver reached for this item before anything else. ``None``
    means the run did not select it at all, which is the answer an editor
    actually wants: an item nothing ever picks is an item to rewrite.
    """

    run_id: str
    created_at: str
    company: str | None = None
    role: str | None = None
    status: str = ""
    selected: bool = False
    rank: int | None = None
    bullets: list[str] = Field(default_factory=list)
    gain: float | None = None


class ProposalInfo(_Model):
    """One proposed rephrasing, with the verdict that decides its fate.

    The text lives here and in the store, never in the index, until
    :meth:`Service.accept_proposal` copies it into the bullet's YAML. Reading a
    proposal is therefore always safe: nothing a model or an agent produced can
    reach a PDF by being read.
    """

    id: str
    bullet_id: str
    archetype: str
    text: str
    status: str = "pending"
    provenance: str | None = None
    created_at: str = ""
    decided_at: str | None = None
    #: Why it was rejected, in the deciding human's words.
    note: str | None = None
    #: The wording the proposal would replace, and where it lives.
    source: str = ""
    item_id: str = ""
    kind: str = "bullet"
    guardrail: GuardrailReport | None = None
    #: Set by :meth:`Service.accept_proposal`: the index file it wrote into.
    path: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.guardrail and self.guardrail.ok)

    def summary_line(self) -> str:
        verdict = self.guardrail.summary() if self.guardrail else "unchecked"
        return f"{self.id}  [{self.status}]  guardrail {verdict}"


class EvalBaseline(_Model):
    """One ``eval/<config_hash>.json`` on disk, without loading its runs."""

    path: str
    config_hash: str = ""
    created_at: str = ""
    engine: str = ""
    provider: str = ""
    runs: int = 0
    means: dict[str, Any] = Field(default_factory=dict)


# ── Sanitising a pasted posting ──────────────────────────────────────────────
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</\s*script\s*>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</\s*style\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"</?[A-Za-z!?][^>]*>")
_ENTITY_RE = re.compile(r"&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]{1,31});")
#: Zero-width and other characters that occupy no space on screen: joiners,
#: bidi marks, word joiners, the BOM, the soft hyphen, the Mongolian vowel
#: separator, and the Unicode tag block — the one a prompt injection hides
#: in, because no reader and no terminal will show it.
_INVISIBLE_RE = re.compile(
    "[\u200b-\u200f\u2060-\u2064\ufeff\u00ad\u180e]|[\U000e0000-\U000e007f]"
)
#: Whitespace that is neither a newline nor a tab: non-breaking spaces, thin
#: spaces, and the rest of a copy-paste from a rendered page.
_ODD_SPACE_RE = re.compile(r"[^\S\n\t]")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_HORIZONTAL_RUN_RE = re.compile(r"[ \t]{2,}")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def sanitise_jd(text: str) -> tuple[str, Sanitised]:
    """Strip markup and invisible characters out of a pasted posting.

    Order matters: ``<script>``/``<style>`` bodies go first (they are not
    prose and dropping only their tags would leave the code behind), then
    comments — where a posting that was scraped keeps its "instructions to the
    reader" — then the remaining tags, then the characters nobody can see.
    """
    original = str(text or "")
    counts = Sanitised()

    body, counts.script_blocks = _SCRIPT_RE.subn(" ", original)
    body, counts.style_blocks = _STYLE_RE.subn(" ", body)
    body, counts.html_comments = _COMMENT_RE.subn(" ", body)
    body, counts.html_tags = _TAG_RE.subn(" ", body)
    counts.entities = len(_ENTITY_RE.findall(body))
    if counts.entities:
        body = html.unescape(body)
    body, counts.invisible_chars = _INVISIBLE_RE.subn("", body)
    body = _ODD_SPACE_RE.sub(" ", body)
    # Whatever control characters a copy-paste from a PDF or a terminal dragged
    # along: they are not text, and one of them is an escape sequence.
    stripped = "".join(
        ch for ch in body
        if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )
    counts.invisible_chars += len(body) - len(stripped)
    body = stripped

    body = _HORIZONTAL_RUN_RE.sub(" ", body)
    body = _TRAILING_WS_RE.sub("", body)
    body, counts.blank_lines = _BLANK_RUN_RE.subn("\n\n", body)
    body = body.strip() + "\n"

    counts.chars_removed = max(0, len(original) - len(body))
    return body, counts


# ── Small helpers ────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_slug(*parts: Any) -> str:
    """``company``/``role`` as one path-safe component. Mirrors the CLI's."""
    raw = "-".join(str(p).strip().lower() for p in parts if str(p).strip())
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw)
    slug = re.sub(r"-+", "-", slug).strip(".-_")
    return slug or "resume"


_METRIC_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|[kKmMbB]\b|x\b|ms\b|s\b|GB\b|MB\b|TB\b|"
    r"fps\b|rps\b|qps\b|hrs?\b|mins?\b)?"
)


def auto_metrics(text: str) -> list[str]:
    """Every number a bullet claims, so a later rephrase cannot quietly drop one.

    Deliberately greedy: ``metrics`` exists for
    :func:`ats_tailor.runs.overrides.metric_warnings`, which asks "does the edited
    wording still carry this?". A number the bullet mentions in passing costs
    one harmless warning; a number that was never recorded costs the check.
    """
    found: list[str] = []
    for match in _METRIC_RE.finditer(str(text or "")):
        value = re.sub(r"\s+", " ", match.group()).strip()
        if value and value not in found:
            found.append(value)
    return found


def _validate_id(value: Any, what: str = "id") -> str:
    identifier = str(value or "").strip()
    if not identifier or not ID_RE.match(identifier):
        raise ServiceError(
            "invalid_input",
            f"{what} must match {ID_RE.pattern} (got {value!r})",
        )
    return identifier


def _clean_archetype(value: Any) -> str:
    """An archetype as it will be written: squashed, bounded, non-empty.

    Not an id: it is a skill-category name ("DevOps & Tooling"), and the
    selection matches it case-insensitively, so the spelling that reaches the
    YAML is the caller's.
    """
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        raise ServiceError("invalid_input", "an archetype is required")
    if len(text) > MAX_ARCHETYPE_CHARS:
        raise ServiceError(
            "invalid_input",
            f"archetype is {len(text)} characters (at most {MAX_ARCHETYPE_CHARS})")
    return text


def _errors_from(exc: ValidationError) -> list[str]:
    return [
        (".".join(str(part) for part in error["loc"]) + ": " if error["loc"] else "")
        + error["msg"].removeprefix("Value error, ")
        for error in exc.errors()
    ]


def _plain(value: Any) -> Any:
    """Strip ruamel wrappers and pydantic models down to JSON-shaped data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _merge(target: Any, patch: Any) -> Any:
    """Recursive dict merge; lists and scalars in ``patch`` replace."""
    if isinstance(target, Mapping) and isinstance(patch, Mapping):
        merged = dict(target)
        for key, value in patch.items():
            merged[key] = _merge(merged.get(key), value) if key in merged else value
        return merged
    return patch


# ── The service ──────────────────────────────────────────────────────────────
class Service:
    """Everything a front-end can do, in one object.

    Directories default to the conventional layout under ``repo_root`` and are
    each overridable, so a test can point one at a temporary copy of
    ``examples/`` without moving anything else. The index and the store load on
    first use: constructing a ``Service`` opens no files and imports no model.
    """

    def __init__(
        self,
        repo_root: str | Path,
        *,
        index_dir: str | Path | None = None,
        jd_dir: str | Path | None = None,
        labels_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        eval_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        profile_path: str | Path | None = None,
        provider: str | None = None,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.index_dir = self._resolve(index_dir, "index")
        self.jd_dir = self._resolve(jd_dir, "jd")
        self.labels_dir = self._resolve(labels_dir, "labels")
        self.output_dir = self._resolve(output_dir, "output")
        self.eval_dir = self._resolve(eval_dir, "eval")
        self.data_dir = self._resolve(data_dir, "data")
        self.profile_path = (
            Path(profile_path).expanduser().resolve() if profile_path
            else self.repo_root / "profile.yaml"
        )
        #: Who reads a posting when a call does not say. ``none`` by default:
        #: the deterministic parser needs nothing running, and the MCP's whole
        #: premise is that the *driving agent* is the model (``external``).
        self.provider_name = str(provider or "none").strip().lower() or "none"
        self._index: Index | None = None
        self._store: Store | None = None
        self._profile: dict | None = None

    def _resolve(self, given: str | Path | None, name: str) -> Path:
        return (
            Path(given).expanduser().resolve() if given else self.repo_root / name
        )

    # ── lifecycle ──
    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def __enter__(self) -> "Service":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def reload(self) -> None:
        """Forget the loaded index and profile; the next call reads them again."""
        self._index = None
        self._profile = None

    @property
    def index(self) -> Index:
        if self._index is None:
            if not self.index_dir.is_dir():
                raise ServiceError("not_found", f"no index directory at {self.rel(self.index_dir)}")
            if detect_layout(self.index_dir) != "v2":
                raise ServiceError(
                    "invalid_input",
                    f"{self.rel(self.index_dir)} is a v1 index layout; "
                    f"run `ats-tailor migrate` first",
                )
            try:
                self._index = load_index_v2(self.index_dir)
            except ValueError as exc:
                raise ServiceError("validation_failed", str(exc)) from exc
        return self._index

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store(self.data_dir / "ats.db")
        return self._store

    @property
    def profile(self) -> dict:
        if self._profile is None:
            from .index.loaders import load_profile

            try:
                self._profile = load_profile(self.profile_path)
            except ValueError as exc:
                raise ServiceError("not_found", str(exc)) from exc
        return self._profile

    def rel(self, path: str | Path) -> str:
        """``path`` relative to the repo root, or absolute when it is outside."""
        resolved = self.locate(path)
        try:
            return str(resolved.resolve().relative_to(self.repo_root))
        except ValueError:
            return str(resolved)

    def locate(self, path: str | Path) -> Path:
        """Where a recorded relative path actually points.

        A run record keeps the output directory it was *given*, and the CLI
        builds that from the working directory — so an older record can say
        ``../output/cpf-sre-intern`` and mean something only the shell that
        started it knew. The repo root, the current directory and the index
        directory are tried in that order and the first that exists wins;
        when none does, the repo-relative reading is what an error message
        will name, which is the reading a user can act on.
        """
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        for base in (self.repo_root, Path.cwd(), self.index_dir):
            resolved = (base / candidate).resolve()
            if resolved.exists():
                return resolved
        return (self.repo_root / candidate).resolve()

    def run_dir(self, run_id: str) -> Path:
        """The directory a run wrote into, located on disk."""
        return self._out_dir_of(self._run_row(run_id))

    def healthcheck(self) -> dict[str, Any]:
        """Whether this repo is usable, and what it holds. Never raises."""
        payload: dict[str, Any] = {
            "repo_root": str(self.repo_root),
            "provider": self.provider_name,
            "engine": "v2",
            "directories": {
                name: {"path": self.rel(path), "exists": path.exists()}
                for name, path in (
                    ("index", self.index_dir), ("jd", self.jd_dir),
                    ("labels", self.labels_dir), ("output", self.output_dir),
                    ("eval", self.eval_dir), ("data", self.data_dir),
                )
            },
            "tools": {
                name: _which(name)
                for name in ("latexmk", "tectonic", "pdftotext", "pdfinfo")
            },
            "latex_engine": _latex_engine(),
        }
        try:
            index = self.index
            payload["index"] = {
                "roles": len(index.roles),
                "projects": len(index.projects),
                "skills": len(index.all_skills()),
                "certifications": len(index.certifications),
                "ok": True,
            }
        except ServiceError as exc:
            payload["index"] = {"ok": False, "error": exc.to_dict()}
        try:
            payload["store"] = {
                "path": self.rel(self.store.path),
                "entities": len(self.store.entities()),
                "runs": len(self.store.list_runs()),
                "ok": True,
            }
        except Exception as exc:  # a broken store must not break the probe
            payload["store"] = {"ok": False, "error": str(exc)}
        payload["jds"] = len(list(self.jd_dir.glob("*.txt"))) if self.jd_dir.is_dir() else 0
        payload["ok"] = bool(payload["index"].get("ok"))
        return payload

    # ── JD metadata sidecar ──
    @property
    def _meta_path(self) -> Path:
        return self.data_dir / JD_META_FILENAME

    def _meta(self) -> dict[str, dict[str, Any]]:
        path = self._meta_path
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_meta(self, slug: str, entry: Mapping[str, Any]) -> None:
        data = self._meta()
        data[slug] = {**data.get(slug, {}), **dict(entry)}
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    # ── Postings ─────────────────────────────────────────────────────────────
    def jd_path(self, slug: str) -> Path:
        """The file for ``slug``, checked to be a posting inside ``jd/``."""
        slug = _validate_id(slug, "jd slug")
        path = (self.jd_dir / f"{slug}.txt").resolve()
        if self.jd_dir.resolve() not in path.parents:
            raise ServiceError("invalid_input", f"jd slug escapes {self.rel(self.jd_dir)}")
        if not path.is_file():
            raise ServiceError("not_found", f"no job description at {self.rel(path)}")
        return path

    def jd_text(self, slug: str) -> str:
        return self.jd_path(slug).read_text(errors="replace")

    def list_jds(self) -> list[JDInfo]:
        """Every posting in ``jd/``, in slug order.

        Alphabetical rather than ranked: a listing an agent pages through has
        to be stable between calls, and nothing here knows which posting
        matters most.
        """
        if not self.jd_dir.is_dir():
            return []
        meta = self._meta()
        labels = {p.stem for p in self.labels_dir.glob("*.yaml")} if self.labels_dir.is_dir() else set()
        runs_by_hash: dict[str, list[Run]] = {}
        try:
            for run in self.store.list_runs():
                if run.jd_hash:
                    runs_by_hash.setdefault(run.jd_hash, []).append(run)
        except Exception:
            runs_by_hash = {}

        listing: list[JDInfo] = []
        for path in sorted(self.jd_dir.glob("*.txt")):
            slug = path.stem
            text = path.read_text(errors="replace")
            digest = jd_hash(text)
            entry = meta.get(slug, {}) if isinstance(meta.get(slug), dict) else {}
            runs = runs_by_hash.get(digest, [])
            company = entry.get("company") or (runs[0].company if runs else None)
            role = entry.get("role") or (runs[0].role if runs else None)
            if company is None or role is None:
                labelled = self._label_of(slug)
                company = company or (labelled.get("company") if labelled else None)
                role = role or (labelled.get("role") if labelled else None)
            stored = self._stored_profile(digest)
            gates_summary = deadline = None
            if stored is not None:
                gates_summary = stored.gates.summary_line()
                deadline = stored.gates.deadline.date
            listing.append(JDInfo(
                slug=slug,
                path=self.rel(path),
                company=company,
                role=role,
                jd_hash=digest,
                chars=len(text),
                has_profile=stored is not None,
                gates_summary=gates_summary,
                deadline=deadline,
                label_exists=slug in labels,
                source_url=entry.get("source_url"),
                runs=len(runs),
            ))
        return listing

    def _label_of(self, slug: str) -> dict | None:
        path = self.labels_dir / f"{slug}.yaml"
        if not path.is_file():
            return None
        try:
            import yaml

            data = yaml.safe_load(path.read_text())
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _stored_profile(self, digest: str) -> JDProfile | None:
        """Any profile already stored for this posting, edited copy preferred."""
        try:
            rows = self.store.jd_profiles(digest)
        except Exception:
            return None
        for row in rows:
            profile = JDProfile.from_json(row.edited)
            if profile is not None:
                return profile.model_copy(update={"provider": row.provider, "cached": True})
        for row in rows:
            profile = JDProfile.from_json(row.extracted)
            if profile is not None:
                return profile.model_copy(update={"provider": row.provider, "cached": True})
        return None

    def add_jd(
        self,
        text: str,
        company: str,
        role: str,
        source_url: str | None = None,
    ) -> JDInfo:
        """Store a pasted posting as ``jd/<slug>.txt``, sanitised.

        The URL is recorded as text and never fetched: retrieving a page is
        somebody else's tool, and a service that follows a link found in its
        own input has already lost the argument about what is data.
        """
        if not str(text or "").strip():
            raise ServiceError("invalid_input", "the posting is empty")
        if not str(company or "").strip() or not str(role or "").strip():
            raise ServiceError("invalid_input", "company and role are required")

        body, counts = sanitise_jd(text)
        if not body.strip():
            raise ServiceError(
                "invalid_input", "nothing but markup was left after sanitising")

        base = safe_slug(company, role)
        _validate_id(base, "derived slug")
        self.jd_dir.mkdir(parents=True, exist_ok=True)
        slug, suffix = base, 1
        while (self.jd_dir / f"{slug}.txt").exists():
            suffix += 1
            slug = f"{base}-{suffix}"
        path = self.jd_dir / f"{slug}.txt"
        path.write_text(body)

        self._write_meta(slug, {
            "company": str(company).strip(),
            "role": str(role).strip(),
            "source_url": str(source_url).strip() if source_url else None,
            "added_at": _now(),
            "sanitised": counts.model_dump(),
        })
        return JDInfo(
            slug=slug,
            path=self.rel(path),
            company=str(company).strip(),
            role=str(role).strip(),
            jd_hash=jd_hash(body),
            chars=len(body),
            has_profile=False,
            label_exists=(self.labels_dir / f"{slug}.yaml").is_file(),
            source_url=str(source_url).strip() if source_url else None,
            sanitised=counts,
        )

    # ── Requirements ─────────────────────────────────────────────────────────
    def _provider(self, name: str | None = None, profile: JDProfile | None = None):
        from .jd.extract import index_vocabulary, term_alias_map
        from .llm.provider import get_provider

        requested = str(name or self.provider_name or "none").strip().lower()
        if requested not in set(config.PROVIDERS):
            raise ServiceError(
                "invalid_input",
                f"unknown provider {requested!r}; expected one of: "
                f"{', '.join(config.PROVIDERS)}",
            )
        try:
            return get_provider(requested, alias_map=term_alias_map(self.index),
                                profile=profile, quiet=True,
                                vocabulary=index_vocabulary(self.index))
        except ValueError as exc:
            raise ServiceError("provider_unavailable", str(exc)) from exc

    def jd_profile(
        self, slug: str, provider: str | None = None, reextract: bool = False
    ) -> JDProfile:
        """The posting read into gates and weighted requirements."""
        from .jd.extract import extract_requirements

        path = self.jd_path(slug)
        text = path.read_text(errors="replace")
        entry = self._meta().get(slug) or {}
        known = [v for v in (entry.get("company"), entry.get("role")) if v]
        return extract_requirements(
            text, self.index, self._provider(provider), self.store,
            jd_path=str(path), reextract=bool(reextract),
            extra_known=known,
            languages=self._languages(),
            say=lambda *_a, **_k: None,
        )

    def _languages(self) -> list[str]:
        try:
            return list(self.profile.get("languages", []) or [])
        except ServiceError:
            return []

    def set_requirements(
        self,
        slug: str,
        profile: JDProfile | Mapping[str, Any],
        provenance: str = "agent",
    ) -> JDProfile:
        """Store a hand- or agent-written profile where it outranks extraction.

        It is written under the deterministic row *and* under the service's own
        provider when that is a different reader. An edit is a statement about
        the posting, not about which model would otherwise have read it, so it
        should not evaporate because the next run was started with a different
        ``--provider``.
        """
        from .jd.extract import FALLBACK_PROVIDER, save_edited

        path = self.jd_path(slug)
        text = path.read_text(errors="replace")
        if isinstance(profile, JDProfile):
            validated = profile
        else:
            try:
                validated = JDProfile.model_validate(_plain(profile))
            except ValidationError as exc:
                raise ServiceError(
                    "validation_failed",
                    "the requirement profile does not validate",
                    _errors_from(exc),
                ) from exc
        if not validated.requirements:
            raise ServiceError("invalid_input", "a profile needs at least one requirement")

        rows = {FALLBACK_PROVIDER}
        if self.provider_name not in {"none", FALLBACK_PROVIDER}:
            rows.add(self.provider_name)
        for row_provider in sorted(rows):
            save_edited(
                self.store, text,
                validated.model_copy(update={"provider": row_provider}),
                prompt_version="" if row_provider == FALLBACK_PROVIDER else config.PROMPT_VERSION,
                jd_path=str(path),
            )
        self._write_meta(slug, {
            "requirements_provenance": {"by": str(provenance), "at": _now()},
        })
        return validated.model_copy(update={"provider": FALLBACK_PROVIDER, "cached": True})

    # ── Fit ──────────────────────────────────────────────────────────────────
    def fit(self, slug: str) -> FitReport:
        """Could this index cover this posting at all — before any tailoring.

        Requirements and claims are built exactly as a run builds them and
        scored with the same matrix, then every requirement is credited with
        its single best claim. Nothing is selected, nothing is rendered, and
        the only thing written is the posting's cached profile.
        """
        from ats_match import build_matrix

        from .engine.adapt import alias_map, claims_from_index, header_claims, skill_claims
        from .ats.check import gap_terms
        from .engine.embedding import SentenceTransformerEmbedder, encode_store

        path = self.jd_path(slug)
        text = path.read_text(errors="replace")
        index = self.index
        profile = self.jd_profile(slug)
        requirements = profile.to_requirements()
        entry = self._meta().get(slug) or {}
        gaps = gap_terms(text, index, extra_known=[
            v for v in (entry.get("company"), entry.get("role")) if v])

        claims = claims_from_index(index)
        if config.HEADER_CLAIMS:
            claims += header_claims(index)
        claims += skill_claims(index, requirements)
        if not requirements or not claims:
            return FitReport(
                slug=slug, title=profile.title, archetype=profile.archetype,
                gates=_gates_json(profile), gates_summary=profile.gates.summary_line(),
                blocking_gates=[name for name, _g in profile.gates.blocking()],
                requirements=len(requirements), gaps=list(gaps),
            )

        embedder = SentenceTransformerEmbedder(config.EMBED_MODEL)
        claim_vectors = encode_store(
            embedder, [c.text for c in claims], self.store,
            ["skill" if c.section == "skill" else "claim" for c in claims],
            embedder.passage_prefix,
        )
        requirement_vectors = encode_store(
            embedder, [r.text for r in requirements], self.store, "req",
            embedder.query_prefix,
        )
        matrix, _sim = build_matrix(
            requirements, claims, embedder, alias_map(index), reranker=None,
            claim_vectors=claim_vectors, requirement_vectors=requirement_vectors,
        )

        best = matrix.max(axis=1) if matrix.size else []
        # A requirement that names nothing ("works well in a team") is answered
        # about equally well by every claim, so its ceiling measures the encoder
        # and not the index. It is excluded from the bounds and still reported.
        hard = [i for i, r in enumerate(requirements) if r.terms]
        must = [i for i in hard if requirements[i].kind == "must"]
        nice = [i for i in hard if requirements[i].kind == "nice"]

        contribution: dict[int, float] = {}
        for r in must or hard or range(len(requirements)):
            for c in range(len(claims)):
                contribution[c] = contribution.get(c, 0.0) + float(matrix[r, c])
        ranked = sorted(contribution.items(), key=lambda pair: -pair[1])
        strengths: list[str] = []
        for claim_index, score in ranked:
            claim = claims[claim_index]
            if claim.section == "skill":
                continue
            strengths.append(f"{claim.group}: {claim.id} ({score:.2f})")
            if len(strengths) >= FIT_STRENGTHS:
                break

        claim_terms = {
            str(term).strip().lower()
            for claim in claims for term in claim.terms if str(term).strip()
        }
        missing_terms: list[str] = []
        for requirement in requirements:
            for term in requirement.terms:
                key = str(term).strip().lower()
                if key and key not in claim_terms and term not in missing_terms:
                    missing_terms.append(str(term))

        uncovered = [
            requirements[i].text
            for i in sorted(must, key=lambda i: float(best[i]))[:5]
            if float(best[i]) < 0.3
        ]
        return FitReport(
            slug=slug,
            title=profile.title,
            archetype=profile.archetype,
            gates=_gates_json(profile),
            gates_summary=profile.gates.summary_line(),
            blocking_gates=[name for name, _g in profile.gates.blocking()],
            requirements=len(requirements),
            must_requirements=len(must),
            nice_requirements=len(nice),
            must_upper_bound=_mean(best, must),
            nice_upper_bound=_mean(best, nice),
            strengths=strengths,
            gaps=list(missing_terms),
            uncovered=uncovered,
        )

    # ── Runs ─────────────────────────────────────────────────────────────────
    def tailor(
        self,
        slug: str,
        company: str | None = None,
        role: str | None = None,
        *,
        out_dir: str | Path | None = None,
        overrides: OverridesPatch | Overrides | None = None,
        check: bool = True,
        provider: str | None = None,
        engine: str = "v2",
        record: bool = True,
    ) -> RunSummary:
        """Tailor the posting ``slug`` and write the run's artefacts."""
        if str(engine) != "v2":
            raise ServiceError(
                "invalid_input",
                f"the service runs the v2 engine only (got {engine!r}); "
                f"use `ats-tailor tailor --engine v1` for the legacy path",
            )
        path = self.jd_path(slug)
        text = path.read_text(errors="replace")
        entry = self._meta().get(slug) or {}
        labelled = self._label_of(slug) or {}
        company = str(company or entry.get("company") or labelled.get("company") or "").strip()
        role = str(role or entry.get("role") or labelled.get("role") or "").strip()
        if not company or not role:
            raise ServiceError(
                "invalid_input",
                f"company and role are required for {slug} "
                f"(nothing recorded for this posting)",
            )

        target = Path(out_dir).expanduser().resolve() if out_dir else (
            self.output_dir / safe_slug(company, role))
        applied, changes = self._overrides_for(target, overrides)
        return self._run(
            slug, text, str(path), company, role, target, applied,
            check=check, provider=provider, record=record, extra_warnings=changes,
        )

    def _overrides_for(
        self, out_dir: Path, patch: OverridesPatch | Overrides | None
    ) -> tuple[Overrides, list[str]]:
        """The overrides a run should use: the file on disk, plus any patch."""
        from .runs import overrides as overrides_mod

        path = overrides_path(out_dir)
        current = overrides_mod.load(path) if path.is_file() else Overrides()
        if patch is None:
            return current, []
        if isinstance(patch, Overrides):
            return patch, []
        updated, changes = patch.apply_to(current)
        if changes:
            overrides_mod.save(path, updated)
        return updated, [f"overrides {change}" for change in changes]

    def _run(
        self,
        slug: str | None,
        text: str,
        jd_path: str | None,
        company: str,
        role: str,
        out_dir: Path,
        applied: Overrides,
        *,
        check: bool,
        provider: str | None,
        record: bool,
        extra_warnings: Sequence[str] = (),
        supersedes: str | None = None,
    ) -> RunSummary:
        from .engine.pipeline import V2Options, tailor_v2
        from .runs.tracker import record as record_run

        warnings: list[str] = list(extra_warnings)
        out_dir.mkdir(parents=True, exist_ok=True)
        options = V2Options(
            index_dir=self.index_dir,
            overrides=applied,
            store=self.store,
            jd_path=jd_path,
            provider=self._provider(provider),
            languages=self._languages(),
            quiet=True,
        )
        try:
            result = tailor_v2(text, self.index, self.profile, company, role,
                               out_dir, options)
        except (ValueError, OSError) as exc:
            raise ServiceError("conflict", f"the run failed: {exc}") from exc
        for name, gate in gate_flags(result.gates):
            warnings.append(f"gate {name}: {gate}")

        check_payload = None
        pdf_path = None
        if check:
            try:
                report = self._check_out_dir(out_dir, text, company, result.entries)
                check_payload = {"parse": report.parse, "coverage": report.coverage}
                pdf_path = out_dir / _pdf_name(self.profile, company)
            except ServiceError as exc:
                warnings.append(f"check skipped: {exc.message}")

        run_id = None
        if record:
            run = record_run(
                self.store, result,
                jd_hash=jd_hash(text),
                jd_path=jd_path,
                config_hash=self._config_hash(),
                overrides=applied.model_dump(mode="json") if not applied.empty() else None,
                check=check_payload,
                pdf_path=pdf_path,
            )
            run_id = run.id
            if supersedes:
                self.store.add_note(run.id, f"re-run of {supersedes}")
        return RunSummary(
            run_id=run_id,
            out_dir=self.rel(out_dir),
            company=company,
            role=role,
            engine=result.engine,
            tex=self.rel(result.tex_path),
            pages=result.pages,
            budget=result.budget,
            iterations=result.iterations,
            jd_slug=slug,
            jd_hash=jd_hash(text),
            title=result.jd_title,
            archetype=result.archetype,
            provider=result.provider,
            coverage=dict(result.coverage),
            selection=dict(result.selection or {}),
            gates=dict(result.gates or {}),
            check=check_payload,
            warnings=warnings,
        )

    def _config_hash(self) -> str | None:
        from .evaluation.evaluate import config_fingerprint

        try:
            return config_fingerprint(self.index_dir, "v2")[0]
        except (OSError, ValueError):
            return None

    def get_run(self, run_id: str) -> RunRecord:
        run = self._run_row(run_id)
        notes = [RunNote(created_at=n.created_at, note=n.note)
                 for n in self.store.notes(run.id)]
        return RunRecord(
            id=run.id, created_at=run.created_at, company=run.company, role=run.role,
            jd_hash=run.jd_hash, engine=run.engine, config_hash=run.config_hash,
            out_dir=self.rel(run.out_dir) if run.out_dir else None,
            status=run.status, final=run.final, pdf_sha256=run.pdf_sha256,
            data=dict(run.data), notes=notes,
        )

    def _run_row(self, reference: str) -> Run:
        """A run by id, or the newest run recorded for an output directory."""
        identifier = str(reference or "").strip()
        if not identifier:
            raise ServiceError("invalid_input", "a run id is required")
        if ID_RE.match(identifier):
            run = self.store.get_run(identifier)
            if run is not None:
                return run
        path = self.locate(identifier)
        if path.exists():
            wanted = path.resolve()
            for candidate in self.store.list_runs():
                if candidate.out_dir and Path(candidate.out_dir).resolve() == wanted:
                    return candidate
        raise ServiceError("not_found", f"no recorded run for {reference!r}")

    def override(
        self, run_id_or_out_dir: str, patch: OverridesPatch
    ) -> RunSummary:
        """Edit a run's ``overrides.yaml`` and re-run it into the same directory."""
        if patch is None or patch.empty():
            raise ServiceError("invalid_input", "the override patch says nothing")
        return self._rerun(run_id_or_out_dir, patch)

    def rerun(self, run_id_or_out_dir: str) -> RunSummary:
        """Run again into the same directory, honouring ``overrides.yaml`` as it is.

        The half of :meth:`override` that a front-end which edits the file one
        pin at a time needs: by the time the user presses "re-run" the patch has
        already been written, and a method that insists on being handed one
        again would either re-apply it or refuse.
        """
        return self._rerun(run_id_or_out_dir, None)

    def set_overrides(
        self, run_id_or_out_dir: str, patch: OverridesPatch
    ) -> OverridesInfo:
        """Write a patch into a run's ``overrides.yaml`` without re-running it.

        Pinning an item is cheap; acting on the pin is a minute of solver and
        LaTeX. Separating them is what lets a person pin four things and pay
        once — and it means the file on disk is always what the screen shows,
        even if the re-run never happens.
        """
        if patch is None or patch.empty():
            raise ServiceError("invalid_input", "the override patch says nothing")
        run = self._run_row(run_id_or_out_dir)
        out_dir = self._out_dir_of(run)
        applied, changes = self._overrides_for(out_dir, patch)
        return OverridesInfo(
            run_id=run.id,
            path=self.rel(overrides_path(out_dir)),
            changes=changes,
            summary=applied.summary_line(),
            overrides=applied.model_dump(mode="json"),
        )

    def get_overrides(self, run_id_or_out_dir: str) -> OverridesInfo:
        """A run's ``overrides.yaml`` as it stands, without changing anything."""
        run = self._run_row(run_id_or_out_dir)
        out_dir = self._out_dir_of(run)
        applied, _changes = self._overrides_for(out_dir, None)
        return OverridesInfo(
            run_id=run.id,
            path=self.rel(overrides_path(out_dir)),
            summary=applied.summary_line(),
            overrides=applied.model_dump(mode="json"),
        )

    def _rerun(self, reference: str, patch: OverridesPatch | None) -> RunSummary:
        run = self._run_row(reference)
        if not run.out_dir:
            raise ServiceError("conflict", f"{run.id} has no recorded output directory")
        out_dir = self.locate(run.out_dir)
        text, jd_path, slug = self._posting_of(run)
        applied, changes = self._overrides_for(out_dir, patch)
        if patch is not None and not changes:
            changes = ["overrides unchanged (the patch was already applied)"]
        return self._run(
            slug, text, jd_path, run.company, run.role, out_dir, applied,
            check=bool((run.data or {}).get("check")), provider=None, record=True,
            extra_warnings=changes, supersedes=run.id,
        )

    def _posting_of(self, run: Run) -> tuple[str, str | None, str | None]:
        """``(text, jd_path, slug)`` for a recorded run.

        The store is asked first: it holds the posting verbatim as it was when
        the run happened, and the file it came from may since have been edited,
        moved, or never have existed (a pasted posting).
        """
        recorded = (run.data or {}).get("jd_path")
        if run.jd_hash:
            row = self.store.get_jd_profile(run.jd_hash)
            if row is not None and row.text:
                slug = Path(row.source_path or recorded or "").stem or None
                return row.text, row.source_path or recorded, slug
        if recorded:
            path = self.locate(recorded)
            if path.is_file():
                return path.read_text(errors="replace"), str(path), path.stem
        raise ServiceError(
            "not_found",
            f"{run.id}: the posting it was tailored to is not on disk or in the store",
        )

    def compile(self, run_id: str) -> CompileSummary:
        """Compile a run's ``resume.tex`` and record the hash of what came out."""
        from .ats.check import CompileError, compile_pdf
        from .runs.tracker import file_sha256

        run = self._run_row(run_id)
        out_dir = self._out_dir_of(run)
        tex_path = out_dir / "resume.tex"
        if not tex_path.is_file():
            raise ServiceError("not_found", f"no resume.tex in {self.rel(out_dir)}")
        engine = _latex_engine()
        if not _which(engine):
            raise ServiceError("compile_failed", f"{engine} is not on PATH")
        try:
            compiled = compile_pdf(tex_path, _pdf_name(self.profile, run.company or ""))
        except (CompileError, OSError) as exc:
            raise ServiceError("compile_failed", str(exc)) from exc
        pdf_path = out_dir / _pdf_name(self.profile, run.company or "")
        if not pdf_path.is_file():
            pdf_path = compiled.pdf_path
        digest = file_sha256(pdf_path)
        self.store.record_run(
            run_id=run.id, company=run.company, role=run.role, data=run.data,
            jd_hash=run.jd_hash, engine=run.engine or "v2",
            config_hash=run.config_hash, out_dir=run.out_dir, status=run.status,
            pdf_sha256=digest, created_at=run.created_at,
        )
        return CompileSummary(
            pdf_path=self.rel(pdf_path), pages=compiled.pages, sha256=digest)

    def _out_dir_of(self, run: Run) -> Path:
        if not run.out_dir:
            raise ServiceError("conflict", f"{run.id} has no recorded output directory")
        return self.locate(run.out_dir)

    def run_pdf(self, run_id: str) -> str:
        """The compiled PDF for a run, repo-relative. Never compiles anything.

        The named copy (``First_Last_Resume_Company.pdf``) is preferred over
        ``resume.pdf``: it is the file that gets uploaded, so it is the one a
        person means when they say "open it".
        """
        run = self._run_row(run_id)
        out_dir = self._out_dir_of(run)
        for name in (_pdf_name(self.profile, run.company or ""), "resume.pdf"):
            candidate = out_dir / name
            if candidate.is_file():
                return self.rel(candidate)
        raise ServiceError(
            "not_found",
            f"{run.id} has no compiled PDF in {self.rel(out_dir)}; compile it first",
        )

    def run_text(self, run_id: str) -> str:
        """What an ATS reads: ``pdftotext -layout`` over the run's PDF.

        The preview a person judges the resume by has to be the extracted text
        and not the LaTeX source, because every failure this project exists to
        catch happens between the two.
        """
        from .ats.check import extract_text

        pdf_path = self.locate(self.run_pdf(run_id))
        if not _which("pdftotext"):
            raise ServiceError("compile_failed", "pdftotext (poppler) is not on PATH")
        try:
            return extract_text(pdf_path)
        except (RuntimeError, OSError) as exc:
            raise ServiceError("compile_failed", str(exc)) from exc

    def check(self, run_id: str, labels: str | None = None) -> CheckReport:
        """Compile (if needed) and run the ATS parse + keyword checks on a run."""
        run = self._run_row(run_id)
        out_dir = self._out_dir_of(run)
        text, _path, slug = self._posting_of(run)
        report = self._check_out_dir(
            out_dir, text, run.company or "", (run.data or {}).get("entries") or [],
            labels=labels if labels is not None else slug,
        )
        report.run_id = run.id
        data = dict(run.data)
        data["check"] = {"parse": report.parse, "coverage": report.coverage}
        self.store.record_run(
            run_id=run.id, company=run.company, role=run.role, data=data,
            jd_hash=run.jd_hash, engine=run.engine or "v2",
            config_hash=run.config_hash, out_dir=run.out_dir, status=run.status,
            pdf_sha256=run.pdf_sha256, created_at=run.created_at,
        )
        return report

    def _check_out_dir(
        self,
        out_dir: Path,
        jd_text: str,
        company: str,
        entries: Sequence[Mapping[str, Any]],
        labels: str | None = None,
    ) -> CheckReport:
        """The parse + keyword check, without printing a word.

        A silent twin of ``tailor._run_ats_check``: same functions, same
        reports on disk, no stdout — an MCP server speaks JSON-RPC over that
        stream and one stray line of progress ends the session.
        """
        from .ats.check import (
            CompileError, compile_pdf, extract_text, index_term_forms,
            keyword_coverage, page_count, parse_check, resolve_terms, split_regions,
            summarize, surface_form_set, write_reports,
        )

        tex_path = out_dir / "resume.tex"
        if not tex_path.is_file():
            raise ServiceError("not_found", f"no resume.tex in {self.rel(out_dir)}")
        engine = _latex_engine()
        if not _which(engine):
            raise ServiceError("compile_failed", f"{engine} is not on PATH")
        if not _which("pdftotext"):
            raise ServiceError("compile_failed", "pdftotext (poppler) is not on PATH")
        try:
            compiled = compile_pdf(tex_path, _pdf_name(self.profile, company))
        except (CompileError, OSError) as exc:
            raise ServiceError("compile_failed", str(exc)) from exc

        text = extract_text(compiled.pdf_path)
        parse_report = parse_check(text, self.profile, list(entries), compiled.pages)

        index = self.index
        forms = index_term_forms(index)
        must, nice = self._terms_for(labels, jd_text, index)
        selected = resolve_terms([*must, *nice], forms)
        coverage = keyword_coverage(
            text, selected, surface_form_set(forms), split_regions(text), must=must)
        write_reports(out_dir, parse_report, coverage)
        return CheckReport(
            out_dir=self.rel(out_dir),
            ok=parse_report.ok,
            pages=compiled.pages or page_count(compiled.pdf_path),
            parse=parse_report.to_dict(),
            coverage=coverage.to_dict(),
            summary=summarize(parse_report, coverage),
        )

    def _terms_for(self, labels: str | None, jd_text: str, index) -> tuple[list, list]:
        """``(must, nice)`` from a label file when there is one, else the JD."""
        from .ats.check import auto_terms

        if labels:
            path = Path(labels)
            if not path.is_absolute() and not path.suffix:
                path = self.labels_dir / f"{labels}.yaml"
            elif not path.is_absolute():
                path = self.repo_root / path
            if path.is_file():
                try:
                    import yaml

                    from .evaluation.labels import Label

                    label = Label.model_validate(yaml.safe_load(path.read_text()))
                except Exception:
                    label = None
                if label is not None and (label.must_terms or label.nice_terms):
                    return list(label.must_terms), list(label.nice_terms)
        must, nice = auto_terms(jd_text, index)
        return list(must), list(nice)

    def review_pack(self, run_id: str) -> ReviewPack:
        """One payload for a reviewing agent: posting, gates, choices, coverage."""
        from .ats.check import gap_terms

        run = self._run_row(run_id)
        data = dict(run.data or {})
        text, _path, _slug = self._posting_of(run)
        index = self.index
        bullets = {
            bullet.id: (item.id, section, bullet.text)
            for section, item in _visible(index)
            for bullet in item.bullets
        }
        edits = ((data.get("overrides") or {}).get("bullet_text") or {})
        selection = data.get("selection") or {}
        covers: dict[str, list[str]] = {}
        for step in selection.get("trace") or []:
            claim_id = str(step.get("claim_id", ""))
            covers.setdefault(claim_id, []).extend(
                str(pair[0]) for pair in (step.get("covered") or []) if pair
            )

        selected: list[SelectedClaim] = []
        for claim_id in selection.get("claims") or []:
            entry = bullets.get(str(claim_id))
            if entry is None:
                continue  # skill claims and headers are reported separately
            item_id, section, source = entry
            selected.append(SelectedClaim(
                bullet_id=str(claim_id), item=item_id, section=section,
                text=str(edits.get(claim_id, source)),
                covers=list(dict.fromkeys(covers.get(str(claim_id), []))),
            ))

        requirements = list(data.get("requirements") or [])
        coverage = data.get("coverage") or {}
        by_id = {str(r.get("id")): r for r in requirements}
        coverage_table = [
            {
                "id": rid,
                "kind": by_id.get(rid, {}).get("kind"),
                "weight": by_id.get(rid, {}).get("weight"),
                "coverage": round(float(value), 4),
                "text": by_id.get(rid, {}).get("text"),
            }
            for rid, value in sorted(coverage.items(), key=lambda kv: float(kv[1]))
        ]
        keyword_table = [
            {k: term.get(k) for k in ("term", "tier", "state", "count", "regions")}
            for term in ((data.get("check") or {}).get("coverage") or {}).get("terms") or []
        ]
        return ReviewPack(
            run_id=run.id,
            company=run.company or "",
            role=run.role or "",
            out_dir=self.rel(run.out_dir) if run.out_dir else "",
            jd_text=text,
            gates=dict(data.get("gates") or {}),
            requirements=requirements,
            selected_claims=selected,
            coverage_table=coverage_table,
            keyword_table=keyword_table,
            gaps=gap_terms(text, index, extra_known=[run.company or "", run.role or ""]),
            skills_lines=list(data.get("skill_lines") or []),
        )

    # ── Tracker ──────────────────────────────────────────────────────────────
    def runs(
        self,
        status: str | None = None,
        company: str | None = None,
        *,
        full: bool = False,
    ) -> list[RunRecord]:
        """Recorded runs, newest first.

        ``full`` returns each run's whole payload instead of the three fields a
        listing shows. A tracker table that wants keyword coverage as well as
        the page count would otherwise have to call :meth:`get_run` per row.
        """
        try:
            rows = self.store.list_runs(status=status, company=company)
        except StatusError as exc:
            raise ServiceError("invalid_input", str(exc)) from exc
        return [
            RunRecord(
                id=run.id, created_at=run.created_at, company=run.company,
                role=run.role, jd_hash=run.jd_hash, engine=run.engine,
                config_hash=run.config_hash,
                out_dir=self.rel(run.out_dir) if run.out_dir else None,
                status=run.status, final=run.final, pdf_sha256=run.pdf_sha256,
                data=dict(run.data) if full else {
                    "pages": run.data.get("pages"),
                    "coverage": run.coverage.get("must_coverage"),
                    "gates": run.data.get("gates"),
                },
            )
            for run in rows
        ]

    def item_usage(self, item_id: str) -> list[ItemUsage]:
        """Where one role or project ranked in every run that could have used it.

        The question an index editor is really asking — "what does changing
        this bullet affect?" — has an answer only the run records hold. Runs
        that predate the item, or that were made against a different index, are
        reported as not selected rather than skipped: "the solver has never
        picked this" is the finding.
        """
        item_id = _validate_id(item_id, "item id")
        owner = {
            bullet.id: item.id
            for _section, item in _visible(self.index)
            for bullet in item.bullets
        }
        usage: list[ItemUsage] = []
        for run in self.store.list_runs():
            selection = (run.data or {}).get("selection") or {}
            claims = [str(claim) for claim in selection.get("claims") or []]
            mine = [claim for claim in claims if owner.get(claim) == item_id]
            groups = selection.get("groups") or {}
            in_groups = any(
                item_id in [str(value) for value in (ids or [])]
                for ids in groups.values()
            )
            gains = {
                str(step.get("claim_id")): float(step.get("gain") or 0.0)
                for step in selection.get("trace") or []
            }
            rank = min((claims.index(claim) for claim in mine), default=None)
            usage.append(ItemUsage(
                run_id=run.id, created_at=run.created_at, company=run.company,
                role=run.role, status=run.status,
                selected=bool(mine or in_groups),
                rank=rank + 1 if rank is not None else None,
                bullets=mine,
                gain=max((gains.get(claim, 0.0) for claim in mine), default=None),
            ))
        return usage

    def set_status(self, run_id: str, status: str, note: str | None = None) -> RunRecord:
        run = self._run_row(run_id)
        try:
            self.store.set_status(run.id, status, note)
        except StatusError as exc:
            raise ServiceError("invalid_input", str(exc)) from exc
        except KeyError as exc:
            raise ServiceError("not_found", str(exc)) from exc
        return self.get_run(run.id)

    def add_note(self, run_id: str, note: str) -> RunRecord:
        run = self._run_row(run_id)
        try:
            self.store.add_note(run.id, note)
        except ValueError as exc:
            raise ServiceError("invalid_input", str(exc)) from exc
        return self.get_run(run.id)

    def quiet(self, days: int = 10) -> list[dict[str, Any]]:
        """Open applications nobody has heard from for ``days`` days."""
        from .runs.tracker import quiet_runs

        return [
            {
                "run_id": run.id, "company": run.company, "role": run.role,
                "status": run.status, "days_quiet": quiet_days,
                "out_dir": self.rel(run.out_dir) if run.out_dir else None,
            }
            for run, quiet_days in quiet_runs(self.store, int(days))
        ]

    # ── Index ────────────────────────────────────────────────────────────────
    def index_get(self, kind: str, entity_id: str = "") -> BaseModel:
        kind = self._kind(kind)
        index = self.index
        if kind == "summaries":
            return index.summaries
        entity_id = _validate_id(entity_id, f"{kind} id")
        if kind == "project":
            return _one(index.projects, entity_id, kind)
        if kind == "role":
            return _one(index.roles, entity_id, kind)
        if kind == "certification":
            return _one(index.certifications, entity_id, kind)
        if kind == "skill":
            for category in index.skills:
                for skill in category.skills:
                    if skill.id == entity_id:
                        return skill
            raise ServiceError("not_found", f"no skill with id {entity_id!r}")
        for education in index.education:
            if slugify(education.institution) == entity_id:
                return education
        raise ServiceError("not_found", f"no education entry with id {entity_id!r}")

    def normalise_kind(self, kind: str) -> str:
        """One of :data:`INDEX_KINDS`, accepting the obvious plurals."""
        value = str(kind or "").strip().lower()
        aliases = {"projects": "project", "roles": "role", "skills": "skill",
                   "cert": "certification", "certifications": "certification",
                   "summary": "summaries", "educations": "education"}
        value = aliases.get(value, value)
        if value not in INDEX_KINDS:
            raise ServiceError(
                "invalid_input",
                f"unknown index kind {kind!r}; expected one of: {', '.join(INDEX_KINDS)}",
            )
        return value

    #: Kept as the internal spelling every method in this file already uses.
    _kind = normalise_kind

    def entity_path(self, kind: str, entity_id: str = "") -> str:
        """Where an entity of this kind lives, repo-relative."""
        kind = self._kind(kind)
        if kind == "summaries":
            return self.rel(self.index_dir / "summaries.yaml")
        if kind == "education":
            return self.rel(self.index_dir / "education.yaml")
        if kind == "certification":
            return self.rel(self.index_dir / "certifications.yaml")
        entity_id = _validate_id(entity_id, f"{kind} id")
        if kind == "skill":
            for category in self.index.skills:
                if any(skill.id == entity_id for skill in category.skills):
                    return self.rel(
                        self.index_dir / "skills" / f"{slugify(category.category)}.yaml")
            raise ServiceError("not_found", f"no skill with id {entity_id!r}")
        return self.rel(self.index_dir / f"{kind}s" / f"{entity_id}.yaml")

    def index_put(
        self,
        kind: str,
        entity_id: str,
        patch: Mapping[str, Any],
        *,
        provenance: str | None = None,
    ) -> BaseModel:
        """Merge ``patch`` into one entity, validate the whole index, write it.

        Dicts merge recursively; lists replace — except ``bullets``, which
        merge by bullet id, so adding one bullet does not mean restating the
        other five. A new bullet needs ``id``, ``kind`` and ``text``; its
        ``metrics`` are filled in from its own numbers when it does not carry
        them, and ``provenance`` becomes a YAML comment beside its id.

        An id the index does not have creates the entity, provided the patch
        validates on its own. That is deliberate: the honesty rule
        (``docs/v2-design.md`` §2) is that facts stated in conversation get
        written back to the index in the same session, and a rule that could
        only ever edit what already exists would not survive its first new job.
        """
        kind = self._kind(kind)
        patch = _plain(patch)
        if not isinstance(patch, Mapping) or not patch:
            raise ServiceError("invalid_input", "the patch must be a non-empty mapping")
        index = self.index

        if kind == "summaries":
            model, new_bullets = self._merged(Summaries, index.summaries, patch, [])
            candidate = index.model_copy(update={"summaries": model})
        elif kind == "education":
            entity_id = _validate_id(entity_id, "education id")
            model, new_bullets, education = self._merged_in_list(
                Education, index.education, entity_id, patch,
                key=lambda e: slugify(e.institution))
            candidate = index.model_copy(update={"education": education})
        elif kind == "certification":
            entity_id = _validate_id(entity_id, "certification id")
            model, new_bullets, certs = self._merged_in_list(
                Certification, index.certifications, entity_id, patch,
                key=lambda c: c.id, set_id=True)
            candidate = index.model_copy(update={"certifications": certs})
        elif kind == "skill":
            entity_id = _validate_id(entity_id, "skill id")
            model, categories = self._merged_skill(index, entity_id, patch)
            new_bullets = []
            candidate = index.model_copy(update={"skills": categories})
        else:
            entity_id = _validate_id(entity_id, f"{kind} id")
            cls = Role if kind == "role" else Project
            items = index.roles if kind == "role" else index.projects
            model, new_bullets, updated = self._merged_in_list(
                cls, items, entity_id, patch, key=lambda i: i.id, set_id=True)
            candidate = index.model_copy(update={
                ("roles" if kind == "role" else "projects"): updated})

        try:
            validated = Index.model_validate(candidate.model_dump(mode="python"))
        except ValidationError as exc:
            raise ServiceError(
                "validation_failed",
                "the edit would make the index invalid",
                _errors_from(exc),
            ) from exc

        written = self._write_entity(kind, validated, model)
        if new_bullets and provenance:
            _comment_bullets(written, new_bullets, provenance)
        self._index = None
        self._sync()
        return model

    def _merged(self, cls, current, patch, new_bullets):
        base = current.model_dump(mode="python") if current is not None else {}
        merged = _merge(base, patch)
        try:
            return cls.model_validate(merged), new_bullets
        except ValidationError as exc:
            raise ServiceError(
                "validation_failed", f"the patched {cls.__name__.lower()} does not validate",
                _errors_from(exc),
            ) from exc

    def _merged_in_list(self, cls, items, entity_id, patch, key, set_id: bool = False):
        """``(model, new bullet ids, the updated list)`` for one entity in a list."""
        existing = next((item for item in items if key(item) == entity_id), None)
        base = existing.model_dump(mode="python") if existing is not None else {}
        if set_id and not base:
            base = {"id": entity_id}
        merged = _merge(base, dict(patch))
        if set_id:
            merged["id"] = entity_id
        new_bullets: list[str] = []
        if "bullets" in merged and "bullets" in patch:
            merged["bullets"], new_bullets = _merge_bullets(
                base.get("bullets") or [], patch.get("bullets") or [])
        try:
            model = cls.model_validate(merged)
        except ValidationError as exc:
            raise ServiceError(
                "validation_failed",
                f"the patched {cls.__name__.lower()} does not validate",
                _errors_from(exc),
            ) from exc
        updated = list(items)
        if existing is None:
            updated.append(model)
        else:
            updated[items.index(existing)] = model
        return model, new_bullets, updated

    def _merged_skill(self, index: Index, skill_id: str, patch: Mapping[str, Any]):
        """One skill merged inside its category (or a category named by the patch)."""
        patch = dict(patch)
        wanted = str(patch.pop("category", "") or "").strip()
        categories = [c.model_copy(deep=True) for c in index.skills]
        holder = next(
            (c for c in categories if any(s.id == skill_id for s in c.skills)), None)
        if holder is None:
            if not wanted:
                raise ServiceError(
                    "not_found",
                    f"no skill with id {skill_id!r}; pass \"category\" in the patch "
                    f"to create it",
                )
            holder = next((c for c in categories if c.category == wanted), None)
            if holder is None:
                holder = SkillCategory(category=wanted, skills=[])
                categories.append(holder)
        elif wanted and holder.category != wanted:
            raise ServiceError(
                "conflict",
                f"{skill_id!r} is in {holder.category!r}; moving a skill between "
                f"categories is a file move, not a patch",
            )
        existing = next((s for s in holder.skills if s.id == skill_id), None)
        base = existing.model_dump(mode="python") if existing is not None else {"id": skill_id}
        merged = {**_merge(base, patch), "id": skill_id}
        try:
            model = Skill.model_validate(merged)
        except ValidationError as exc:
            raise ServiceError(
                "validation_failed", "the patched skill does not validate",
                _errors_from(exc),
            ) from exc
        if existing is None:
            holder.skills.append(model)
        else:
            holder.skills[holder.skills.index(existing)] = model
        return model, categories

    def _write_entity(self, kind: str, index: Index, model) -> Path:
        if kind == "project":
            return save_project(model, self.index_dir)
        if kind == "role":
            return save_role(model, self.index_dir)
        if kind == "certification":
            return save_certifications(index.certifications, self.index_dir)
        if kind == "education":
            return save_education(index.education, self.index_dir)
        if kind == "summaries":
            return save_summaries(index.summaries, self.index_dir)
        category = next(
            c for c in index.skills if any(s.id == model.id for s in c.skills))
        return save_skill_category(category, self.index_dir)

    def _sync(self) -> None:
        from .runs.sync import sync

        try:
            sync(self.index_dir, store=self.store, index=self.index)
        except (OSError, ValueError):
            pass  # the index is the source of truth; a stale store is rebuildable

    def index_validate(self) -> tuple[list[str], list[str]]:
        """``(errors, warnings)``. Errors are fatal; warnings are the author's call."""
        try:
            index = load_index_v2(self.index_dir)
        except ValueError as exc:
            return [line.lstrip("- ") for line in str(exc).splitlines() if line.strip()], []
        except FileNotFoundError as exc:
            return [str(exc)], []
        return [], index.warnings()

    # ── Eval and labels ──────────────────────────────────────────────────────
    def eval(
        self,
        engine: str = "v2",
        provider: str = "none",
        only: Sequence[str] | None = None,
        variants: bool = True,
        variants_report: bool = False,
    ) -> EvalSummary:
        from .evaluation.evaluate import evaluate, render_table

        if engine not in {"v1", "v2"}:
            raise ServiceError("invalid_input", f"unknown engine {engine!r}")
        if provider not in set(config.PROVIDERS):
            raise ServiceError("invalid_input", f"unknown provider {provider!r}")
        try:
            payload = evaluate(
                self.index_dir, self.labels_dir, self.jd_dir, self.eval_dir,
                profile_path=self.profile_path, engine=engine, provider=provider,
                only=list(only) if only else None,
                variants=bool(variants), variants_report=bool(variants_report),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ServiceError("not_found", str(exc)) from exc
        return EvalSummary(
            path=self.rel(payload["path"]),
            config_hash=payload["config_hash"],
            table=render_table(payload),
            means=dict(payload.get("means") or {}),
            runs=len(payload.get("runs") or []),
            skipped=list(payload.get("skipped") or []),
            variants=dict(payload.get("variants") or {}),
        )

    def eval_baselines(self) -> list[EvalBaseline]:
        """Every ``eval/*.json`` on disk, newest first, without its per-run rows."""
        if not self.eval_dir.is_dir():
            return []
        found: list[tuple[float, EvalBaseline]] = []
        for path in self.eval_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or "means" not in payload:
                continue
            options = payload.get("options") or {}
            found.append((path.stat().st_mtime, EvalBaseline(
                path=self.rel(path),
                config_hash=str(payload.get("config_hash") or ""),
                created_at=str(payload.get("generated") or ""),
                engine=str(options.get("engine") or ""),
                provider=str(options.get("provider") or ""),
                runs=len(payload.get("runs") or []),
                means=dict(payload.get("means") or {}),
            )))
        return [baseline for _mtime, baseline in
                sorted(found, key=lambda pair: -pair[0])]

    def eval_report(self, path: str | Path) -> EvalSummary:
        """The rendered table for one baseline file already on disk."""
        from .evaluation.evaluate import render_table

        resolved = self.locate(path)
        if not resolved.is_file():
            raise ServiceError("not_found", f"no eval result at {self.rel(resolved)}")
        try:
            payload = json.loads(resolved.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ServiceError("validation_failed", f"{self.rel(resolved)}: {exc}") from exc
        return EvalSummary(
            path=self.rel(resolved),
            config_hash=str(payload.get("config_hash") or ""),
            table=render_table(payload),
            means=dict(payload.get("means") or {}),
            runs=len(payload.get("runs") or []),
            skipped=list(payload.get("skipped") or []),
        )

    def eval_compare(self, left: str | Path, right: str | Path) -> str:
        """The per-metric delta table between two baselines (``eval --compare``)."""
        from .evaluation.evaluate import compare

        paths = []
        for given in (left, right):
            resolved = self.locate(given)
            if not resolved.is_file():
                raise ServiceError("not_found", f"no eval result at {self.rel(resolved)}")
            paths.append(resolved)
        try:
            return compare(*paths)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ServiceError("validation_failed", str(exc)) from exc

    def labels_from_run(self, run_id: str) -> LabelUpdate:
        """Promote a run's pins and exclusions into its label file."""
        from .evaluation.labels import merge_overrides, slug as label_slug

        run = self._run_row(run_id)
        overrides = (run.data or {}).get("overrides") or {}
        pins = list(overrides.get("pin_items") or [])
        excludes = list(overrides.get("exclude_items") or [])
        if not pins and not excludes:
            return LabelUpdate(path="", changes=[])
        jd_path = (run.data or {}).get("jd_path")
        if not jd_path:
            raise ServiceError(
                "not_found",
                f"{run.id} has no recorded JD path, so its label file is unknown")
        try:
            path, changes = merge_overrides(
                self.labels_dir, label_slug(jd_path), pins, excludes,
                self.index.item_kinds())
        except FileNotFoundError as exc:
            raise ServiceError("not_found", str(exc)) from exc
        except ValueError as exc:
            raise ServiceError("validation_failed", str(exc)) from exc
        return LabelUpdate(path=self.rel(path), changes=changes)

    # ── Variants and the guardrail (Phase 5) ─────────────────────────────────
    def find_bullet(self, bullet_id: str):
        """``(bullet, item, kind)`` for one bullet id, anywhere in the index."""
        bullet_id = _validate_id(bullet_id, "bullet id")
        for kind, item in _visible(self.index):
            for bullet in item.bullets:
                if bullet.id == bullet_id:
                    return bullet, item, kind
        raise ServiceError("not_found", f"no bullet with id {bullet_id!r}")

    def propose_variant(
        self,
        bullet_id: str,
        archetype: str,
        text: str | None = None,
        requirement_ids: Sequence[str] | None = None,
        *,
        jd_slug: str | None = None,
        provider: str | None = None,
        provenance: str | None = None,
    ) -> ProposalInfo:
        """Propose alternate wording for one bullet, and judge it.

        With ``text`` the caller *is* the model — an agent through the MCP, or
        a person in the TUI — and only the guardrail runs. Without it the
        configured provider is asked, and a provider with nothing to say
        (``none``, ``external``, an unreachable model) is a
        ``provider_unavailable``, not an empty proposal.

        Either way the result is stored ``pending``. Nothing here writes the
        index; :meth:`accept_proposal` does, and only for a proposal that
        passed.
        """
        archetype = _clean_archetype(archetype)
        bullet, item, _kind = self.find_bullet(bullet_id)
        requirements = self._requirement_texts(requirement_ids, jd_slug)

        proposed = str(text or "").strip()
        source = "agent"
        if not proposed:
            engine = self._provider(provider)
            answer = engine.propose_variant(
                bullet.text, archetype, requirements,
                # Exactly what the guardrail will check, in its own words.
                metrics=source_metrics(bullet),
                said=source_technologies(bullet.text, item, self.index),
                allowed=list(item_terms(item, self.index)),
            )
            proposed = str(answer or "").strip()
            source = engine.name
            if not proposed:
                raise ServiceError(
                    "provider_unavailable",
                    f"provider {engine.name!r} proposed nothing for {bullet.id}; "
                    f"pass `text` to submit wording of your own",
                )

        report = check_variant(bullet, item, proposed, self.index)
        stored = self.store.add_proposal(
            bullet_id=bullet.id,
            archetype=archetype,
            text=proposed,
            guardrail=report.model_dump(mode="json"),
            provenance=str(provenance or source),
        )
        return self._proposal_info(stored)

    def proposals(
        self,
        status: str | None = None,
        bullet_id: str | None = None,
        archetype: str | None = None,
    ) -> list[ProposalInfo]:
        """Every stored proposal, newest first, optionally filtered."""
        from .runs.store import ProposalStatusError

        try:
            rows = self.store.list_proposals(
                status=status, bullet_id=bullet_id,
                archetype=_clean_archetype(archetype) if archetype else None,
            )
        except ProposalStatusError as exc:
            raise ServiceError("invalid_input", str(exc)) from exc
        return [self._proposal_info(row) for row in rows]

    def get_proposal(self, proposal_id: str) -> ProposalInfo:
        row = self.store.get_proposal(str(proposal_id or "").strip())
        if row is None:
            raise ServiceError("not_found", f"no proposal with id {proposal_id!r}")
        return self._proposal_info(row)

    def accept_proposal(self, proposal_id: str) -> ProposalInfo:
        """Write an accepted proposal into the index, and mark it accepted.

        **This changes tracked content.** The text lands in the bullet's
        ``variants[<archetype>]`` (or ``summaries.variants``) with a provenance
        comment beside it, and every future run for that archetype prints it.
        A proposal the guardrail refused cannot be accepted at all: the point
        of the guardrail is that no human has to remember to check.
        """
        from .runs.store import ProposalStatusError

        info = self.get_proposal(proposal_id)
        if info.status != "pending":
            raise ServiceError(
                "conflict", f"{info.id} is already {info.status}")
        if not info.ok:
            raise ServiceError(
                "validation_failed",
                f"{info.id} did not pass the guardrail and cannot be accepted",
                [f"{v.rule}: {v.detail}"
                 for v in (info.guardrail.violations if info.guardrail else [])],
            )

        if info.kind == "summary":
            summaries = self.index.summaries
            self.index_put(
                "summaries", "",
                {"variants": {**dict(summaries.variants), info.archetype: info.text}},
            )
            path = self.index_dir / "summaries.yaml"
            _comment_variant(path, None, info.archetype, info.provenance or "agent")
        else:  # noqa: RET506 - the two branches write two different files
            bullet, item, kind = self.find_bullet(info.bullet_id)
            self.index_put(
                kind, item.id,
                {"bullets": [{"id": bullet.id,
                              "variants": {info.archetype: info.text}}]},
            )
            path = self.locate(self.entity_path(kind, item.id))
            _comment_variant(path, bullet.id, info.archetype,
                             info.provenance or "agent")
        self._index = None

        try:
            decided = self.store.decide_proposal(info.id, "accepted")
        except (KeyError, ProposalStatusError) as exc:
            raise ServiceError("conflict", str(exc)) from exc
        return self._proposal_info(decided).model_copy(
            update={"path": self.rel(path)})

    def reject_proposal(self, proposal_id: str, reason: str | None = None) -> ProposalInfo:
        """Mark a proposal rejected. Nothing is written to the index."""
        from .runs.store import ProposalStatusError

        info = self.get_proposal(proposal_id)
        if info.status != "pending":
            raise ServiceError("conflict", f"{info.id} is already {info.status}")
        try:
            decided = self.store.decide_proposal(info.id, "rejected", reason)
        except (KeyError, ProposalStatusError) as exc:
            raise ServiceError("conflict", str(exc)) from exc
        return self._proposal_info(decided)

    def propose_summary(
        self,
        slug: str,
        text: str | None = None,
        *,
        provider: str | None = None,
        provenance: str | None = None,
    ) -> ProposalInfo:
        """Propose a tightened Summary for one posting's archetype.

        The starting point is what a run of this posting would print — the
        archetype's variant if the index has one, the default otherwise, plus
        the target title — and the guardrail is the same contract with one
        difference: a summary may only restate what the résumé already says, so
        every technology and every number in it must appear in the index's
        summaries, its highlights, or the claims this posting selected.
        """
        profile = self.jd_profile(slug, provider=provider)
        archetype = _clean_archetype(profile.archetype or "")
        summaries = self.index.summaries
        current = str(
            summaries.variants.get(archetype) or summaries.default or "").strip()
        if not current:
            raise ServiceError(
                "not_found",
                "the index has no summary to polish; write one with "
                "index_put(\"summaries\", …) first",
            )
        allowed = self._summary_corpus(slug)

        proposed = str(text or "").strip()
        source = "agent"
        if not proposed:
            engine = self._provider(provider)
            answer = engine.propose_summary(
                current, archetype, profile.title, allowed)
            proposed = str(answer or "").strip()
            source = engine.name
            if not proposed:
                raise ServiceError(
                    "provider_unavailable",
                    f"provider {engine.name!r} proposed nothing for the summary; "
                    f"pass `text` to submit wording of your own",
                )

        report = check_summary(proposed, allowed, source=current, index=self.index)
        stored = self.store.add_proposal(
            bullet_id=SUMMARY_TARGET,
            archetype=archetype,
            text=proposed,
            guardrail=report.model_dump(mode="json"),
            provenance=str(provenance or source),
        )
        return self._proposal_info(stored, source_text=current)

    # ── the parts the four methods above share ──
    def _proposal_info(self, row, source_text: str | None = None) -> ProposalInfo:
        kind = "summary" if row.bullet_id == SUMMARY_TARGET else "bullet"
        source, item_id = str(source_text or ""), ""
        if kind == "bullet":
            try:
                bullet, item, _kind = self.find_bullet(row.bullet_id)
                source, item_id = bullet.text, item.id
            except ServiceError:
                pass  # the bullet may have been renamed since; the row stands
        elif not source:
            summaries = self.index.summaries
            source = str(summaries.variants.get(row.archetype)
                         or summaries.default or "")
        guardrail = None
        if row.guardrail is not None:
            try:
                guardrail = GuardrailReport.model_validate(row.guardrail)
            except ValidationError:
                guardrail = None
        return ProposalInfo(
            id=row.id, bullet_id=row.bullet_id, archetype=row.archetype,
            text=row.text, status=row.status, provenance=row.provenance,
            created_at=row.created_at, decided_at=row.decided_at,
            note=getattr(row, "note", None),
            source=source, item_id=item_id, kind=kind, guardrail=guardrail,
        )

    def _requirement_texts(
        self, requirement_ids: Sequence[str] | None, jd_slug: str | None = None
    ) -> list[str]:
        """The requirements a proposal is being aligned to, as their own text.

        Ids are the extractor's (``req-3``), which only mean something next to
        a posting: ``jd_slug`` names it, and without one the recorded runs are
        searched newest first — which is what the TUI means, where the ids on
        screen came from the run that is open. An id nothing resolves is
        dropped rather than passed to a model as the literal string "req-3".
        """
        wanted = [str(i).strip() for i in (requirement_ids or []) if str(i).strip()]
        if not wanted:
            return []
        found: dict[str, str] = {}
        if jd_slug:
            for spec in self.jd_profile(jd_slug).requirements:
                if spec.id in wanted:
                    found[spec.id] = spec.text
        if len(found) < len(wanted):
            for run in self.store.list_runs():
                for row in (run.data or {}).get("requirements") or []:
                    rid = str(row.get("id") or "")
                    if rid in wanted and rid not in found:
                        found[rid] = str(row.get("text") or "")
                if len(found) == len(wanted):
                    break
        return [found[rid] for rid in wanted if found.get(rid)][:PROPOSAL_REQUIREMENTS]

    def _summary_corpus(self, slug: str | None = None) -> list[str]:
        """Everything a summary is allowed to be made of.

        Exactly what ``docs/v2-design.md`` §5 allows it to restate: the index's
        summaries, its highlights, and the claims this posting selected — the
        newest recorded run of it when there is one, every visible bullet when
        there is not. Deliberately *not* the skills taxonomy: a skill the index
        lists and no bullet demonstrates is a line on the Skills row, not a
        sentence about what the candidate has done.
        """
        index = self.index
        lines: list[str] = [index.summaries.default, *index.summaries.highlights]
        lines.extend(str(v) for v in index.summaries.variants.values())
        selected: list[str] = []
        if slug:
            for run in self.store.list_runs():
                data = run.data or {}
                if Path(str(data.get("jd_path") or "")).stem != slug:
                    continue
                selected = list((data.get("selection") or {}).get("claims") or [])
                break
        chosen = set(selected)
        for _kind, item in _visible(index):
            if chosen and not any(b.id in chosen for b in item.bullets):
                continue
            lines.append(getattr(item, "summary", "") or "")
            lines.append(getattr(item, "title", "") or "")
            lines.append(getattr(item, "company", "") or "")
            for bullet in item.bullets:
                if not chosen or bullet.id in chosen:
                    lines.append(bullet.text)
                    lines.extend(str(v) for v in (bullet.variants or {}).values())
        return [line for line in (str(v).strip() for v in lines) if line]


# ── module helpers ───────────────────────────────────────────────────────────
def _which(name: str) -> bool:
    import shutil

    return bool(shutil.which(name))


def _latex_engine() -> str:
    """The typesetter a compile would use here — ``latexmk`` or ``tectonic``.

    Read at call time rather than at import, because the container sets
    ``ATS_LATEX_ENGINE`` and a long-lived service must not have cached the
    answer from before.
    """
    from .ats.check import resolve_latex_engine

    try:
        return resolve_latex_engine()
    except ValueError as exc:
        raise ServiceError("invalid_input", str(exc)) from exc


def _pdf_name(profile: Mapping[str, Any], company: str) -> str:
    from .ats.check import output_filename

    return output_filename(str(profile.get("name", "")), company)


def _mean(values, indices: Sequence[int]) -> float | None:
    picked = [float(values[i]) for i in indices]
    return round(sum(picked) / len(picked), 4) if picked else None


def _gates_json(profile: JDProfile) -> dict[str, Any]:
    from .jd.gates import gate_dict

    return gate_dict(profile.gates)


def gate_flags(gates: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(name, one-line reason)`` for every gate that says something.

    Takes the JSON shape a run record and :class:`FitReport` carry, so a
    front-end can read a stored payload without importing the gate models.
    """
    blocking: list[tuple[str, str]] = []
    for name, gate in (gates or {}).items():
        if not isinstance(gate, Mapping):
            continue
        verdict = str(gate.get("verdict", "")).lower()
        if verdict in {"fail", "flag"}:
            detail = gate.get("note") or gate.get("quote") or ""
            blocking.append((name, f"{verdict} {detail}".strip()))
    return blocking


def _one(items: Iterable[Any], entity_id: str, kind: str):
    for item in items:
        if item.id == entity_id:
            return item
    raise ServiceError("not_found", f"no {kind} with id {entity_id!r}")


def _visible(index: Index):
    return [("role", r) for r in index.roles if not r.hidden] + [
        ("project", p) for p in index.projects]


def _merge_bullets(
    existing: Sequence[Mapping[str, Any]], patch: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Merge bullets by id: known ids update in place, new ids append."""
    merged = [dict(bullet) for bullet in existing]
    positions = {str(b.get("id")): i for i, b in enumerate(merged)}
    added: list[str] = []
    for raw in patch:
        if not isinstance(raw, Mapping):
            raise ServiceError("invalid_input", "each bullet must be a mapping")
        bullet = dict(raw)
        bullet_id = str(bullet.get("id") or "").strip()
        if not bullet_id:
            raise ServiceError("invalid_input", "a bullet needs an id")
        if bullet_id in positions:
            merged[positions[bullet_id]] = _merge(merged[positions[bullet_id]], bullet)
            continue
        missing = [key for key in ("kind", "text") if not str(bullet.get(key) or "").strip()]
        if missing:
            raise ServiceError(
                "invalid_input",
                f"a new bullet needs {', '.join(['id', *missing])} "
                f"({bullet_id!r} is missing {', '.join(missing)})",
            )
        bullet.setdefault("metrics", auto_metrics(bullet.get("text", "")))
        if not bullet["metrics"]:
            bullet.pop("metrics")
        try:
            Bullet.model_validate(bullet)
        except ValidationError as exc:
            raise ServiceError(
                "validation_failed", f"bullet {bullet_id!r} does not validate",
                _errors_from(exc),
            ) from exc
        merged.append(bullet)
        added.append(bullet_id)
    return merged, added


def _comment_variant(
    path: Path, bullet_id: str | None, archetype: str, provenance: str
) -> None:
    """Mark an accepted variant in the YAML with where it came from.

    ``# proposed 2026-08-26 via external, accepted`` beside the archetype key,
    for the same reason a new bullet gets one: the person reading the pull
    request is entitled to know that a sentence in their résumé was written by
    a model and approved by them, and a schema field would carry that fact into
    the store, the report and every payload instead.
    """
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    try:
        data = yaml.load(path.read_text())
    except Exception:
        return
    if not isinstance(data, dict):
        return
    if bullet_id is None:
        holder = data.get("variants")
    else:
        holder = None
        for bullet in data.get("bullets") or []:
            if str(bullet.get("id")) == str(bullet_id):
                holder = bullet.get("variants")
                break
    if holder is None or archetype not in holder:
        return
    if archetype in getattr(holder.ca, "items", {}):
        return  # a comment is already there; do not stack them up
    stamp = datetime.now().date().isoformat()
    holder.yaml_add_eol_comment(f"proposed {stamp} via {provenance}, accepted",
                                archetype)
    import io as _io

    buffer = _io.StringIO()
    yaml.dump(data, buffer)
    path.write_text(buffer.getvalue())


def _comment_bullets(path: Path, bullet_ids: Sequence[str], provenance: str) -> None:
    """Mark new bullets in the YAML with where they came from.

    A comment rather than a field: provenance is for the human reading the diff
    in a pull request, and adding it to the schema would put it in the store,
    the report and every projection for no reader's benefit.
    """
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    try:
        data = yaml.load(path.read_text())
    except Exception:
        return
    bullets = data.get("bullets") if isinstance(data, dict) else None
    if not bullets:
        return
    stamp = datetime.now().date().isoformat()
    marked = False
    for bullet in bullets:
        if str(bullet.get("id")) in set(bullet_ids) and "id" not in getattr(
            bullet.ca, "items", {}
        ):
            bullet.yaml_add_eol_comment(f"added {stamp} via {provenance}", "id")
            marked = True
    if not marked:
        return
    import io

    buffer = io.StringIO()
    yaml.dump(data, buffer)
    path.write_text(buffer.getvalue())
