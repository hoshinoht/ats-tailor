"""Everything the TUI asks of the engine, and the only file that may ask.

The rule from ``docs/v2-design.md`` §7 is that a front-end is an adapter: it
parses input, it formats output, and in between it calls a :class:`Service`
method. So the widgets in this package import *this* module and nothing else
from ``ats_tailor`` — no pipeline, no solver, no store, no YAML.

Two jobs live here. The first is :func:`attempt`, which turns the service's one
error type into a value: a Textual worker running on a thread cannot raise into
the UI, and a screen that has to wrap every call in ``try`` is a screen that
will eventually forget to. The second is shaping — the service returns what is
true about a run, and a table needs rows.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..service import (
    INDEX_KINDS,
    RUN_STATUSES,
    EvalBaseline,
    ItemUsage,
    OverridesPatch,
    RunRecord,
    Service,
    ServiceError,
    gate_flags,
)

__all__ = [
    "Actions",
    "ApplicationRow",
    "Outcome",
    "RequirementRow",
    "SelectionNode",
    "TailorView",
    "attempt",
    "INDEX_KINDS",
    "RUN_STATUSES",
]

#: How long an application may go unmentioned before the tracker says so.
QUIET_DAYS = 10

#: Coverage thresholds the requirement list is coloured by (``phase-4b`` §1).
COVERED = 0.6
PARTIAL = 0.3

#: The weights ``w`` offers. Four values rather than a free number: a
#: requirement is either half-meant, meant, or the reason the posting exists.
WEIGHTS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)


# ── Calling the service ──────────────────────────────────────────────────────
@dataclass
class Outcome:
    """The result of one service call, success or failure, as a value."""

    ok: bool
    value: Any = None
    code: str = ""
    message: str = ""
    details: list[str] = field(default_factory=list)

    def toast(self) -> str:
        """What a notification should say."""
        line = f"[{self.code}] {self.message}" if self.code else self.message
        if self.details:
            line += "\n" + "\n".join(f"• {detail}" for detail in self.details[:6])
        return line


def attempt(call: Callable[[], Any]) -> Outcome:
    """Run ``call``; never raise.

    ``ServiceError`` keeps its code, because the code is what the toast shows
    and what a user can act on. Anything else is reported under ``internal``
    rather than killing the app: a TUI that dies on a malformed YAML file is
    worse at its job than one that says so in a corner.
    """
    try:
        return Outcome(True, call())
    except ServiceError as exc:
        details = exc.details if isinstance(exc.details, list) else (
            [str(exc.details)] if exc.details is not None else [])
        return Outcome(False, code=exc.code, message=exc.message,
                       details=[str(detail) for detail in details])
    except Exception as exc:  # noqa: BLE001 - a front-end may not crash
        return Outcome(False, code="internal", message=f"{type(exc).__name__}: {exc}")


# ── Row shapes ───────────────────────────────────────────────────────────────
@dataclass
class ApplicationRow:
    """One line of the tracker table."""

    run_id: str
    created: str
    company: str
    role: str
    status: str
    flags: list[tuple[str, str]]
    pages: int | None
    must: float | None
    nice: float | None
    context: float | None
    days_quiet: int | None

    @property
    def flagged(self) -> bool:
        return bool(self.flags)

    def status_cell(self) -> str:
        return f"⚠ {self.status}" if self.flagged else self.status


@dataclass
class RequirementRow:
    requirement_id: str
    kind: str
    weight: float
    coverage: float
    text: str

    @property
    def band(self) -> str:
        """``covered`` | ``partial`` | ``missing`` — what the colour means."""
        if self.coverage >= COVERED:
            return "covered"
        return "partial" if self.coverage >= PARTIAL else "missing"


@dataclass
class SelectionNode:
    """One row of the right-hand tree: a section, an item, or a bullet."""

    kind: str  # "section" | "item" | "bullet"
    node_id: str
    label: str
    item_id: str = ""
    children: list["SelectionNode"] = field(default_factory=list)


@dataclass
class TailorView:
    """One run, in the shape the three panes draw.

    Built by a single worker call so the screen is never half-loaded: a preview
    from the previous run beside this run's requirements would be a lie about
    what is on the page.
    """

    run_id: str
    company: str
    role: str
    out_dir: str
    status: str
    pages: int | None
    must: float | None
    nice: float | None
    context: float | None
    gates: list[tuple[str, str]]
    gates_summary: str
    requirements: list[RequirementRow]
    sections: list[SelectionNode]
    preview: str
    preview_error: str
    overrides_summary: str

    def header(self) -> str:
        """"page N / must X% / ctx Y%" — the three numbers a run is judged by."""
        return (
            f"page {self.pages if self.pages is not None else '?'}"
            f"  /  must {_percent(self.must)}"
            f"  /  ctx {_percent(self.context)}"
        )


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{float(value) * 100:.0f}%"


def format_percent(value: float | None) -> str:
    return _percent(value)


# ── The facade ───────────────────────────────────────────────────────────────
class Actions:
    """One object holding the service, with the calls the screens make.

    Every method here is synchronous and may be slow; the app runs them in a
    Textual thread worker. None of them touch a widget.
    """

    def __init__(self, service: Service):
        self.service = service

    # ── status bar ──
    @property
    def repo(self) -> Path:
        return self.service.repo_root

    @property
    def provider(self) -> str:
        return self.service.provider_name

    def set_provider(self, name: str) -> str:
        self.service.provider_name = str(name).strip().lower() or "none"
        return self.service.provider_name

    def health(self) -> dict[str, Any]:
        return self.service.healthcheck()

    def status_line(self, health: Mapping[str, Any] | None = None) -> str:
        payload = dict(health or {})
        store = (payload.get("store") or {}).get("path", "data/ats.db")
        return (
            f" {payload.get('repo_root', self.service.repo_root)}"
            f"   engine {payload.get('engine', 'v2')}"
            f"/{payload.get('latex_engine', '?')}"
            f"   provider {payload.get('provider', self.provider)}"
            f"   store {store}"
        )

    # ── applications ──
    def applications(self, quiet_only: bool = False) -> list[ApplicationRow]:
        quiet = {row["run_id"]: int(row["days_quiet"])
                 for row in self.service.quiet(0)}
        rows: list[ApplicationRow] = []
        for record in self.service.runs(full=True):
            data = record.data or {}
            coverage = ((data.get("check") or {}).get("coverage") or {})
            rows.append(ApplicationRow(
                run_id=record.id,
                created=(record.created_at or "")[:16].replace("T", " "),
                company=record.company or "",
                role=record.role or "",
                status=record.status,
                flags=gate_flags(data.get("gates") or {}),
                pages=data.get("pages"),
                must=_number(coverage.get("must_coverage")),
                nice=_number(coverage.get("nice_coverage")),
                context=_number(coverage.get("context_rate")),
                days_quiet=quiet.get(record.id),
            ))
        if quiet_only:
            rows = [row for row in rows
                    if row.days_quiet is not None and row.days_quiet >= QUIET_DAYS]
        return rows

    def run_record(self, run_id: str) -> RunRecord:
        return self.service.get_run(run_id)

    def notes(self, run_id: str) -> list[tuple[str, str]]:
        record = self.service.get_run(run_id)
        return [(note.created_at, note.note) for note in record.notes]

    def set_status(self, run_id: str, status: str) -> RunRecord:
        return self.service.set_status(run_id, status)

    def add_note(self, run_id: str, note: str) -> RunRecord:
        return self.service.add_note(run_id, note)

    def add_posting(
        self, text: str, company: str, role: str, source_url: str | None = None
    ) -> dict[str, Any]:
        """Store a pasted posting, then triage it. One keystroke, two answers."""
        info = self.service.add_jd(text, company, role, source_url or None)
        report = self.service.fit(info.slug)
        return {"jd": info, "fit": report}

    # ── one run ──
    def tailor_view(self, run_id: str) -> TailorView:
        record = self.service.get_run(run_id)
        data = record.data or {}
        pack = self.service.review_pack(record.id)
        coverage = ((data.get("check") or {}).get("coverage") or {})
        requirements = [
            RequirementRow(
                requirement_id=str(row.get("id") or ""),
                kind=str(row.get("kind") or "must"),
                weight=float(row.get("weight") or 0.0),
                coverage=float(row.get("coverage") or 0.0),
                text=str(row.get("text") or ""),
            )
            for row in pack.coverage_table
        ]
        gains = {
            str(step.get("claim_id")): float(step.get("gain") or 0.0)
            for step in (data.get("selection") or {}).get("trace") or []
        }
        overrides = attempt(lambda: self.service.get_overrides(record.id))
        preview, preview_error = "", ""
        text = attempt(lambda: self.service.run_text(record.id))
        if text.ok:
            preview = str(text.value)
        else:
            preview_error = text.toast()
        return TailorView(
            run_id=record.id,
            company=record.company or "",
            role=record.role or "",
            out_dir=record.out_dir or "",
            status=record.status,
            pages=data.get("pages"),
            must=_number(coverage.get("must_coverage")),
            nice=_number(coverage.get("nice_coverage")),
            context=_number(coverage.get("context_rate")),
            gates=gate_flags(data.get("gates") or {}),
            gates_summary=_gates_summary(data.get("gates") or {}),
            requirements=requirements,
            sections=_selection_tree(pack, gains),
            preview=preview,
            preview_error=preview_error,
            overrides_summary=(
                overrides.value.summary if overrides.ok else overrides.toast()),
        )

    def pin_item(self, run_id: str, item_id: str) -> str:
        return self._overrides(run_id, OverridesPatch(pin_items=[item_id]))

    def exclude_item(self, run_id: str, item_id: str) -> str:
        return self._overrides(run_id, OverridesPatch(exclude_items=[item_id]))

    def pin_bullet(self, run_id: str, bullet_id: str) -> str:
        return self._overrides(run_id, OverridesPatch(pin_bullets=[bullet_id]))

    def exclude_bullet(self, run_id: str, bullet_id: str) -> str:
        return self._overrides(run_id, OverridesPatch(exclude_bullets=[bullet_id]))

    def reweight(self, run_id: str, requirement_id: str, weight: float) -> str:
        return self._overrides(
            run_id, OverridesPatch(requirement_reweight={requirement_id: float(weight)}))

    def _overrides(self, run_id: str, patch: OverridesPatch) -> str:
        info = self.service.set_overrides(run_id, patch)
        return "; ".join(info.changes) or "already set"

    def rerun(self, run_id: str) -> str:
        summary = self.service.rerun(run_id)
        return (
            f"{summary.run_id or run_id}: {summary.pages or '?'} page(s), "
            f"must {format_percent(summary.coverage.get('must'))}"
        )

    def check(self, run_id: str) -> str:
        report = self.service.check(run_id)
        return report.summary or ("parse ok" if report.ok else "parse failed")

    def pdf_path(self, run_id: str) -> Path:
        return self.service.locate(self.service.run_pdf(run_id))

    def open_pdf(self, run_id: str) -> str:
        """Hand the PDF to whatever the desktop uses. Never blocks the UI."""
        path = self.pdf_path(run_id)
        opener = _opener()
        if opener is None:
            raise ServiceError(
                "not_implemented",
                f"no system opener on this platform; the PDF is at {path}")
        subprocess.Popen(  # noqa: S603 - a fixed argv and a path inside the repo
            [*opener, str(path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(path)

    def page_image(self, run_id: str, out_dir: Path) -> Path:
        """Page 1 of the run's PDF as a PNG, for the inline raster."""
        if not shutil.which("pdftoppm"):
            raise ServiceError(
                "compile_failed", "pdftoppm (poppler) is not on PATH")
        pdf_path = self.pdf_path(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = out_dir / f"{run_id}-page"
        result = subprocess.run(  # noqa: S603 - fixed argv
            ["pdftoppm", "-png", "-r", "110", "-f", "1", "-l", "1",
             str(pdf_path), str(prefix)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise ServiceError("compile_failed",
                               f"pdftoppm failed: {result.stderr.strip()}")
        rendered = sorted(out_dir.glob(f"{run_id}-page*.png"))
        if not rendered:
            raise ServiceError("not_found", "pdftoppm wrote no image")
        return rendered[0]

    def labels_from_run(self, run_id: str) -> str:
        update = self.service.labels_from_run(run_id)
        if not update.path:
            return "nothing pinned or excluded in this run"
        return f"{update.path}: " + ("; ".join(update.changes) or "no change")

    # ── index ──
    def index_tree(self) -> list[tuple[str, str, list[tuple[str, str]]]]:
        """``(kind, group label, [(entity id, label)])`` for the whole tree."""
        index = self.service.index
        groups: list[tuple[str, str, list[tuple[str, str]]]] = [
            ("project", "projects",
             [(item.id, item.name) for item in index.projects]),
            ("role", "roles",
             [(item.id, f"{item.title} · {item.company}") for item in index.roles]),
        ]
        for category in index.skills:
            groups.append((
                "skill", f"skills · {category.category}",
                [(skill.id, skill.name) for skill in category.skills],
            ))
        groups.append((
            "certification", "certifications",
            [(item.id, item.name) for item in index.certifications],
        ))
        return groups

    def index_get(self, kind: str, entity_id: str) -> dict[str, Any]:
        return self.service.index_get(kind, entity_id).model_dump(mode="json")

    def index_put(self, kind: str, entity_id: str, patch: Mapping[str, Any]) -> str:
        self.service.index_put(kind, entity_id, dict(patch), provenance="tui")
        errors, warnings = self.service.index_validate()
        if errors:
            raise ServiceError("validation_failed", "the index no longer loads", errors)
        note = f"saved {kind} {entity_id}"
        return f"{note} ({len(warnings)} warning(s))" if warnings else note

    def index_validate(self) -> tuple[list[str], list[str]]:
        return self.service.index_validate()

    def item_usage(self, item_id: str) -> list[ItemUsage]:
        return self.service.item_usage(item_id)

    def entity_path(self, kind: str, entity_id: str) -> str:
        return self.service.entity_path(kind, entity_id)

    # ── eval ──
    def baselines(self) -> list[EvalBaseline]:
        return self.service.eval_baselines()

    def eval_table(self, path: str) -> str:
        return self.service.eval_report(path).table

    def eval_compare(self, left: str, right: str) -> str:
        return self.service.eval_compare(left, right)

    def run_eval(self, engine: str, provider: str) -> str:
        summary = self.service.eval(engine=engine, provider=provider)
        return f"{summary.path}  ({summary.runs} run(s), config {summary.config_hash[:12]})"

    @property
    def providers(self) -> tuple[str, ...]:
        from .. import config as engine_config

        return tuple(engine_config.PROVIDERS)


# ── helpers ──────────────────────────────────────────────────────────────────
def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _gates_summary(gates: Mapping[str, Any]) -> str:
    flags = gate_flags(gates)
    if not gates:
        return "no gates recorded for this posting"
    if not flags:
        return f"{len(gates)} gate(s), none blocking"
    return f"{len(flags)} of {len(gates)} gate(s) flagged"


def _selection_tree(pack, gains: Mapping[str, float]) -> list[SelectionNode]:
    """Selected claims as section → item → bullet, in selection order."""
    sections: dict[str, SelectionNode] = {}
    items: dict[tuple[str, str], SelectionNode] = {}
    for claim in pack.selected_claims:
        section = claim.section or "other"
        node = sections.get(section)
        if node is None:
            node = SelectionNode(kind="section", node_id=section, label=section)
            sections[section] = node
        key = (section, claim.item)
        item_node = items.get(key)
        if item_node is None:
            item_node = SelectionNode(
                kind="item", node_id=claim.item, label=claim.item, item_id=claim.item)
            items[key] = item_node
            node.children.append(item_node)
        gain = float(gains.get(claim.bullet_id, 0.0))
        covers = ", ".join(claim.covers[:3])
        label = f"{claim.bullet_id}  +{gain:.3f}"
        if covers:
            label += f"  → {covers}"
        item_node.children.append(SelectionNode(
            kind="bullet", node_id=claim.bullet_id, label=label,
            item_id=claim.item,
        ))
    return list(sections.values())


def _opener() -> Sequence[str] | None:
    system = platform.system()
    if system == "Darwin" and shutil.which("open"):
        return ["open"]
    if system == "Windows":
        return ["cmd", "/c", "start", ""]
    for candidate in ("xdg-open", "gio"):
        if shutil.which(candidate):
            return [candidate, "open"] if candidate == "gio" else [candidate]
    if os.environ.get("BROWSER"):
        return [os.environ["BROWSER"]]
    return None
