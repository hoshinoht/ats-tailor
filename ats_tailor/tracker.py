"""Applications, statuses and follow-ups — what happened after the run.

A tailored resume is not the end of the job; it is the start of an application
that goes quiet for three weeks. The tracker is deliberately dumb: it records
what was sent, when, and what came back, in one vocabulary
(``docs/v2-design.md`` §7), and it never infers an outcome. Outcomes are far too
sparse to learn from and far too easy to guess wrong about, so every status
change is a thing the user said, kept as a dated note.

Facts get written back here (the PDF's hash, the coverage the check measured,
the overrides that were applied) precisely so a later question — "what did I
actually send them?" — has an answer that is not a memory.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .store import FINAL_STATUSES, INITIAL_STATUS, Run, Store

#: An application with no note for this many days is "quiet".
QUIET_DAYS = 10


def run_id(
    company: str,
    role: str,
    when: datetime | None = None,
    taken: Callable[[str], bool] | None = None,
) -> str:
    """``<company>-<role>-<YYYYMMDD-HHMMSS>`` — sortable, unique per second.

    "Per second" is not "unique". Two runs of the same application inside one
    second (a script, a retry, a `--reextract` loop) would otherwise collide,
    and the second would overwrite the first's record — the one thing a record
    exists not to do. ``taken`` answers whether an id is already in the store;
    the suffix counts up from ``-2`` until it is not.
    """
    from .tailor import _safe_slug

    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = f"{_safe_slug(company, role)}-{stamp}"
    if taken is None:
        return base
    candidate, suffix = base, 1
    while taken(candidate):
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


def file_sha256(path: str | Path) -> str | None:
    """The hash of the artefact that was actually sent, or ``None`` if absent."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(
    store: Store,
    result,
    jd_hash: str,
    jd_path: str | Path | None = None,
    config_hash: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    check: Mapping[str, Any] | None = None,
    pdf_path: str | Path | None = None,
    identifier: str | None = None,
) -> Run:
    """Write one run into the store. Everything the CLI knows, in one row."""
    payload: dict[str, Any] = dict(result.to_dict())
    payload["jd_hash"] = jd_hash
    if jd_path:
        payload["jd_path"] = str(jd_path)
    if overrides:
        payload["overrides"] = dict(overrides)
    if check:
        payload["check"] = dict(check)
    return store.record_run(
        run_id=identifier or run_id(
            result.company, result.role,
            taken=lambda candidate: store.get_run(candidate) is not None),
        company=result.company,
        role=result.role,
        data=payload,
        jd_hash=jd_hash,
        engine=result.engine,
        config_hash=config_hash,
        out_dir=result.out_dir,
        status=INITIAL_STATUS,
        pdf_sha256=file_sha256(pdf_path) if pdf_path else None,
    )


# ── Reading it back ──────────────────────────────────────────────────────────
def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.0f}%"


def gate_mark(run: Run) -> str:
    """``⚠`` when the posting's gates carry a ``fail`` or a ``flag``."""
    gates = (run.data or {}).get("gates") or {}
    if not isinstance(gates, dict):
        return " "
    verdicts = {
        str((gate or {}).get("verdict", "")).lower()
        for gate in gates.values() if isinstance(gate, dict)
    }
    return "\u26a0" if verdicts & {"fail", "flag"} else " "


def _coverage_cell(run: Run) -> str:
    coverage = run.coverage
    if not coverage:
        return "-"
    return f"{_pct(coverage.get('must_coverage'))}/{_pct(coverage.get('nice_coverage'))}"


def render_list(runs: Sequence[Run]) -> str:
    """The tracker table: newest first, one line per application."""
    if not runs:
        return "no runs recorded"
    rows = [
        (
            gate_mark(run),
            run.id,
            run.created_at[:10],
            f"{run.company}/{run.role}",
            run.status,
            _coverage_cell(run),
            run.out_dir or "-",
        )
        for run in runs
    ]
    headers = ("!", "id", "created", "company/role", "status", "must/nice", "out_dir")
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    def line(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    return "\n".join([
        line(headers),
        "  ".join("-" * width for width in widths),
        *(line(row) for row in rows),
    ])


def render_show(run: Run, notes: Sequence) -> str:
    """One run in full: what was selected, what it covered, what was overridden."""
    data = run.data
    lines = [
        f"{run.id}",
        f"  created     {run.created_at}",
        f"  company     {run.company}",
        f"  role        {run.role}",
        f"  status      {run.status}" + ("  (final)" if run.final else ""),
        f"  engine      {run.engine}",
        f"  out_dir     {run.out_dir or '-'}",
    ]
    if data.get("applied_at"):
        lines.append(f"  applied_at  {data['applied_at']}")
    if run.jd_hash:
        lines.append(f"  jd          {data.get('jd_path', '')} ({run.jd_hash[:12]})")
    if run.config_hash:
        lines.append(f"  config      {run.config_hash[:12]}")
    if run.pdf_sha256:
        lines.append(f"  pdf sha256  {run.pdf_sha256}")

    gates = data.get("gates") or {}
    blocking = [
        (name, gate) for name, gate in gates.items()
        if isinstance(gate, dict) and str(gate.get("verdict")).lower() in {"fail", "flag"}
    ]
    if data.get("jd_title"):
        lines.append(f"  target      {data['jd_title']}")
    if data.get("provider"):
        lines.append(f"  read by     {data['provider']}")
    if blocking:
        lines.append("")
        lines.append("gates")
        for name, gate in blocking:
            detail = gate.get("note") or gate.get("quote") or ""
            lines.append(f"  {name:<11} {gate.get('verdict')}  {detail}")

    lines.append("")
    lines.append("selection")
    lines.append(f"  roles       {', '.join(data.get('roles') or []) or '-'}")
    lines.append(f"  projects    {', '.join(data.get('projects') or []) or '-'}")
    lines.append(f"  certs       {', '.join(data.get('certifications') or []) or '-'}")
    for skill_line in data.get("skill_lines") or []:
        lines.append(f"  skills      {skill_line.get('category')}: {skill_line.get('skills')}")
    if data.get("budget") is not None:
        lines.append(
            f"  budget      {data['budget']} lines, {data.get('iterations', 0)} "
            f"fit iteration(s), {data.get('pages', '?')} page(s)"
        )

    check = data.get("check") or {}
    coverage = check.get("coverage") or {}
    if coverage:
        lines.append("")
        lines.append("coverage")
        lines.append(
            f"  must {_pct(coverage.get('must_coverage'))}  "
            f"nice {_pct(coverage.get('nice_coverage'))}  "
            f"context {_pct(coverage.get('context_rate'))}"
        )
        missing = [
            term["term"] for term in coverage.get("terms") or []
            if str(term.get("state", "")).startswith("missing")
        ]
        if missing:
            lines.append(f"  missing     {', '.join(missing)}")
    if check.get("parse"):
        failures = [c["check"] for c in check["parse"].get("checks", []) if not c["ok"]]
        lines.append(f"  parse       {'PASS' if check['parse'].get('ok') else 'FAIL'}"
                     + (f" ({', '.join(failures)})" if failures else ""))

    overrides = {
        key: value for key, value in (data.get("overrides") or {}).items()
        if _set(value)
    }
    if overrides:
        lines.append("")
        lines.append("overrides")
        width = max(len(key) for key in overrides)
        for key, value in overrides.items():
            lines.append(f"  {key.ljust(width)}  {_render_value(value)}")

    lines.append("")
    lines.append("notes")
    if not notes:
        lines.append("  (none)")
    for note in notes:
        lines.append(f"  {note.created_at[:16]}  {note.note}")
    return "\n".join(lines)


def _set(value: Any) -> bool:
    """Whether an override field says anything — ``{"add": [], "remove": []}`` does not."""
    if isinstance(value, dict):
        return any(_set(item) for item in value.values())
    return value not in (None, [], {}, "")


def _render_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={_render_value(v)}" for k, v in value.items() if _set(v))
    return str(value)


def days_since(timestamp: str, now: datetime | None = None) -> int:
    moment = datetime.fromisoformat(timestamp)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0, (( now or datetime.now(timezone.utc)) - moment).days)


def quiet_runs(
    store: Store, days: int = QUIET_DAYS, now: datetime | None = None
) -> list[tuple[Run, int]]:
    """Open applications with nothing said about them for ``days`` days.

    "Open" excludes both ends: a final status is closed, and a run still in
    ``drafted`` was never sent, so there is nobody to follow up with.
    """
    rows: list[tuple[Run, int]] = []
    for run in store.list_runs():
        if run.status == INITIAL_STATUS or run.status in FINAL_STATUSES:
            continue
        last = store.last_note_at(run.id) or run.data.get("applied_at") or run.created_at
        quiet = days_since(last, now)
        if quiet >= days:
            rows.append((run, quiet))
    return sorted(rows, key=lambda row: -row[1])


def render_quiet(rows: Sequence[tuple[Run, int]], days: int) -> str:
    if not rows:
        return f"no open application has been quiet for {days} days"
    lines = [f"{len(rows)} open application(s) quiet for {days}+ days:"]
    for run, quiet in rows:
        lines.append(
            f"  {run.id:<40} {run.company}/{run.role:<20} {run.status:<10} "
            f"{quiet} days quiet"
        )
    return "\n".join(lines)
