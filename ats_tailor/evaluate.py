"""Score the pipeline against the hand-written labels — the Phase 2 baseline.

Every scoring change after this is judged by re-running ``ats-tailor eval`` and
diffing the JSON: a config hash pins the run to its inputs (all ``ATS_*``
settings, the embedding model, the index content, the engine's git HEAD), so
two numbers are only ever compared when they mean the same thing.

Metrics per JD (``docs/v2-design.md`` §6):

``must_coverage`` / ``nice_coverage`` / ``context_rate``
    from :mod:`ats_tailor.atscheck` against the compiled PDF.
``parse_ok`` / ``pages``
    the structural check, and whether it still fits on one page.
``precision@k``
    of the top-``k`` selected items against the label's pins (``k`` = number
    of pins), for projects and roles separately.
``exclusion_violations``
    pinned-out items the run selected anyway — should always be 0.
``kendall_tau``
    rank correlation against the label's full preference order, over the items
    the run actually selected.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ats_match import kendall_tau, precision_at_k

from . import config
from .atscheck import (
    CompileError,
    auto_terms,
    compile_pdf,
    extract_text,
    index_term_forms,
    keyword_coverage,
    output_filename,
    parse_check,
    resolve_terms,
    split_regions,
    surface_form_set,
    write_reports,
)
from .labels import Label, load_labels

#: Columns of the per-JD table, in print order.
METRICS = (
    "must_coverage", "nice_coverage", "context_rate", "parse_ok", "pages",
    "precision_projects", "precision_roles", "exclusion_violations", "kendall_tau",
)


# ``precision_at_k`` and ``kendall_tau`` live in ``ats_match.evaluate``: they
# are pure ranking metrics with no notion of a resume, and the matching core
# needs them for its own tests. They are re-exported here so the eval harness
# and its callers keep one import site.


# ── Configuration fingerprint ─────────────────────────────────────────────────
def index_hash(index_dir: str | Path) -> str:
    """sha256 over every YAML in the index, path-ordered; caches excluded."""
    digest = hashlib.sha256()
    index_dir = Path(index_dir)
    for path in sorted(index_dir.rglob("*.yaml")):
        if any(part in {".emb_cache", ".schema"} for part in path.parts):
            continue
        digest.update(str(path.relative_to(index_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_head(repo: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def code_hash() -> str:
    """sha256 over the package's own sources and templates.

    ``git_head`` alone would call an edited working tree identical to the
    commit it sits on, which is exactly the case while a scoring change is
    being tried out — the moment the hash most needs to differ.
    """
    digest = hashlib.sha256()
    # Both distributions: a tuning change inside the matching core has to move
    # the fingerprint exactly as much as one in the application does.
    import ats_match

    for package in (Path(__file__).resolve().parent, Path(ats_match.__file__).parent):
        sources = sorted(
            path
            for pattern in ("*.py", "*.j2")
            for path in package.rglob(pattern)
            if "__pycache__" not in path.parts
        )
        for path in sources:
            digest.update(f"{package.name}/{path.relative_to(package)}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def config_fingerprint(
    index_dir: str | Path, engine: str = "v1"
) -> tuple[str, dict[str, Any]]:
    """``(hash, the values it was computed from)``.

    ``git_head`` is recorded but deliberately *not* hashed: committing the
    engine unchanged would otherwise rename every baseline and make the tracked
    numbers un-comparable to the code that produced them. ``code_hash`` already
    identifies the sources exactly, so the commit is provenance, not identity.
    """
    values = {
        "ats_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("ATS_")},
        "embed_model": config.EMBED_MODEL,
        "engine": engine,
        "index_hash": index_hash(index_dir),
        "code_hash": code_hash(),
    }
    payload = json.dumps(values, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return digest, {**values, "git_head": git_head(Path(__file__).resolve().parent.parent)}


# ── One run ───────────────────────────────────────────────────────────────────
def jd_candidates(label: Label, slug: str, jd_dir: str | Path | None) -> list[Path]:
    """Where a label's JD might live: its ``jd:`` field, then ``<jd_dir>/…``."""
    candidates = [Path(label.jd)]
    if jd_dir:
        candidates += [Path(jd_dir) / Path(label.jd).name, Path(jd_dir) / f"{slug}.txt"]
    seen: set[str] = set()
    return [c for c in candidates if not (str(c) in seen or seen.add(str(c)))]


def find_jd(label: Label, slug: str, jd_dir: str | Path | None) -> Path | None:
    """The label's JD file, or ``None`` when the posting is not on this machine."""
    return next((c for c in jd_candidates(label, slug, jd_dir) if c.is_file()), None)


def resolve_jd(label: Label, slug: str, jd_dir: str | Path | None) -> Path:
    """:func:`find_jd`, but raising when there is nothing to evaluate against."""
    found = find_jd(label, slug, jd_dir)
    if found is None:
        tried = ", ".join(str(c) for c in jd_candidates(label, slug, jd_dir))
        raise FileNotFoundError(f"{slug}: no JD file found (tried {tried})")
    return found


def evaluate_one(
    slug: str,
    label: Label,
    jd_path: Path,
    index_dir: Path,
    profile_path: Path,
    run_dir: Path,
    llm: bool = False,
    rerank: bool = False,
    engine: str = "v1",
) -> dict[str, Any]:
    """Run the pipeline for one label, compile, check, and score it."""
    from .tailor import load_index_for_terms, run as run_pipeline

    run_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--index", str(index_dir), "--profile", str(profile_path),
        "--jd", str(jd_path), "--company", label.company, "--role", label.role,
        "--output", str(run_dir),
        "--rerank" if rerank else "--no-rerank",
        "--engine", engine,
    ]
    argv.append("--llm" if llm else "--no-llm")

    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            result = run_pipeline(argv)
    except Exception as exc:  # a failed run is a data point, not a crash
        (run_dir / "run.log").write_text(log.getvalue())
        return {"slug": slug, "company": label.company, "role": label.role,
                "error": f"{type(exc).__name__}: {exc}", "parse_ok": False, "pages": 0}
    (run_dir / "run.log").write_text(log.getvalue())

    profile = _load_profile(profile_path)
    record: dict[str, Any] = {
        "slug": slug, "company": label.company, "role": label.role,
        "engine": engine,
        "selected_projects": result.project_ids, "selected_roles": result.role_ids,
    }
    if result.selection is not None:
        record["requirements"] = len(result.requirements)
        record["budget"] = result.budget
        record["fit_iterations"] = result.iterations

    # ── compile + parse ──
    try:
        compiled = compile_pdf(result.tex_path,
                               output_filename(profile.get("name", ""), label.company))
        text = extract_text(compiled.pdf_path)
        pages = compiled.pages
    except (CompileError, RuntimeError, FileNotFoundError) as exc:
        record.update(error=f"{type(exc).__name__}: {exc}", parse_ok=False, pages=0)
        return record

    parse_report = parse_check(text, profile, result.entries, pages)
    record["parse_ok"] = parse_report.ok
    record["pages"] = pages
    record["parse_failures"] = [c.check for c in parse_report.failures]

    # ── coverage ──
    index_obj = load_index_for_terms(index_dir)
    all_terms = index_term_forms(index_obj)
    must, nice = list(label.must_terms), list(label.nice_terms)
    if not must and not nice:
        must, nice = auto_terms(jd_path.read_text(), index_obj)
    terms = resolve_terms([*must, *nice], all_terms)
    coverage = keyword_coverage(
        text, terms, surface_form_set(all_terms), split_regions(text), must=must)
    record.update(
        must_coverage=coverage.must_coverage,
        nice_coverage=coverage.nice_coverage,
        context_rate=coverage.context_rate,
        gaps=[t.term for t in coverage.by_state("missing (gap)")],
        dropped=[t.term for t in coverage.by_state("missing (have it)")],
        synonym_only=[t.term for t in coverage.by_state("synonym-only")],
        stuffing=[{"term": t, "count": c} for t, c in coverage.stuffing],
    )

    # ── selection quality ──
    record["precision_projects"] = precision_at_k(result.project_ids, label.pin_projects)
    record["precision_roles"] = precision_at_k(result.role_ids, label.pin_roles)
    selected = set(result.project_ids) | set(result.role_ids)
    record["exclusion_violations"] = len(selected & set(label.exclude))
    taus = [
        tau for key, ranked in (("projects", result.project_ids), ("roles", result.role_ids))
        if (tau := kendall_tau(ranked, label.ranking.get(key, []))) is not None
    ]
    record["kendall_tau"] = sum(taus) / len(taus) if taus else None

    write_reports(run_dir, parse_report, coverage)
    return record


def _load_profile(profile_path: Path) -> dict:
    from .loaders import load_profile

    return load_profile(profile_path)


# ── The whole sweep ───────────────────────────────────────────────────────────
def evaluate(
    index_dir: str | Path,
    labels_dir: str | Path,
    jd_dir: str | Path | None,
    out_dir: str | Path,
    profile_path: str | Path | None = None,
    llm: bool = False,
    rerank: bool = False,
    only: Sequence[str] | None = None,
    engine: str = "v1",
) -> dict[str, Any]:
    index_dir = Path(index_dir).resolve()
    out_dir = Path(out_dir).resolve()
    profile = Path(profile_path).resolve() if profile_path else index_dir.parent / "profile.yaml"
    labels = load_labels(labels_dir)
    if only:
        labels = {slug: label for slug, label in labels.items() if slug in set(only)}
    if not labels:
        raise ValueError(f"no label files to evaluate in {labels_dir}")

    digest, values = config_fingerprint(index_dir, engine)
    runs = []
    skipped: list[str] = []
    for i, (slug, label) in enumerate(labels.items(), start=1):
        jd_path = find_jd(label, slug, jd_dir)
        if jd_path is None:
            # Labels are tracked but postings are not, so a label without its
            # JD on this machine is normal — skip it rather than fail the sweep.
            tried = ", ".join(str(c) for c in jd_candidates(label, slug, jd_dir))
            print(f"skipped {slug}: no JD file (tried {tried})", flush=True)
            skipped.append(slug)
            continue
        print(f"[{i}/{len(labels)}] {slug} ({label.company}/{label.role})", flush=True)
        runs.append(evaluate_one(
            slug, label, jd_path, index_dir, profile,
            out_dir / "runs" / slug, llm=llm, rerank=rerank, engine=engine,
        ))

    payload = {
        "config_hash": digest,
        "config": values,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "options": {"llm": llm, "rerank": rerank, "engine": engine},
        "runs": runs,
        "skipped": skipped,
        "means": means(runs),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{digest[:12]}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    payload["path"] = str(path)
    return payload


def means(runs: Sequence[dict]) -> dict[str, float | None]:
    """Mean of each metric over the runs that produced one."""
    result: dict[str, float | None] = {}
    for metric in METRICS:
        values = [
            float(run[metric]) for run in runs
            if run.get(metric) is not None and not isinstance(run.get(metric), str)
        ]
        result[metric] = sum(values) / len(values) if values else None
    return result


# ── Presentation ──────────────────────────────────────────────────────────────
_HEADERS = {
    "must_coverage": "must", "nice_coverage": "nice", "context_rate": "ctx",
    "parse_ok": "parse", "pages": "pg", "precision_projects": "P@k proj",
    "precision_roles": "P@k role", "exclusion_violations": "excl",
    "kendall_tau": "tau",
}


def _cell(metric: str, value: Any) -> str:
    if value is None:
        return "—"
    if metric in {"must_coverage", "nice_coverage", "context_rate"}:
        return f"{value:.0%}"
    if metric in {"precision_projects", "precision_roles", "kendall_tau"}:
        return f"{value:+.2f}" if metric == "kendall_tau" else f"{value:.2f}"
    if metric == "parse_ok":
        # A single run is a boolean; the mean over runs is a pass rate.
        return ("ok" if value else "FAIL") if isinstance(value, bool) else f"{value:.0%}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_table(payload: dict[str, Any]) -> str:
    """The per-JD table plus the means row."""
    rows = []
    for run in payload["runs"]:
        cells = [run["slug"]] + [_cell(m, run.get(m)) for m in METRICS]
        if run.get("error"):
            cells[-1] = "error"
        rows.append(cells)
    mean_row = ["mean"] + [_cell(m, payload["means"].get(m)) for m in METRICS]
    header = ["jd"] + [_HEADERS[m] for m in METRICS]

    widths = [max(len(row[i]) for row in [header, *rows, mean_row]) for i in range(len(header))]
    def line(cells):
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths)))

    out = [line(header), "  ".join("-" * w for w in widths)]
    out += [line(row) for row in rows]
    out += ["  ".join("-" * w for w in widths), line(mean_row)]
    errors = [f"  {run['slug']}: {run['error']}" for run in payload["runs"] if run.get("error")]
    if errors:
        out += ["", "errors:", *errors]
    if payload.get("skipped"):
        out += ["", f"skipped: {', '.join(payload['skipped'])} (no JD file on this machine)"]
    return "\n".join(out)


def compare(a_path: str | Path, b_path: str | Path) -> str:
    """Per-metric delta table between two eval JSON files."""
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    rows = [("metric", Path(a_path).stem, Path(b_path).stem, "delta")]
    for metric in METRICS:
        left, right = a["means"].get(metric), b["means"].get(metric)
        delta = "—" if left is None or right is None else f"{right - left:+.3f}"
        rows.append((metric, _cell(metric, left), _cell(metric, right), delta))

    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    out = ["  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                     for i, (c, w) in enumerate(zip(rows[0], widths))),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                      for i, (c, w) in enumerate(zip(row, widths)))
            for row in rows[1:]]
    if a.get("config_hash") == b.get("config_hash"):
        out.append("\nsame config hash — identical inputs and settings.")
    return "\n".join(out)
