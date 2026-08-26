"""The v2 pipeline: requirements in, one page of the best-covering claims out.

Where v1 ranks whole items against the posting and then trims, v2 parses the
posting into weighted requirements, treats every bullet and every skill as an
independently selectable claim, and asks the solver for the subset that covers
the most requirement weight inside a line budget. The page-fit loop closes over
the real thing: render, compile, count pages, shrink the budget, re-solve.

``ats_match`` does the selecting and knows nothing about any of this;
``adapt.py`` is the seam.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from ats_match import Params, build_matrix, select
from ats_match.types import Constraints

from . import config
from .adapt import (
    PAGE_TERMS_ATTR,
    alias_map,
    claims_from_index,
    header_claims,
    pinned_skill_groups,
    render_context,
    section_rules,
    skill_claims,
)
from .atscheck import CompileError, compile_pdf, gap_terms, index_term_forms, output_filename
from .embedding import SentenceTransformerEmbedder, encode_store
from .index_io import to_legacy
from .jd.hashing import jd_hash
from .jd.requirements import requirements_from_jd
from .models import Index
from .overrides import (
    Overrides,
    apply_requirements,
    metric_warnings,
    resolve as resolve_overrides,
    text_map,
)
from .rendering import render_prompt, render_resume
from .reporting import generate_requirement_visualization, generate_v2_report
from .results import RunResult
from .scoring import build_cert_text, chunk_text, compute_keyword_bonus
from .selection import estimate_line_count, select_certifications
from .store import Store

#: How much of the page a run gives back when the PDF spills onto page two.
BUDGET_STEP = 4
MAX_ITERATIONS = 3
#: Requirements below this coverage are reported as gaps rather than hits.
GAP_THRESHOLD = 0.3


def default_params() -> Params:
    """The objective knobs, from the ``ATS_TAU`` / ``ATS_MU`` / ``ATS_BETA`` settings."""
    return Params(
        tau=config.SELECT_TAU,
        mu=config.SELECT_MU,
        beta=config.SELECT_BETA,
        redundancy_in_context=config.SELECT_REDUNDANCY_IN_SKILLS,
        breadth_first=config.SELECT_BREADTH_FIRST,
        context_terms_attr=PAGE_TERMS_ATTR,
    )


@dataclass
class V2Options:
    """Everything the v2 run needs that is not content."""

    model: str = config.EMBED_MODEL
    index_dir: Path | None = None
    rerank: bool = False
    compile_page_check: bool = True
    max_page_lines: int = config.MAX_PAGE_LINES
    max_iterations: int = MAX_ITERATIONS
    budget_step: int = BUDGET_STEP
    params: Params = field(default_factory=lambda: default_params())
    write_prompt: bool = True
    quiet: bool = False
    #: Per-application pins, exclusions, phrasing and requirement edits.
    overrides: Overrides = field(default_factory=Overrides)
    #: The runtime store backing the embedding cache and the JD profile. Left
    #: unset, the run opens (and closes) the one belonging to ``index_dir``.
    store: Store | None = None
    #: Where the posting came from, recorded with the JD profile.
    jd_path: str | None = None


def tailor_v2(
    jd_text: str,
    index: Index,
    profile: dict,
    company: str,
    role: str,
    out_dir: str | Path,
    options: V2Options | None = None,
) -> RunResult:
    """Run the v2 engine end to end and write the run's artefacts."""
    options = options or V2Options()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = (lambda *_a, **_k: None) if options.quiet else print
    overrides = options.overrides or Overrides()
    store = options.store or Store.open(options.index_dir)
    owns_store = options.store is None

    try:
        return _tailor_v2(jd_text, index, profile, company, role, out_dir,
                          options, overrides, store, say)
    finally:
        if owns_store:
            store.close()


def _tailor_v2(jd_text, index, profile, company, role, out_dir, options,
               overrides, store, say) -> RunResult:
    # ── 1. posting → requirements, index → claims ──
    term_forms = index_term_forms(index)
    gaps = gap_terms(jd_text, index, extra_known=[company, role])
    extracted = requirements_from_jd(jd_text, term_forms, gaps)
    requirements = apply_requirements(extracted, overrides.requirements)
    if not overrides.requirements.empty():
        say(f"  Overrides: requirements {len(extracted)} → {len(requirements)} "
            f"({len(overrides.requirements.reweight)} reweighted)")
    # ``extracted`` is what the parser said; ``edited`` is what this run used,
    # recorded separately so 3c's LLM extractor can be judged against the
    # deterministic one without a human edit being mistaken for either.
    store.put_jd_profile(
        jd_hash(jd_text), jd_text, provider="fallback",
        source_path=options.jd_path,
        extracted=[_requirement_dict(r) for r in extracted],
        edited=([_requirement_dict(r) for r in requirements]
                if not overrides.requirements.empty() else None),
    )

    text_edits = text_map(overrides)
    for warning in metric_warnings(index, text_edits):
        say(f"  Warning: {warning}")
    claims = claims_from_index(index, text_overrides=text_edits)
    if config.HEADER_CLAIMS:
        claims += header_claims(index)
    claims += skill_claims(index, requirements)
    say(f"Parsed {len(requirements)} requirement(s) "
        f"({sum(1 for r in requirements if r.kind == 'must')} must) "
        f"and {len(claims)} claim(s)")
    if gaps:
        say(f"  Warning: {len(gaps)} JD term(s) not in index: {', '.join(gaps)}")

    resolved = resolve_overrides(overrides, index)
    for warning in resolved.warnings():
        say(f"  Warning: {warning}")
    if not overrides.empty():
        say(f"  Overrides: {overrides.summary_line()}")

    # ── 2. the score matrix ──
    embedder = SentenceTransformerEmbedder(options.model)
    claim_vectors = encode_store(
        embedder, [c.text for c in claims], store,
        ["skill" if c.section == "skill" else "claim" for c in claims],
        embedder.passage_prefix,
    )
    requirement_vectors = encode_store(
        embedder, [r.text for r in requirements], store, "req", embedder.query_prefix
    )
    reranker = _load_reranker() if options.rerank else None
    matrix, claim_sim = build_matrix(
        requirements, claims, embedder, alias_map(index), reranker=reranker,
        claim_vectors=claim_vectors, requirement_vectors=requirement_vectors,
    )

    # ── 3. the parts of the page the solver does not choose ──
    _skills, _projects, legacy_exp, legacy_certs, summary_data = to_legacy(index)
    education = legacy_exp["education"][:2]
    certs = legacy_certs["certifications"]
    selected_certs = select_certifications(certs, _cert_scores(embedder, jd_text, certs, store))
    summary_data = _apply_summary(summary_data, overrides.summary)

    fixed_lines = estimate_line_count([], [], summary_data, selected_certs)
    # The fixed estimate already reserves a flat block for the skills section;
    # the solver pays for each skills line through its group cost, so give that
    # part of the block back rather than charging for it twice.
    budget = options.max_page_lines - (fixed_lines - config.MAX_SKILL_LINES)
    if overrides.budget_lines is not None:
        budget = int(overrides.budget_lines)
        say(f"  Fixed lines: {fixed_lines}; claim budget: {budget} (overridden)")
    else:
        say(f"  Fixed lines: {fixed_lines}; claim budget: {budget}")

    # ── 4-5. select, render, and shrink until it fits on one page ──
    rules = section_rules()
    pinned = pinned_skill_groups() & {c.group for c in claims if c.section == "skill"}
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for iteration in range(max(1, options.max_iterations)):
        selection = select(
            matrix, requirements, claims,
            Constraints(
                budget=budget,
                sections=rules,
                pin_groups=set(pinned) | resolved.pin_groups,
                pin_claims=set(resolved.pin_claims),
                exclude_groups=set(resolved.exclude_groups),
                exclude_claims=set(resolved.exclude_claims),
            ),
            options.params, claim_sim=claim_sim,
        )
        exp_ctx, proj_ctx, skill_lines = render_context(
            selection, index, claims, text_overrides=text_edits)
        resume_tex = render_resume(
            company=company, role=role, profile=profile, education=education,
            experience=exp_ctx, projects=proj_ctx, skill_lines=skill_lines,
            certifications=selected_certs, summary=summary_data,
        )
        tex_path = out_dir / "resume.tex"
        tex_path.write_text(resume_tex)
        pages = _page_count(tex_path, profile, company, options, say)
        attempt = {
            "budget": budget, "pages": pages, "selection": selection,
            "exp": exp_ctx, "proj": proj_ctx, "skills": skill_lines,
            "tex": resume_tex, "used": selection.budget_used,
        }
        attempts.append(attempt)
        say(f"  Budget {budget}: {selection.budget_used:.0f} lines used, "
            f"{len(exp_ctx)} roles, {len(proj_ctx)} projects, "
            f"{len(skill_lines)} skill lines, pages={pages if pages else '?'}")
        if pages is None or pages <= 1:
            best = attempt
            break
        best = attempt
        budget -= options.budget_step

    fitting = [a for a in attempts if a["pages"] == 1]
    chosen = fitting[-1] if fitting else min(
        attempts, key=lambda a: (a["pages"] or 99, -a["budget"])
    )
    if chosen is not best:
        # The last render on disk is not the one being kept: put it back.
        (out_dir / "resume.tex").write_text(chosen["tex"])
        _page_count(out_dir / "resume.tex", profile, company, options, say)
    if not fitting and any(a["pages"] for a in attempts):
        say(f"  ! No budget produced a single page; keeping the smallest "
            f"({chosen['pages']} pages)")

    selection = chosen["selection"]
    exp_ctx, proj_ctx, skill_lines = chosen["exp"], chosen["proj"], chosen["skills"]

    # ── 6. artefacts ──
    if options.write_prompt:
        (out_dir / "prompt.md").write_text(
            render_prompt(company=company, role=role, jd_text=jd_text,
                          experience=exp_ctx, projects=proj_ctx)
        )
    (out_dir / "match_report.md").write_text(generate_v2_report(
        company, role, requirements, claims, selection, matrix, skill_lines,
        certs=certs, selected_certs=selected_certs, gaps=gaps,
        budget=chosen["budget"], pages=chosen["pages"], attempts=len(attempts),
    ))
    generate_requirement_visualization(
        out_dir, company, role, requirements, selection.coverage, GAP_THRESHOLD
    )

    result = RunResult(
        out_dir=out_dir,
        tex_path=out_dir / "resume.tex",
        company=company,
        role=role,
        role_ids=_ranked_groups(selection, claims, "role"),
        project_ids=_ranked_groups(selection, claims, "project"),
        cert_ids=[c["id"] for c in selected_certs],
        skill_lines=[dict(line) for line in skill_lines],
        entries=(
            [{"kind": "education", "title": f'{e["degree"]} in {e["field"]}',
              "organisation": e["institution"], "period": e["period"]} for e in education]
            + [{"kind": "role", "id": ctx["id"], "title": ctx["title"],
                "organisation": ctx["company"], "period": ctx["period"]} for ctx in exp_ctx]
            + [{"kind": "project", "id": ctx["id"], "title": ctx["name"],
                "period": ctx["period"]} for ctx in proj_ctx]
        ),
        engine="v2",
        selection=selection.to_dict(),
        requirements=[_requirement_dict(r) for r in requirements],
        coverage={k: round(float(v), 6) for k, v in selection.coverage.items()},
        pages=chosen["pages"],
        budget=chosen["budget"],
        iterations=len(attempts),
    )
    (out_dir / "selection.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    say(f"\nDone! Output in: {out_dir}")
    return result


# ── helpers ───────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_reranker():
    """A cross-encoder wrapped to satisfy ``ats_match.Reranker`` (opt-in).

    Cached per process for the same reason the embedder is: ``eval`` runs the
    pipeline a dozen times and re-loading the weights each time would dominate
    any measurement of what the reranker actually costs.
    """
    from sentence_transformers import CrossEncoder

    class _CrossEncoderReranker:
        def __init__(self):
            self.model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", token=config.HF_TOKEN
            )

        def score(self, pairs):
            return [float(s) for s in self.model.predict(list(pairs))]

    return _CrossEncoderReranker()


def _requirement_dict(requirement) -> dict[str, Any]:
    return {
        "id": requirement.id, "kind": requirement.kind,
        "weight": requirement.weight, "terms": list(requirement.terms),
        "text": requirement.text,
    }


def _apply_summary(summary_data: dict | None, replacement: str | None) -> dict | None:
    """The overridden summary text, or no Summary section when it is emptied."""
    if replacement is None:
        return summary_data
    if not replacement.strip():
        return None
    return {**(summary_data or {"highlights": []}), "summary": replacement.strip()}


def _cert_scores(embedder, jd_text: str, certs: Sequence[dict], store: Store) -> list[float]:
    """v1's certification scoring, unchanged: best JD-chunk cosine plus keywords."""
    if not certs:
        return []
    chunks = chunk_text(jd_text)
    jd_vectors = embedder.encode(chunks)
    cert_vectors = encode_store(
        embedder, [build_cert_text(c) for c in certs], store, "cert"
    )
    similarity = cert_vectors @ jd_vectors.T
    jd_lower = jd_text.lower()
    return [
        float(similarity[i].max())
        + compute_keyword_bonus(cert.get("ats_keywords", []), jd_lower, jd_text)
        for i, cert in enumerate(certs)
    ]


def _page_count(tex_path: Path, profile: dict, company: str, options: V2Options, say) -> int | None:
    """Compile and count pages, or ``None`` when there is no LaTeX to compile with."""
    if not options.compile_page_check or not shutil.which("latexmk"):
        return None
    try:
        compiled = compile_pdf(
            tex_path, output_filename(str(profile.get("name", "")), company)
        )
    except (CompileError, OSError) as exc:
        say(f"  ! compile failed: {exc}")
        return None
    return compiled.pages


def _ranked_groups(selection, claims, section: str) -> list[str]:
    """Group ids in the order the solver first reached for them."""
    section_of = {c.id: (c.section, c.group) for c in claims}
    ordered: list[str] = []
    for step in selection.trace:
        entry = section_of.get(step.claim_id)
        if entry and entry[0] == section and entry[1] not in ordered:
            ordered.append(entry[1])
    return ordered
