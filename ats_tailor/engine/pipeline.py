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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ats_match import Params, build_matrix, select
from ats_match.types import Constraints

from .. import config
from .adapt import (
    PAGE_TERMS_ATTR,
    VARIANT_ATTR,
    alias_map,
    claims_from_index,
    header_claims,
    pinned_skill_groups,
    render_context,
    section_rules,
    skill_claims,
)
from ..ats.check import (
    CompileError,
    compile_pdf,
    gap_terms,
    index_term_forms,
    latex_available,
    output_filename,
)
from .embedding import SentenceTransformerEmbedder, encode_store
from ..index.io import to_legacy
from ..jd.extract import extract_requirements, save_edited
from ..jd.gates import gate_dict
from ..llm.provider import NoneProvider
from ..index.models import Index
from ..runs.overrides import (
    Overrides,
    apply_requirements,
    metric_warnings,
    resolve as resolve_overrides,
    text_map,
)
from ..render.surface import align_labels, jd_forms
from .rerank import load_reranker
from ..render.latex import render_prompt, render_resume
from ..render.report import generate_requirement_visualization, generate_v2_report
from .results import RunResult
from .legacy.scoring import build_cert_text, chunk_text, compute_keyword_bonus
from .legacy.selection import estimate_line_count, select_certifications
from ..runs.store import Store

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
    #: ``ATS_RERANK`` mode when ``rerank`` is on: ``cross`` or ``llm``.
    rerank_mode: str | None = None
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
    #: Who reads the posting. ``None`` means nobody: the deterministic parser
    #: runs and the run needs no model, which is what a library caller and
    #: every test get unless they ask otherwise. The CLI is where a configured
    #: provider is resolved, because that is where the user asked for one.
    provider: Any = None
    #: Ignore any cached extraction for this posting and read it again.
    reextract: bool = False
    #: Append the posting's job title to the Summary (``ATS_TARGET_TITLE``).
    target_title: bool = config.TARGET_TITLE
    #: Languages the candidate declares, for the language gate.
    languages: Sequence[str] = ()
    #: Use the index's accepted phrasing variants for the posting's archetype.
    #: Off is what ``eval --no-variants`` measures against: the same run with
    #: every bullet in its own words.
    use_variants: bool = True


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
    # ── 1. posting → profile (gates first), index → claims ──
    provider = options.provider or NoneProvider()
    term_forms = index_term_forms(index)
    gaps = gap_terms(jd_text, index, extra_known=[company, role])
    jd_profile = extract_requirements(
        jd_text, index, provider, store,
        jd_path=options.jd_path, reextract=options.reextract,
        term_forms=term_forms, gaps=gaps, languages=options.languages,
        say=say,
    )
    # Gates come before anything is scored: a citizenship clause or a closed
    # posting decides whether the rest of this run is worth reading, and it is
    # the user's decision, so nothing here aborts.
    say("Gates")
    for line in jd_profile.gates.render().splitlines():
        say(f"  {line}")
    if jd_profile.title:
        say(f"  target title  {jd_profile.title}")
    say(f"  {jd_profile.summary_line()}")

    extracted = jd_profile.to_requirements()
    requirements = apply_requirements(extracted, overrides.requirements)
    if not overrides.requirements.empty():
        say(f"  Overrides: requirements {len(extracted)} → {len(requirements)} "
            f"({len(overrides.requirements.reweight)} reweighted)")
        # An edited requirement set is the user's reading of the posting, so it
        # is stored where it outranks any future extraction of the same text.
        save_edited(
            store, jd_text,
            jd_profile.model_copy(update={
                "requirements": [_requirement_spec(r) for r in requirements]}),
            jd_path=options.jd_path,
        )

    text_edits = text_map(overrides)
    for warning in metric_warnings(index, text_edits):
        say(f"  Warning: {warning}")
    archetype = jd_profile.archetype if options.use_variants else None
    claims = claims_from_index(index, text_overrides=text_edits, archetype=archetype)
    used_variants = {
        claim.id: str(claim.attrs.get(VARIANT_ATTR))
        for claim in claims if claim.attrs.get(VARIANT_ATTR)
    }
    if used_variants:
        say(f"  Variants: {len(used_variants)} bullet(s) phrased for "
            f"{archetype!r}: {', '.join(sorted(used_variants))}")
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
    reranker = _reranker(options, provider)
    matrix, claim_sim = build_matrix(
        requirements, claims, embedder, alias_map(index), reranker=reranker,
        top_k=config.RERANK_TOP_K,
        claim_vectors=claim_vectors, requirement_vectors=requirement_vectors,
    )

    # ── 3. the parts of the page the solver does not choose ──
    _skills, _projects, legacy_exp, legacy_certs, summary_data = to_legacy(index)
    education = legacy_exp["education"][:2]
    certs = legacy_certs["certifications"]
    selected_certs = select_certifications(certs, _cert_scores(embedder, jd_text, certs, store))
    summary_data = _summary_variant(summary_data, index, archetype, say)
    summary_data = _apply_summary(summary_data, overrides.summary)
    # The posting's own spelling, for the skills line only: bullets are the
    # user's sentences and are never rewritten here (Phase 5 owns variants).
    skill_labels = align_labels(index, jd_forms(jd_text, index))
    target_title = jd_profile.title if options.target_title else None

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
            selection, index, claims, text_overrides=text_edits,
            skill_labels=skill_labels, archetype=archetype)
        resume_tex = render_resume(
            company=company, role=role, profile=profile, education=education,
            experience=exp_ctx, projects=proj_ctx, skill_lines=skill_lines,
            certifications=selected_certs, summary=summary_data,
            target_title=target_title,
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
    variants = _variants_used(selection, used_variants, claims, store)

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
        profile=jd_profile, variants=variants,
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
        gates=gate_dict(jd_profile.gates),
        jd_title=jd_profile.title,
        archetype=jd_profile.archetype,
        provider=jd_profile.provider,
        provider_cached=jd_profile.cached,
        variants=variants,
    )
    (out_dir / "selection.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    say(f"\nDone! Output in: {out_dir}")
    return result


# ── helpers ───────────────────────────────────────────────────────────────────
def _reranker(options: V2Options, provider):
    """The reranker this run asked for (``ats_tailor.engine.rerank``), or ``None``."""
    if not options.rerank:
        return None
    mode = options.rerank_mode or (
        config.RERANK_MODE if config.RERANK_MODE != "none" else "cross")
    return load_reranker(mode, provider)


def _variants_used(selection, used_variants, claims, store) -> list[dict[str, Any]]:
    """The variants that reached the page, with the verdict that let them.

    The guardrail report is read back from the store rather than recomputed:
    what belongs in a run record is the decision a human actually accepted, not
    a re-judgement made against whatever the index says today.
    """
    if not used_variants:
        return []
    texts = {claim.id: claim.text for claim in claims}
    rows: list[dict[str, Any]] = []
    for claim_id in selection.claim_ids:
        archetype = used_variants.get(claim_id)
        if not archetype:
            continue
        row: dict[str, Any] = {
            "bullet_id": claim_id,
            "archetype": archetype,
            "text": texts.get(claim_id, ""),
        }
        try:
            accepted = store.accepted_variant(claim_id, archetype)
        except Exception:  # a store without the table is not a failed run
            accepted = None
        if accepted is not None:
            row["proposal_id"] = accepted.id
            row["guardrail"] = accepted.guardrail
        rows.append(row)
    return rows


def _summary_variant(summary_data: dict | None, index: Index, archetype, say):
    """``summaries.variants[archetype]`` when the index has one for it.

    ``archetype`` is ``None`` under ``--no-variants``, which turns this off
    with the bullet variants: since Phase 5 a summary variant is an accepted
    proposal like any other, and a flag that measured the page without the
    bullets' variants but with the summary's would be measuring neither state.
    """
    variants = getattr(index.summaries, "variants", {}) or {}
    if not summary_data or not archetype:
        return summary_data
    variant = variants.get(archetype)
    if not variant or not str(variant).strip():
        return summary_data
    say(f"  Summary: variant for {archetype!r}")
    return {**summary_data, "summary": str(variant).strip()}


def _requirement_spec(requirement):
    """One ``ats_match.Requirement`` back in the profile's own shape."""
    from ..jd.models import RequirementSpec

    return RequirementSpec(
        id=requirement.id, text=requirement.text, kind=requirement.kind,
        weight=requirement.weight, terms=list(requirement.terms),
        category=requirement.category,
    )


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
    if not options.compile_page_check or not latex_available():
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
