#!/usr/bin/env python3
"""ATS Resume Tailoring Tool.

Takes a job description, matches it against the YAML index using semantic
embeddings, selects the best content, and outputs a 1-page A4 LaTeX resume.

Usage:
    python -m ats_tailor.tailor --jd path/to/jd.txt --company acme --role backend
    python -m ats_tailor.tailor --profile profile.yaml --index index/ --jd jd.txt --company acme --role backend
"""

from sentence_transformers import SentenceTransformer
import argparse
import math
import re
import sys
from pathlib import Path

from .scoring import (
    build_cert_text,
    build_item_vectors,
    build_project_text,
    chunk_text,
    compute_jd_weights,
    compute_keyword_bonus,
    compute_tag_overlap_bonus,
    score_against_jd,
    score_against_jd_multi,
)
from . import config
from .config import (
    CERT_THRESHOLD, EMBED_MODEL, HF_TOKEN, LLM_EXPAND, LLM_MODEL, MAX_PAGE_LINES,
    MIN_EXP_BULLETS, MIN_EXPERIENCE, MIN_PROJECT_BULLETS, MIN_PROJECTS,
    RERANK, SECTION_PRIORITY,
)
from .selection import (
    estimate_line_count,
    select_certifications,
    select_experience,
    select_projects,
    select_skills,
)
from .rendering import (
    build_project_context,
    build_role_context,
    render_prompt,
    render_resume,
)
from .reporting import (
    generate_report,
    generate_visualization,
)
from .loaders import (
    build_cache_key,
    _text_hash,
    _yaml_hash,
    load_cached_embeddings,
    load_index,
    load_profile,
    parse_start_date,
    recency_multiplier,
    save_cached_embeddings,
)
from .llm import (
    detect_coverage_gaps,
    expand_jd_with_llm,
)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def _safe_slug(*parts):
    """Build a path-safe slug from company/role names."""
    raw = "-".join(str(p).strip().lower() for p in parts if str(p).strip())
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw)
    slug = re.sub(r"-+", "-", slug).strip(".-_")
    return slug or "resume"


def main():
    parser = argparse.ArgumentParser(
        description="Tailor a resume to a job description")
    raw_args = sys.argv[1:]
    ignore_config_parse_keys = set()
    if any(a == "--no-llm" or a == "--llm" or a.startswith("--llm=") for a in raw_args):
        ignore_config_parse_keys.add("ATS_LLM_EXPAND")
    if any(a in {"--rerank", "--no-rerank"} for a in raw_args):
        ignore_config_parse_keys.add("ATS_RERANK")
    parser.add_argument(
        "--jd", type=str, help="Path to job description text file")
    parser.add_argument("--company", required=False,
                        help="Company name (used in output path)")
    parser.add_argument("--role", required=False,
                        help="Role name (used in output path)")
    parser.add_argument(
        "--output", type=str, help="Output directory (default: output/{company}-{role})")
    parser.add_argument("--index", type=str, default="index",
                        help="Path to YAML index directory (default: index/)")
    parser.add_argument("--profile", type=str, default=None,
                        help="Path to profile.yaml (default: profile.yaml next to index dir)")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--llm", nargs="?", const=LLM_MODEL, default=argparse.SUPPRESS,
                           help="Expand JD keywords via LLM (env: ATS_LLM_EXPAND + ATS_LLM_MODEL)")
    llm_group.add_argument("--no-llm", action="store_true",
                           help="Disable LLM keyword expansion even if ATS_LLM_EXPAND is set")
    parser.add_argument("--model", type=str, default=EMBED_MODEL,
                        help=f"SentenceTransformer model name (env: ATS_EMBED_MODEL, default: {EMBED_MODEL})")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=RERANK,
                        help="Re-score top candidates with a cross-encoder (env: ATS_RERANK)")
    parser.add_argument("--llm-backend", choices=["auto", "mlx", "lmstudio", "ollama"],
                        default=config.LLM_BACKEND,
                        help="LLM backend (env: ATS_LLM_BACKEND, default: auto)")
    args = parser.parse_args()

    # Wire CLI --llm-backend to config so llm.py picks it up
    config.LLM_BACKEND = args.llm_backend

    if args.no_llm:
        args.llm = None
    elif not hasattr(args, "llm"):
        args.llm = LLM_MODEL if LLM_EXPAND else None

    try:
        config.validate_config(ignore_parse_keys=ignore_config_parse_keys)
    except ValueError as e:
        parser.error(str(e))

    if not args.company or not args.role:
        parser.error("--company and --role are required")

    # ── Resolve paths ──
    index_dir = Path(args.index).resolve()
    if args.profile:
        profile_path = Path(args.profile).resolve()
    else:
        profile_path = index_dir.parent / "profile.yaml"

    # ── Read JD ──
    if args.jd:
        jd_text = Path(args.jd).read_text()
    elif not sys.stdin.isatty():
        jd_text = sys.stdin.read()
    else:
        print("Error: provide --jd or pipe JD text via stdin", file=sys.stderr)
        sys.exit(1)

    jd_text = jd_text.strip()
    if not jd_text:
        print("Error: job description is empty", file=sys.stderr)
        sys.exit(1)
    jd_lower = jd_text.lower()

    # ── Output dir ──
    slug = _safe_slug(args.company, args.role)
    out_dir = Path(args.output) if args.output else Path("output") / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load profile ──
    print(f"Loading profile from {profile_path}...")
    profile = load_profile(profile_path)

    # ── Load index ──
    print(f"Loading YAML index from {index_dir}...")
    skills_data, projects_data, exp_data, certs_data, summary_data = load_index(
        index_dir)

    categories = skills_data["categories"]
    projects = projects_data["projects"]
    roles = exp_data["roles"]
    education = exp_data["education"][:2]
    certs = certs_data["certifications"]

    # ── Coverage gap detection (before LLM expansion, so gaps reflect original JD) ──
    gaps = detect_coverage_gaps(
        jd_text, skills_data, projects_data, exp_data, certs_data,
        extra_known=[args.company, args.role])
    if gaps:
        print(
            f"  Warning: {len(gaps)} JD term(s) not in index: {', '.join(gaps)}")

    # ── Optional LLM keyword expansion ──
    llm_terms = None
    if args.llm:
        backend_label = args.llm_backend if args.llm_backend != "auto" else "auto (MLX→LM Studio→Ollama)"
        print(f"Expanding JD keywords via {backend_label} ({args.llm})...")
        expanded = expand_jd_with_llm(
            jd_text, args.llm,
            categories=[cat["name"] for cat in categories])
        if expanded:
            llm_terms = [t.strip() for t in expanded.split("\n") if t.strip()]
            print(f"  LLM expanded terms: {', '.join(llm_terms)}")
            jd_text = jd_text + "\n" + expanded
            jd_lower = jd_text.lower()

    # ── Load model & embed ──
    print(f"Loading embedding model ({args.model})...")
    model = SentenceTransformer(args.model, token=HF_TOKEN)
    embed_dim = model.get_sentence_embedding_dimension()

    print("Embedding job description...")
    jd_chunks = chunk_text(jd_text)
    jd_weights = compute_jd_weights(jd_chunks)
    jd_cache_dir = out_dir
    jd_cache_key = build_cache_key("jd", args.model, _text_hash(jd_text), "jdv2")
    jd_embeddings = load_cached_embeddings(
        jd_cache_dir, jd_cache_key, embed_dim, expected_ndim=2, expected_rows=len(jd_chunks)
    )
    if jd_embeddings is None:
        jd_embeddings = model.encode(jd_chunks)
        save_cached_embeddings(jd_cache_dir, jd_cache_key, jd_embeddings)
    else:
        print("  (using cached JD embeddings)")

    # ── Score everything (hybrid: semantic + keyword + recency) ──
    print("Scoring candidates...")
    emb_cache_dir = index_dir / ".emb_cache"

    role_scores = []
    role_components = []
    for r in roles:
        cache_key = build_cache_key("role", args.model, _yaml_hash(r), "itemmv3")
        cached = load_cached_embeddings(
            emb_cache_dir, cache_key, embed_dim, expected_ndim=2, expected_min_rows=2
        )
        if cached is not None:
            r_embeds = cached
        else:
            r_embeds = build_item_vectors(r, "role", model)
            save_cached_embeddings(emb_cache_dir, cache_key, r_embeds)
        sem = score_against_jd_multi(jd_embeddings, r_embeds, jd_weights)
        kw = compute_keyword_bonus(r.get("ats_tags", []), jd_lower, jd_text)
        tag = compute_tag_overlap_bonus(
            r.get("ats_tags", []), jd_lower, jd_text)
        recency = recency_multiplier(r["period"])
        total = (sem + kw + tag) * recency
        role_scores.append(total)
        role_components.append({
            "semantic": sem,
            "keyword": kw,
            "tag": tag,
            "recency": recency,
            "evidence": 0.0,
            "rerank": 0.0,
            "total": total,
        })

    proj_scores = []
    proj_components = []
    for p in projects:
        cache_key = build_cache_key("proj", args.model, _yaml_hash(p), "itemmv3")
        cached = load_cached_embeddings(
            emb_cache_dir, cache_key, embed_dim, expected_ndim=2, expected_min_rows=2
        )
        if cached is not None:
            p_embeds = cached
        else:
            p_embeds = build_item_vectors(p, "project", model)
            save_cached_embeddings(emb_cache_dir, cache_key, p_embeds)
        sem = score_against_jd_multi(jd_embeddings, p_embeds, jd_weights)
        kw = compute_keyword_bonus(p.get("ats_tags", []), jd_lower, jd_text)
        tag = compute_tag_overlap_bonus(
            p.get("ats_tags", []), jd_lower, jd_text)
        recency = recency_multiplier(p["period"])
        total = (sem + kw + tag) * recency
        proj_scores.append(total)
        proj_components.append({
            "semantic": sem,
            "keyword": kw,
            "tag": tag,
            "recency": recency,
            "evidence": 0.0,
            "rerank": 0.0,
            "total": total,
        })

    cert_scores = []
    cert_components = []
    for c in certs:
        cache_key = build_cache_key("cert", args.model, _yaml_hash(c), "certv2")
        cached = load_cached_embeddings(emb_cache_dir, cache_key, embed_dim, expected_ndim=1)
        if cached is not None:
            c_emb = cached
        else:
            c_emb = model.encode(build_cert_text(c))
            save_cached_embeddings(emb_cache_dir, cache_key, c_emb)
        sem = score_against_jd(jd_embeddings, c_emb)
        kw = compute_keyword_bonus(
            c.get("ats_keywords", []), jd_lower, jd_text)
        total = sem + kw
        cert_scores.append(total)
        cert_components.append({
            "semantic": sem,
            "keyword": kw,
            "tag": 0.0,
            "recency": 1.0,
            "evidence": 0.0,
            "rerank": 0.0,
            "total": total,
        })

    # ── Select (skills first for evidence backlinks) ──
    print("Selecting best content...")
    skill_lines, all_skill_scores = select_skills(
        categories, jd_embeddings, model, jd_lower, jd_text)

    # ── Skill evidence backlinks ──
    evidence_bonus = {}
    for skill_name, _cat_name, score in all_skill_scores:
        if score < 0.3:
            continue
        for cat in categories:
            for skill in cat["skills"]:
                if skill["name"] == skill_name:
                    for ev in skill.get("evidence", []):
                        ref = ev.get("ref", "")
                        bonus = min(score * 0.05, 0.05)
                        evidence_bonus[ref] = evidence_bonus.get(
                            ref, 0) + bonus
    for i, p in enumerate(projects):
        if p["id"] in evidence_bonus:
            bonus = evidence_bonus[p["id"]]
            proj_scores[i] += bonus
            proj_components[i]["evidence"] += bonus
            proj_components[i]["total"] += bonus
    for i, r in enumerate(roles):
        if r["id"] in evidence_bonus:
            bonus = evidence_bonus[r["id"]]
            role_scores[i] += bonus
            role_components[i]["evidence"] += bonus
            role_components[i]["total"] += bonus

    sel_roles, sel_role_scores = select_experience(roles, role_scores)
    sel_projects, sel_proj_scores = select_projects(projects, proj_scores)
    sel_certs = select_certifications(certs, cert_scores)

    # ── Cross-encoder reranking ──
    if args.rerank:
        print("Cross-encoder reranking top candidates...")
        from sentence_transformers import CrossEncoder
        ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", token=HF_TOKEN)
        top_n = min(10, len(projects))
        ranked_indices = sorted(
            range(len(projects)), key=lambda i: proj_scores[i], reverse=True
        )[:top_n]
        ce_pairs = [(jd_text, build_project_text(projects[i]))
                    for i in ranked_indices]
        ce_scores_arr = ce_model.predict(ce_pairs)
        # Sigmoid-normalize logits to [0,1] before blending
        ce_norm = [1.0 / (1.0 + math.exp(-float(s))) for s in ce_scores_arr]
        for idx, ce_s in zip(ranked_indices, ce_norm):
            bonus = ce_s * 0.15
            proj_scores[idx] += bonus
            proj_components[idx]["rerank"] += bonus
            proj_components[idx]["total"] += bonus
        sel_projects, sel_proj_scores = select_projects(projects, proj_scores)

    # ── Build contexts ──
    exp_ctx = [build_role_context(r) for r in sel_roles]
    proj_ctx = [build_project_context(p) for p in sel_projects]

    # ── Budget enforcement (priority-aware: low-priority section trimmed/dropped first) ──
    line_est = estimate_line_count(exp_ctx, proj_ctx, summary_data, sel_certs)
    max_lines = MAX_PAGE_LINES

    # Build ordered passes: trim bullets, then drop items.
    # Low-priority section goes first so high-priority content is preserved.
    if SECTION_PRIORITY == "experience":
        bullet_passes = [
            ("proj", proj_ctx, MIN_PROJECT_BULLETS),
            ("exp",  exp_ctx,  MIN_EXP_BULLETS),
        ]
        drop_passes = [
            ("proj", proj_ctx, sel_projects, sel_proj_scores, MIN_PROJECTS),
            ("exp",  exp_ctx,  sel_roles,    sel_role_scores,  MIN_EXPERIENCE),
        ]
    else:
        bullet_passes = [
            ("exp",  exp_ctx,  MIN_EXP_BULLETS),
            ("proj", proj_ctx, MIN_PROJECT_BULLETS),
        ]
        drop_passes = [
            ("exp",  exp_ctx,  sel_roles,    sel_role_scores,  MIN_EXPERIENCE),
            ("proj", proj_ctx, sel_projects, sel_proj_scores, MIN_PROJECTS),
        ]

    # Passes 1-2: trim bullets to minimum
    for _label, ctx_list, min_bullets in bullet_passes:
        if line_est <= max_lines:
            break
        for item in ctx_list:
            if len(item["bullets"]) > min_bullets:
                item["bullets"] = item["bullets"][:min_bullets]
        line_est = estimate_line_count(exp_ctx, proj_ctx, summary_data, sel_certs)

    # Pass 2.5: trim certifications before dropping substantive content.
    while line_est > max_lines and len(sel_certs) > config.MIN_CERTIFICATIONS:
        sel_certs.pop()
        line_est = estimate_line_count(exp_ctx, proj_ctx, summary_data, sel_certs)

    # Passes 3-4: drop lowest-scoring items (lists are still score-ranked)
    for _label, ctx_list, sel_list, score_list, min_count in drop_passes:
        while line_est > max_lines and len(ctx_list) > min_count:
            ctx_list.pop()
            sel_list.pop()
            score_list.pop()
            line_est = estimate_line_count(exp_ctx, proj_ctx, summary_data, sel_certs)

    # ── Chronological reordering (most recent first) ──
    proj_order = sorted(range(len(proj_ctx)), key=lambda i: parse_start_date(proj_ctx[i]["period"]), reverse=True)
    proj_ctx = [proj_ctx[i] for i in proj_order]
    sel_projects = [sel_projects[i] for i in proj_order]
    sel_proj_scores = [sel_proj_scores[i] for i in proj_order]

    exp_order = sorted(range(len(exp_ctx)), key=lambda i: parse_start_date(exp_ctx[i]["period"]), reverse=True)
    exp_ctx = [exp_ctx[i] for i in exp_order]
    sel_roles = [sel_roles[i] for i in exp_order]
    sel_role_scores = [sel_role_scores[i] for i in exp_order]

    print(f"  Estimated lines: {line_est} (budget: {max_lines})")
    print(f"  Experience: {len(exp_ctx)} roles")
    print(f"  Projects: {len(proj_ctx)} projects")
    print(f"  Skill lines: {len(skill_lines)}")
    print(f"  Certifications: {len(sel_certs)}")

    # ── Bullet wrap warnings ──
    _LATEX_LINE_WIDTH = 114

    def _warn_bullet_wraps(contexts, label):
        for item in contexts:
            name = item.get("company") or item.get("name", "?")
            for bullet in item.get("bullets", []):
                if len(bullet) <= _LATEX_LINE_WIDTH:
                    continue
                wrap_at = bullet.rfind(" ", 0, _LATEX_LINE_WIDTH)
                if wrap_at == -1:
                    wrap_at = _LATEX_LINE_WIDTH
                second_line = bullet[wrap_at:].strip()
                if len(second_line.split()) <= 2:
                    print(f"  ! Wrap [{label}] {name}: {len(bullet)} chars, "
                          f"only {len(second_line.split())} word(s) on line 2: '{second_line}'")

    _warn_bullet_wraps(exp_ctx, "exp")
    _warn_bullet_wraps(proj_ctx, "proj")

    # ── Render ──
    print("Rendering LaTeX...")
    resume_tex = render_resume(
        company=args.company,
        role=args.role,
        profile=profile,
        education=education,
        experience=exp_ctx,
        projects=proj_ctx,
        skill_lines=skill_lines,
        certifications=sel_certs,
        summary=summary_data,
    )
    (out_dir / "resume.tex").write_text(resume_tex)

    print("Generating LLM prompt...")
    prompt_md = render_prompt(
        company=args.company,
        role=args.role,
        jd_text=jd_text,
        experience=exp_ctx,
        projects=proj_ctx,
    )
    (out_dir / "prompt.md").write_text(prompt_md)

    # ── Report ──
    print("Generating match report...")
    sel_role_ids = {r["id"] for r in sel_roles}
    exp_ranked = sorted(
        [(r, role_components[i], r["id"] in sel_role_ids)
         for i, r in enumerate(roles)],
        key=lambda x: x[1]["total"],
        reverse=True,
    )
    sel_proj_ids = {p["id"] for p in sel_projects}
    proj_ranked = sorted(
        [(p, proj_components[i], p["id"] in sel_proj_ids)
         for i, p in enumerate(projects)],
        key=lambda x: x[1]["total"],
        reverse=True,
    )
    sel_cert_ids = {c["id"] for c in sel_certs}
    cert_ranked = [
        (c, cert_components[i], c["id"] in sel_cert_ids)
        for i, c in sorted(
            enumerate(certs), key=lambda x: cert_components[x[0]]["total"], reverse=True
        )
    ]

    report = generate_report(args.company, args.role, exp_ranked,
                             proj_ranked, skill_lines, cert_ranked, gaps, llm_terms)
    (out_dir / "match_report.md").write_text(report)

    # ── Visualization ──
    print("Generating visualization...")
    viz_path = generate_visualization(
        out_dir, args.company, args.role,
        projects, proj_scores, sel_proj_ids,
        roles, role_scores, sel_role_ids,
        all_skill_scores,
        certs, cert_scores, sel_cert_ids, CERT_THRESHOLD,
    )

    print(f"\nDone! Output in: {out_dir}")
    print(f"  resume.tex       – compile with pdflatex")
    print(f"  prompt.md        – copy-paste into Claude/ChatGPT")
    print(f"  match_report.md  – similarity scores & rankings")
    print(f"  match_viz.png    – visual score breakdown")


if __name__ == "__main__":
    main()
