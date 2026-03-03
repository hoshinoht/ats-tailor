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
from .config import EMBED_MODEL, LLM_MODEL, MAX_PAGE_LINES, MIN_PROJECTS, RERANK
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
    _text_hash,
    _yaml_hash,
    load_cached_embeddings,
    load_index,
    load_profile,
    recency_multiplier,
    save_cached_embeddings,
)
from .llm import (
    detect_coverage_gaps,
    expand_jd_with_llm,
)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tailor a resume to a job description")
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
    parser.add_argument("--llm", nargs="?", const=LLM_MODEL or "qwen3.5:4b",
                        default=LLM_MODEL,
                        help="Expand JD keywords via ollama LLM (env: ATS_LLM_MODEL)")
    parser.add_argument("--model", type=str, default=EMBED_MODEL,
                        help=f"SentenceTransformer model name (env: ATS_EMBED_MODEL, default: {EMBED_MODEL})")
    parser.add_argument("--rerank", action="store_true", default=RERANK,
                        help="Re-score top candidates with a cross-encoder (env: ATS_RERANK)")
    args = parser.parse_args()

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
    slug = f"{args.company}-{args.role}".lower().replace(" ", "-")
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
        print(f"Expanding JD keywords via ollama ({args.llm})...")
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
    model = SentenceTransformer(args.model)

    print("Embedding job description...")
    jd_chunks = chunk_text(jd_text)
    jd_weights = compute_jd_weights(jd_chunks)
    jd_cache_dir = out_dir
    jd_cache_key = f"jd_{_text_hash(jd_text)[:16]}"
    jd_embeddings = load_cached_embeddings(jd_cache_dir, jd_cache_key)
    if jd_embeddings is None:
        jd_embeddings = model.encode(jd_chunks)
        save_cached_embeddings(jd_cache_dir, jd_cache_key, jd_embeddings)
    else:
        print("  (using cached JD embeddings)")

    # ── Score everything (hybrid: semantic + keyword + recency) ──
    print("Scoring candidates...")
    emb_cache_dir = index_dir / ".emb_cache"

    role_scores = []
    for r in roles:
        cache_key = f"role_{r['id']}_{_yaml_hash(r)[:16]}_mv2"
        cached = load_cached_embeddings(emb_cache_dir, cache_key)
        if cached is not None:
            r_embeds = cached
        else:
            r_embeds = build_item_vectors(r, "role", model)
            save_cached_embeddings(emb_cache_dir, cache_key, r_embeds)
        sem = score_against_jd_multi(jd_embeddings, r_embeds, jd_weights)
        kw = compute_keyword_bonus(r.get("ats_tags", []), jd_lower, jd_text)
        tag = compute_tag_overlap_bonus(
            r.get("ats_tags", []), jd_lower, jd_text)
        role_scores.append((sem + kw + tag) * recency_multiplier(r["period"]))

    proj_scores = []
    for p in projects:
        cache_key = f"proj_{p['id']}_{_yaml_hash(p)[:16]}_mv2"
        cached = load_cached_embeddings(emb_cache_dir, cache_key)
        if cached is not None:
            p_embeds = cached
        else:
            p_embeds = build_item_vectors(p, "project", model)
            save_cached_embeddings(emb_cache_dir, cache_key, p_embeds)
        sem = score_against_jd_multi(jd_embeddings, p_embeds, jd_weights)
        kw = compute_keyword_bonus(p.get("ats_tags", []), jd_lower, jd_text)
        tag = compute_tag_overlap_bonus(
            p.get("ats_tags", []), jd_lower, jd_text)
        proj_scores.append((sem + kw + tag) * recency_multiplier(p["period"]))

    cert_scores = []
    for c in certs:
        cache_key = f"cert_{c['id']}_{_yaml_hash(c)[:16]}"
        cached = load_cached_embeddings(emb_cache_dir, cache_key)
        if cached is not None:
            c_emb = cached
        else:
            c_emb = model.encode(build_cert_text(c))
            save_cached_embeddings(emb_cache_dir, cache_key, c_emb)
        sem = score_against_jd(jd_embeddings, c_emb)
        kw = compute_keyword_bonus(
            c.get("ats_keywords", []), jd_lower, jd_text)
        cert_scores.append(sem + kw)

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
            proj_scores[i] += evidence_bonus[p["id"]]
    for i, r in enumerate(roles):
        if r["id"] in evidence_bonus:
            role_scores[i] += evidence_bonus[r["id"]]

    sel_roles, sel_role_scores = select_experience(roles, role_scores)
    sel_projects, sel_proj_scores = select_projects(projects, proj_scores)
    sel_certs = select_certifications(certs, cert_scores)

    # ── Cross-encoder reranking ──
    if args.rerank:
        print("Cross-encoder reranking top candidates...")
        from sentence_transformers import CrossEncoder
        ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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
            proj_scores[idx] += ce_s * 0.15
        sel_projects, sel_proj_scores = select_projects(projects, proj_scores)

    # ── Build contexts ──
    exp_ctx = [build_role_context(r) for r in sel_roles]
    proj_ctx = [build_project_context(p) for p in sel_projects]

    # ── Budget check (lax: trim bullets first, drop projects only as last resort) ──
    line_est = estimate_line_count(exp_ctx, proj_ctx, summary_data)
    max_lines = MAX_PAGE_LINES

    # Pass 1: trim to 2 bullets per project
    if line_est > max_lines:
        for p in proj_ctx:
            if len(p["bullets"]) > 2:
                p["bullets"] = p["bullets"][:2]
        line_est = estimate_line_count(exp_ctx, proj_ctx, summary_data)

    # Pass 2: drop lowest-scoring projects if still over
    while line_est > max_lines and len(proj_ctx) > MIN_PROJECTS:
        proj_ctx.pop()
        sel_projects.pop()
        sel_proj_scores.pop()
        line_est = estimate_line_count(exp_ctx, proj_ctx, summary_data)

    print(f"  Estimated lines: {line_est} (budget: {max_lines})")
    print(f"  Experience: {len(exp_ctx)} roles")
    print(f"  Projects: {len(proj_ctx)} projects")
    print(f"  Skill lines: {len(skill_lines)}")
    print(f"  Certifications: {len(sel_certs)}")

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
    exp_ranked = sorted(zip(roles, role_scores),
                        key=lambda x: x[1], reverse=True)
    sel_proj_ids = {p["id"] for p in sel_projects}
    proj_ranked = sorted(
        [(p, s, p["id"] in sel_proj_ids)
         for p, s in zip(projects, proj_scores)],
        key=lambda x: x[1],
        reverse=True,
    )
    sel_cert_ids = {c["id"] for c in sel_certs}
    cert_ranked = [
        (c, s, c["id"] in sel_cert_ids)
        for c, s in sorted(zip(certs, cert_scores), key=lambda x: x[1], reverse=True)
    ]

    report = generate_report(args.company, args.role, exp_ranked,
                             proj_ranked, skill_lines, cert_ranked, gaps, llm_terms)
    (out_dir / "match_report.md").write_text(report)

    # ── Visualization ──
    print("Generating visualization...")
    sel_role_ids = {r["id"] for r in sel_roles}
    viz_path = generate_visualization(
        out_dir, args.company, args.role,
        projects, proj_scores, sel_proj_ids,
        roles, role_scores, sel_role_ids,
        all_skill_scores,
        certs, cert_scores, sel_cert_ids,
    )

    print(f"\nDone! Output in: {out_dir}")
    print(f"  resume.tex       – compile with pdflatex")
    print(f"  prompt.md        – copy-paste into Claude/ChatGPT")
    print(f"  match_report.md  – similarity scores & rankings")
    print(f"  match_viz.png    – visual score breakdown")


if __name__ == "__main__":
    main()
