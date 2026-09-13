"""Match report markdown and matplotlib visualization."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CAT_COLORS = {
    "Programming Languages": "#3B82F6",
    "Machine Learning & AI": "#8B5CF6",
    "Cloud & Infrastructure": "#F59E0B",
    "Observability & Monitoring": "#EF4444",
    "Architecture & Protocols": "#10B981",
    "IoT & Embedded Systems": "#EC4899",
    "Databases": "#06B6D4",
    "Mobile Development": "#F97316",
    "Frontend & Web": "#84CC16",
    "Game Development": "#6366F1",
    "DevOps & Tooling": "#14B8A6",
    "Testing & QA": "#A855F7",
}
SEL_COLOR = "#2563EB"
UNSEL_COLOR = "#D1D5DB"
THRESH_COLOR = "#EF4444"


def _md_cell(value):
    """Escape a value for a Markdown table cell."""
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", r"\|")


def _fmt(component, key):
    return f"{float(component.get(key, 0.0)):.3f}"


def generate_report(company, role, exp_ranked, proj_ranked, skill_lines, cert_ranked, gaps=None, llm_terms=None):
    """Generate a markdown match report."""
    lines = [
        f"# Match Report: {company} – {role}",
        "",
        "## Experience Rankings",
        "",
        "| Rank | Selected | Role | Company | Total | Semantic | Keyword | Tags | Recency | Evidence |",
        "|------|----------|------|---------|-------|----------|---------|------|---------|----------|",
    ]
    for i, (r, comp, sel) in enumerate(exp_ranked, 1):
        mark = "Y" if sel else ""
        lines.append(
            f"| {i} | {mark} | {_md_cell(r['title'])} | {_md_cell(r['company'])} | "
            f"{_fmt(comp, 'total')} | {_fmt(comp, 'semantic')} | {_fmt(comp, 'keyword')} | "
            f"{_fmt(comp, 'tag')} | {float(comp.get('recency', 1.0)):.2f}x | {_fmt(comp, 'evidence')} |"
        )

    lines += [
        "",
        "## Project Rankings",
        "",
        "| Rank | Selected | Project | Total | Semantic | Keyword | Tags | Recency | Evidence | Rerank |",
        "|------|----------|---------|-------|----------|---------|------|---------|----------|--------|",
    ]
    for i, (p, comp, sel) in enumerate(proj_ranked, 1):
        mark = "Y" if sel else ""
        lines.append(
            f"| {i} | {mark} | {_md_cell(p['name'])} | {_fmt(comp, 'total')} | "
            f"{_fmt(comp, 'semantic')} | {_fmt(comp, 'keyword')} | {_fmt(comp, 'tag')} | "
            f"{float(comp.get('recency', 1.0)):.2f}x | {_fmt(comp, 'evidence')} | {_fmt(comp, 'rerank')} |"
        )

    lines += [
        "",
        "## Selected Skills",
        "",
    ]
    for sl in skill_lines:
        lines.append(f"- **{sl['category']}:** {sl['skills']}")

    if cert_ranked:
        lines += [
            "",
            "## Certification Rankings",
            "",
            "| Selected | Cert | Issuer | Total | Semantic | Keyword |",
            "|----------|------|--------|-------|----------|---------|",
        ]
        for c, comp, sel in cert_ranked:
            mark = "Y" if sel else ""
            lines.append(
                f"| {mark} | {_md_cell(c['name'])} | {_md_cell(c['issuer'])} | "
                f"{_fmt(comp, 'total')} | {_fmt(comp, 'semantic')} | {_fmt(comp, 'keyword')} |"
            )

    if gaps:
        lines += [
            "",
            "## Coverage Gaps",
            "",
            "The following tech-looking terms appear in the JD but are not in any index file.",
            "Consider adding them to `index/skills.yaml` or relevant project entries.",
            "",
        ]
        for g in gaps:
            lines.append(f"- {g}")

    if llm_terms:
        lines += [
            "",
            "## LLM-Expanded Keywords",
            "",
            "The following terms were inferred from the JD by a local LLM and injected",
            "into the keyword matching pool to improve tag overlap scoring.",
            "",
            ", ".join(llm_terms),
        ]

    lines.append("")
    return "\n".join(lines)


def generate_visualization(
    out_dir, company, role,
    projects, proj_scores, sel_proj_ids,
    roles, role_scores, sel_role_ids,
    all_skill_scores,
    certs, cert_scores, sel_cert_ids, cert_threshold=0.25,
):
    """Generate a 4-panel match visualization PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Match Analysis: {company} – {role}",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.subplots_adjust(hspace=0.35, wspace=0.45, top=0.93,
                        bottom=0.04, left=0.18, right=0.96)

    # Panel 1: Projects
    ax = axes[0, 0]
    ranked = sorted(zip(projects, proj_scores), key=lambda x: x[1])
    names = [p["name"] for p, _ in ranked]
    scores = [s for _, s in ranked]
    colors = [SEL_COLOR if p["id"]
              in sel_proj_ids else UNSEL_COLOR for p, _ in ranked]
    bars = ax.barh(names, scores, color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Hybrid Score", fontsize=9)
    ax.set_title("Projects", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
    ax.tick_params(axis="y", labelsize=8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=7, color="#374151")
    ax.legend(
        handles=[Patch(facecolor=SEL_COLOR, label="Selected"), Patch(
            facecolor=UNSEL_COLOR, label="Not selected")],
        loc="lower right", fontsize=8, framealpha=0.9,
    )

    # Panel 2: Experience
    ax = axes[0, 1]
    ranked = sorted(zip(roles, role_scores), key=lambda x: x[1])
    names = [f"{r['title']}\n({r['company']})" for r, _ in ranked]
    scores = [s for _, s in ranked]
    colors = [SEL_COLOR if r["id"]
              in sel_role_ids else UNSEL_COLOR for r, _ in ranked]
    bars = ax.barh(names, scores, color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Hybrid Score", fontsize=9)
    ax.set_title("Experience", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
    ax.tick_params(axis="y", labelsize=8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=7, color="#374151")

    # Panel 3: Skills
    ax = axes[1, 0]
    sorted_skills = sorted(all_skill_scores, key=lambda x: x[2])
    top_n = 25
    if len(sorted_skills) > top_n:
        sorted_skills = sorted_skills[-top_n:]
    s_names = [s[0] for s in sorted_skills]
    s_scores = [s[2] for s in sorted_skills]
    s_colors = [CAT_COLORS.get(s[1], "#9CA3AF") for s in sorted_skills]
    bars = ax.barh(s_names, s_scores, color=s_colors,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Hybrid Score", fontsize=9)
    ax.set_title(f"Skills (Top {len(sorted_skills)})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(s_scores) * 1.15 if s_scores else 1)
    ax.tick_params(axis="y", labelsize=7)
    for bar, score in zip(bars, s_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=6, color="#374151")
    unique_cats = list(dict.fromkeys(s[1] for s in sorted_skills))
    ax.legend(
        handles=[Patch(facecolor=CAT_COLORS.get(c, "#9CA3AF"), label=c)
                 for c in unique_cats],
        loc="lower right", fontsize=6, framealpha=0.9, ncol=1,
    )

    # Panel 4: Certifications
    ax = axes[1, 1]
    ranked = sorted(zip(certs, cert_scores), key=lambda x: x[1])
    names = [f"{c['name']}\n({c['issuer']})" for c, _ in ranked]
    scores = [s for _, s in ranked]
    colors = [SEL_COLOR if c["id"]
              in sel_cert_ids else UNSEL_COLOR for c, _ in ranked]
    bars = ax.barh(names, scores, color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.axvline(x=cert_threshold, color=THRESH_COLOR, linestyle="--",
               linewidth=1, alpha=0.7, label=f"Threshold ({cert_threshold:.2f})")
    ax.set_xlabel("Hybrid Score", fontsize=9)
    ax.set_title("Certifications", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(max(scores) if scores else 0, cert_threshold) * 1.15 or 1)
    ax.tick_params(axis="y", labelsize=8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=7, color="#374151")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    out_path = out_dir / "match_viz.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ── v2: requirement coverage ──────────────────────────────────────────────────
KIND_COLORS = {"must": "#2563EB", "responsibility": "#10B981", "nice": "#9CA3AF"}
_TRACE_ROWS = 40


def _truncate(text, width=70):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def generate_v2_report(
    company, role, requirements, claims, selection, matrix, skill_lines,
    certs=None, selected_certs=None, gaps=None, budget=None, pages=None,
    attempts=1, gap_threshold=0.3, profile=None, variants=None,
):
    """The v2 match report: what was asked for, what was shown, and why.

    Unlike the v1 report — a ranking of whole items — this is a coverage
    ledger: every requirement the posting was parsed into, the claim that
    covers it best, and the greedy trace that got there. ``profile`` is the
    :class:`ats_tailor.jd.models.JDProfile` the posting was read into; its
    gates go at the top, above any score, for the same reason the run prints
    them first.
    """
    by_id = {c.id: c for c in claims}
    index_of = {c.id: i for i, c in enumerate(claims)}
    chosen = list(selection.claim_ids)
    coverage = selection.coverage

    lines = [
        f"# Match Report (v2): {company} – {role}",
        "",
        f"- requirements parsed: **{len(requirements)}** "
        f"({sum(1 for r in requirements if r.kind == 'must')} must, "
        f"{sum(1 for r in requirements if r.kind == 'nice')} nice, "
        f"{sum(1 for r in requirements if r.kind == 'responsibility')} responsibility)",
        f"- claims available: **{len(claims)}**, selected: **{len(chosen)}**",
        f"- objective: **{selection.objective:.3f}**, "
        f"lines used: **{selection.budget_used:.0f}**"
        + (f" of **{budget:.0f}**" if budget is not None else ""),
        f"- page-fit attempts: **{attempts}**"
        + (f", final page count: **{pages}**" if pages else ""),
        "",
    ]
    if profile is not None:
        lines += _gate_section(profile)
    lines += [
        "## Requirements",
        "",
        "| id | kind | w | coverage | covered | best claim | score | requirement |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for requirement in requirements:
        best_id, best_score = _best_claim(matrix, index_of, chosen, requirement, requirements)
        value = float(coverage.get(requirement.id, 0.0))
        lines.append(
            f"| {requirement.id} | {requirement.kind} | {requirement.weight:.1f} | "
            f"{value:.2f} | {'Y' if value >= gap_threshold else ''} | "
            f"{_md_cell(best_id or '—')} | {best_score:.2f} | "
            f"{_md_cell(_truncate(requirement.text))} |"
        )

    lines += ["", "## Selection trace", "",
              "| # | claim | section | group | cost | gain | opened | covers |",
              "|---|---|---|---|---|---|---|---|"]
    for i, step in enumerate(selection.trace[:_TRACE_ROWS], 1):
        claim = by_id.get(step.claim_id)
        covers = ", ".join(f"{rid} {delta:+.2f}" for rid, delta in step.covered[:3]) or "—"
        lines.append(
            f"| {i} | {_md_cell(step.claim_id)} | {claim.section if claim else '?'} | "
            f"{_md_cell(claim.group if claim else '?')} | "
            f"{claim.cost:.0f} | {step.gain:.3f} | "
            f"{'yes' if step.opened_group else ''}{' (repair)' if step.repair else ''} | "
            f"{covers} |"
        )
    if len(selection.trace) > _TRACE_ROWS:
        lines.append(f"| … | {len(selection.trace) - _TRACE_ROWS} more | | | | | | |")

    lines += ["", "## Sections", "",
              "| section | group | claims | lines |", "|---|---|---|---|"]
    for section, groups in selection.group_ids_by_section.items():
        for group in groups:
            held = [c for c in chosen if by_id.get(c) and by_id[c].group == group]
            cost = sum(by_id[c].cost for c in held)
            lines.append(f"| {section} | {_md_cell(group)} | {len(held)} | {cost:.0f} |")

    lines += ["", "## Skills lines", "",
              "Ordered by how much requirement weight the category won; "
              "skills within a line by marginal gain.", ""]
    for line in skill_lines:
        lines.append(f"- **{line['category']}:** {line['skills']}")

    if selected_certs is not None:
        lines += ["", "## Certifications", "",
                  ", ".join(c.get("name", "") for c in selected_certs) or "none"]

    weak = [r for r in requirements if float(coverage.get(r.id, 0.0)) < gap_threshold]
    if weak:
        lines += ["", f"## Uncovered requirements (coverage < {gap_threshold})", "",
                  "| id | kind | requirement |", "|---|---|---|"]
        for requirement in weak:
            lines.append(
                f"| {requirement.id} | {requirement.kind} | "
                f"{_md_cell(_truncate(requirement.text, 90))} |"
            )

    if variants:
        lines += ["", "## Phrasing variants used", "",
                  "Accepted alternate wording for this posting's archetype. The "
                  "claim is the same claim; only the sentence differs, and the "
                  "guardrail checked that it still says the same thing.", "",
                  "| bullet | archetype | guardrail | text |", "|---|---|---|---|"]
        for row in variants:
            verdict = row.get("guardrail") or {}
            state = "ok" if verdict.get("ok") else (
                "—" if not verdict else "FAILED")
            lines.append(
                f"| {_md_cell(str(row.get('bullet_id', '')))} | "
                f"{_md_cell(str(row.get('archetype', '')))} | {state} | "
                f"{_md_cell(_truncate(str(row.get('text', '')), 90))} |")

    if gaps:
        lines += ["", "## Index gaps", "",
                  "Terms this posting's requirements name with no claim behind "
                  "them — report them, never stuff them.", "", ", ".join(gaps)]

    lines.append("")
    return "\n".join(lines)


def _gate_section(profile) -> list[str]:
    """Gates, quoted, before anything the selection decided."""
    lines = [
        f"- read by: **{profile.provider}**"
        + (f" (prompt {profile.prompt_version})" if profile.prompt_version else "")
        + (" — cached" if profile.cached else ""),
    ]
    if profile.title:
        lines.append(f"- target title: **{profile.title}**")
    if profile.archetype:
        lines.append(f"- archetype: **{profile.archetype}**")
    lines += ["", "## Gates", "",
              "Hard filters, quoted from the posting. Nothing here changes a "
              "score — it decides whether the application is possible at all.",
              "", "| gate | verdict | detail | posting says |", "|---|---|---|---|"]
    for name, gate in profile.gates.items():
        detail = gate.note or ""
        if gate.date:
            detail = f"{gate.date} — {detail}" if detail else gate.date
        quote = _md_cell(gate.quote or "—")
        lines.append(f"| {name} | {gate.verdict} | {_md_cell(detail or '—')} | {quote} |")
    blocking = profile.gates.blocking()
    if blocking:
        lines += ["", "**Check before applying:** " + ", ".join(
            f"{name} ({gate.verdict})" for name, gate in blocking)]
    lines.append("")
    return lines


def _best_claim(matrix, index_of, chosen, requirement, requirements):
    """The selected claim covering ``requirement`` best, and its matrix score."""
    row = list(requirements).index(requirement)
    best_id, best_score = None, 0.0
    for claim_id in chosen:
        column = index_of.get(claim_id)
        if column is None:
            continue
        score = float(matrix[row][column])
        if best_id is None or score > best_score:
            best_id, best_score = claim_id, score
    return best_id, best_score


def generate_requirement_visualization(
    out_dir, company, role, requirements, coverage, gap_threshold=0.3
):
    """One bar per requirement, coloured by kind, with the gap threshold marked."""
    out_path = out_dir / "match_viz.png"
    if not requirements:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no requirements parsed", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path

    ordered = sorted(
        requirements,
        key=lambda r: (float(coverage.get(r.id, 0.0)), r.id),
    )
    labels = [f"{r.id}  {_truncate(r.text, 58)}" for r in ordered]
    values = [float(coverage.get(r.id, 0.0)) for r in ordered]
    colors = [KIND_COLORS.get(r.kind, "#9CA3AF") for r in ordered]

    height = max(3.0, 0.28 * len(ordered) + 1.4)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=gap_threshold, color=THRESH_COLOR, linestyle="--", linewidth=1,
               alpha=0.8, label=f"gap threshold ({gap_threshold:.2f})")
    ax.set_xlabel("coverage (soft-pooled best match over the selected claims)", fontsize=9)
    ax.set_title(f"Requirement coverage: {company} – {role}",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(1.0, max(values) * 1.1))
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(
        handles=[Patch(facecolor=color, label=kind) for kind, color in KIND_COLORS.items()]
        + [Patch(facecolor=THRESH_COLOR, label=f"threshold {gap_threshold:.2f}")],
        loc="lower right", fontsize=8, framealpha=0.9,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
