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


def generate_report(company, role, exp_ranked, proj_ranked, skill_lines, cert_ranked, gaps=None, llm_terms=None):
    """Generate a markdown match report."""
    lines = [
        f"# Match Report: {company} – {role}",
        "",
        "## Experience Rankings",
        "",
        "| Rank | Role | Company | Score |",
        "|------|------|---------|-------|",
    ]
    for i, (r, s) in enumerate(exp_ranked, 1):
        lines.append(f"| {i} | {r['title']} | {r['company']} | {s:.3f} |")

    lines += [
        "",
        "## Project Rankings",
        "",
        "| Rank | Project | Score | Selected |",
        "|------|---------|-------|----------|",
    ]
    for i, (p, s, sel) in enumerate(proj_ranked, 1):
        mark = "Y" if sel else ""
        lines.append(f"| {i} | {p['name']} | {s:.3f} | {mark} |")

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
            "| Cert | Score | Selected |",
            "|------|-------|----------|",
        ]
        for c, s, sel in cert_ranked:
            mark = "Y" if sel else ""
            lines.append(f"| {c['name']} | {s:.3f} | {mark} |")

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
    certs, cert_scores, sel_cert_ids,
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
    ax.axvline(x=0.25, color=THRESH_COLOR, linestyle="--",
               linewidth=1, alpha=0.7, label="Threshold (0.25)")
    ax.set_xlabel("Hybrid Score", fontsize=9)
    ax.set_title("Certifications", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
    ax.tick_params(axis="y", labelsize=8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=7, color="#374151")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    out_path = out_dir / "match_viz.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
