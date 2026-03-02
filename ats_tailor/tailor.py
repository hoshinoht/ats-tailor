#!/usr/bin/env python3
"""ATS Resume Tailoring Tool.

Takes a job description, matches it against the YAML index using semantic
embeddings, selects the best content, and outputs a 1-page A4 LaTeX resume.

Usage:
    python -m ats_tailor.tailor --jd path/to/jd.txt --company acme --role backend
    python -m ats_tailor.tailor --profile profile.yaml --index index/ --jd jd.txt --company acme --role backend
"""

import argparse
import math
import os
import re
import sys
from datetime import datetime, date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
PKG_DIR = Path(__file__).resolve().parent

# ── Page budget (approximate line counts for 1 A4 page at 10pt) ────────────
MAX_EXPERIENCE = 2
MAX_PROJECTS = 4
MIN_PROJECTS = 3
MAX_SKILL_LINES = 4
MAX_PROJECT_BULLETS = 3  # per project
MAX_EXP_BULLETS = 3  # per role
CHARS_PER_BULLET_LINE = 80  # approx chars before LaTeX wraps a bullet

AMBIGUOUS_KEYWORDS = {"go", "c", "r"}

PROF_RANK = {"advanced": 0, "intermediate": 1, "familiar": 2}
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ── Keyword matching & recency ────────────────────────────────────────────────
def keyword_in_text(keyword, text_lower, text_original=""):
    """Check if keyword appears in text with word boundaries.

    For ambiguous short keywords (Go, C, R, etc.), uses case-sensitive
    matching against the original text — JDs write languages canonically
    ("Go", "C") while English homonyms are lowercase.
    """
    kw = keyword.lower()
    # Keywords with special chars (C++, C#, .NET) — substring match
    if not kw.replace(" ", "").replace("-", "").isalnum():
        return kw in text_lower
    # Ambiguous short keywords — case-sensitive match on original text
    if kw in AMBIGUOUS_KEYWORDS or (len(kw) <= 2 and kw.isalpha()):
        if not text_original:
            return bool(re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
        return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text_original))
    # Alphanumeric keywords — word boundary match
    return bool(re.search(r"\b" + re.escape(kw) + r"\b", text_lower))


def compute_keyword_bonus(keywords, jd_lower, jd_text=""):
    """Bonus score for exact keyword matches found in JD text."""
    matches = sum(1 for kw in keywords if keyword_in_text(kw, jd_lower, jd_text))
    return min(matches * 0.06, 0.20)


def compute_tag_overlap_bonus(tags, jd_lower, jd_text=""):
    """Bonus for proportion of an item's tags that appear in the JD.

    Unlike keyword_bonus (which caps at 4 matches), this rewards items
    whose *entire tag set* aligns with the JD — i.e. domain relevance.
    A project where 8/10 tags match the JD scores higher than one where
    only 2/10 match, even if both hit the keyword_bonus cap.
    """
    if not tags:
        return 0.0
    hits = sum(1 for t in tags if keyword_in_text(t, jd_lower, jd_text))
    ratio = hits / len(tags)
    return ratio * 0.25  # max 0.25 when 100% of tags match


def parse_end_date(period):
    """Extract the most recent date from a period string."""
    if "present" in period.lower():
        return datetime.now()
    matches = re.findall(r"([A-Za-z]{3})\s+(\d{4})", period)
    if matches:
        month_str, year_str = matches[-1]
        month = MONTH_MAP.get(month_str.lower()[:3], 6)
        return datetime(int(year_str), month, 1)
    years = re.findall(r"(\d{4})", period)
    if years:
        return datetime(int(years[-1]), 6, 1)
    return datetime(2020, 1, 1)


def recency_multiplier(period):
    """Recency boost: current=1.10, <12mo=1.05, <24mo=1.02, older=1.00."""
    end_date = parse_end_date(period)
    now = datetime.now()
    months_ago = (now.year - end_date.year) * 12 + (now.month - end_date.month)
    if months_ago <= 0:
        return 1.10
    elif months_ago <= 12:
        return 1.05
    elif months_ago <= 24:
        return 1.02
    return 1.00


# ── YAML loading ───────────────────────────────────────────────────────────────
def load_index(index_dir):
    """Load all YAML index files from the given directory."""
    index_dir = Path(index_dir)
    with open(index_dir / "skills.yaml") as f:
        skills_data = yaml.safe_load(f)
    with open(index_dir / "projects.yaml") as f:
        projects_data = yaml.safe_load(f)
    with open(index_dir / "experience.yaml") as f:
        exp_data = yaml.safe_load(f)
    with open(index_dir / "certifications.yaml") as f:
        certs_data = yaml.safe_load(f)
    return skills_data, projects_data, exp_data, certs_data


def load_profile(profile_path):
    """Load profile.yaml with personal info (name, email, phone, links)."""
    with open(profile_path) as f:
        return yaml.safe_load(f)


# ── Embedding helpers ──────────────────────────────────────────────────────────
def chunk_text(text, max_len=512):
    """Split text into sentence-ish chunks for embedding."""
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(". ", ".\n").split("\n")
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences if sentences else [text[:max_len]]


def build_skill_text(skill):
    """Build a text representation of a skill for embedding."""
    keywords = " ".join(skill.get("ats_keywords", []))
    return f"{skill['name']} {keywords}"


def build_project_text(proj):
    """Build a text representation of a project for embedding."""
    tags = " ".join(proj.get("ats_tags", []))
    tech = []
    for key in ("languages", "frameworks", "infrastructure", "patterns", "tools"):
        tech.extend(proj.get("tech_stack", {}).get(key, []))
    tech_str = " ".join(tech)
    resps = " ".join(proj.get("responsibilities", []))
    return f"{proj['name']} {proj.get('summary', '')} {tags} {tech_str} {resps}"


def build_role_text(role):
    """Build a text representation of a role for embedding."""
    skills = " ".join(role.get("skills_used", []))
    tags = " ".join(role.get("ats_tags", []))
    bullets = " ".join(role.get("bullets", []))
    return f"{role['title']} {role['company']} {skills} {tags} {bullets}"


def build_cert_text(cert):
    """Build a text representation of a certification for embedding."""
    keywords = " ".join(cert.get("ats_keywords", []))
    return f"{cert['name']} {cert['issuer']} {keywords}"


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def score_against_jd(jd_embeddings, item_embedding):
    """Max-pooled cosine similarity: best match across JD sentences."""
    return max(cosine_sim(jd_emb, item_embedding) for jd_emb in jd_embeddings)


# ── Selection logic ────────────────────────────────────────────────────────────
def select_experience(roles, scores, max_n=MAX_EXPERIENCE):
    """Select top roles by similarity. Prefer recent + relevant."""
    ranked = sorted(zip(roles, scores), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked[:max_n]], [s for _, s in ranked[:max_n]]


def select_projects(projects, scores, max_n=MAX_PROJECTS):
    """Select top projects by similarity score."""
    ranked = sorted(zip(projects, scores), key=lambda x: x[1], reverse=True)
    selected = [p for p, _ in ranked[:max_n]]
    selected_scores = [s for _, s in ranked[:max_n]]
    return selected, selected_scores


def select_skills(categories, jd_embeddings, model, jd_lower, jd_text="", max_lines=MAX_SKILL_LINES):
    """Select top skills grouped by category, returning formatted lines."""
    cat_scores = []
    lang_line = None
    all_skill_scores = []

    for cat in categories:
        skill_rankings = []
        for skill in cat["skills"]:
            text = build_skill_text(skill)
            emb = model.encode(text)
            sem = score_against_jd(jd_embeddings, emb)
            kw = compute_keyword_bonus(skill.get("ats_keywords", []), jd_lower, jd_text)
            score = sem + kw
            skill_rankings.append((skill, score))
            all_skill_scores.append((skill["name"], cat["name"], score))

        skill_rankings.sort(
            key=lambda x: (PROF_RANK.get(x[0].get("proficiency", "familiar"), 2), -x[1])
        )
        top5_scores = sorted([s for _, s in skill_rankings], reverse=True)[:5]
        avg_score = np.mean(top5_scores) if top5_scores else 0

        if cat["name"] == "Programming Languages":
            top_skills = [s["name"] for s, sc in skill_rankings[:12] if sc > 0.1]
            if not top_skills:
                top_skills = [s["name"] for s, _ in skill_rankings[:6]]
            lang_line = {"category": "Languages", "skills": ", ".join(top_skills)}
        else:
            cat_scores.append((cat, skill_rankings, avg_score))

    cat_scores.sort(key=lambda x: x[2], reverse=True)

    lines = []
    if lang_line:
        lines.append(lang_line)

    for cat, skill_rankings, _ in cat_scores[:max_lines - len(lines)]:
        top_skills = [s["name"] for s, sc in skill_rankings[:10] if sc > 0.15]
        if not top_skills:
            top_skills = [s["name"] for s, _ in skill_rankings[:3]]
        if top_skills:
            lines.append({
                "category": cat["name"],
                "skills": ", ".join(top_skills),
            })

    return lines[:max_lines], all_skill_scores


def select_certifications(certs, scores, threshold=0.25):
    """Select certifications above the relevance threshold."""
    ranked = sorted(zip(certs, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if s > threshold]


# ── Coverage gap detection ─────────────────────────────────────────────────
_GAP_STOPWORDS = {
    "API", "CEO", "CTO", "CRM", "ERP", "HR", "IT", "KPI", "MBA", "OKR",
    "PM", "QA", "ROI", "SLA", "UI", "UX", "VP", "B2B", "B2C", "SaaS",
    "SEO", "SEM", "ETL", "USA", "APAC", "EMEA", "NUS", "NTU", "SMU",
    "MNC", "JD", "CV", "ID", "AI", "ML", "DL", "BSc", "MSc", "PhD",
    "The", "This", "That", "These", "Those", "What", "When", "Where",
    "Which", "While", "Who", "How", "Why", "Our", "Your", "They",
    "You", "We", "He", "She", "Its", "Are", "Can", "May", "Will",
    "Must", "Should", "Would", "Could", "Has", "Have", "Had", "Does",
    "Did", "Not", "But", "And", "For", "With", "From", "About",
    "Into", "Over", "After", "Before", "Between", "Through",
    "Under", "During", "Without", "Within", "Along", "Among",
    "Also", "Just", "Only", "Very", "Most", "Some", "Any", "All",
    "Each", "Every", "Both", "Few", "More", "Less", "Much", "Many",
    "Such", "Own", "Other", "Another", "New", "Good", "Great",
    "Strong", "Key", "Well", "Best", "First", "Last", "High", "Low",
    "Full", "Part", "Real", "True", "Open", "Free", "Large", "Small",
    "Long", "Short", "Hard", "Soft", "Deep", "Wide", "Fast", "Slow",
    "Big", "End", "Top", "Set", "Run", "Use", "Get", "Let", "Put",
    "Say", "See", "Try", "Ask", "Need", "Want", "Like", "Make",
    "Take", "Give", "Come", "Work", "Know", "Think", "Help", "Join",
    "Lead", "Build", "Drive", "Create", "Develop", "Design", "Manage",
    "Support", "Ensure", "Provide", "Maintain", "Implement", "Deliver",
    "Collaborate", "Contribute", "Communicate", "Analyse", "Analyze",
    "Review", "Apply", "Experience", "Ability", "Skills", "Team",
    "Role", "Position", "Company", "Location", "Singapore", "Remote",
    "Hybrid", "Office", "Department", "Level", "Senior", "Junior",
    "Principal", "Staff", "Intern", "Associate", "Manager", "Director",
    "Engineer", "Developer", "Analyst", "Scientist", "Architect",
    "Consultant", "Specialist", "Coordinator", "Administrator",
}


def detect_coverage_gaps(jd_text, skills_data, projects_data, exp_data, certs_data):
    """Find tech-looking terms in the JD that aren't in any index file."""
    known = set()
    for cat in skills_data.get("categories", []):
        for skill in cat.get("skills", []):
            known.add(skill["name"].lower())
            for kw in skill.get("ats_keywords", []):
                known.add(kw.lower())
    for proj in projects_data.get("projects", []):
        for tag in proj.get("ats_tags", []):
            known.add(tag.lower())
        for key in ("languages", "frameworks", "infrastructure", "patterns", "tools"):
            for t in proj.get("tech_stack", {}).get(key, []):
                known.add(t.lower())
    for role in exp_data.get("roles", []):
        for tag in role.get("ats_tags", []):
            known.add(tag.lower())
        for s in role.get("skills_used", []):
            known.add(s.lower())
    for cert in certs_data.get("certifications", []):
        for kw in cert.get("ats_keywords", []):
            known.add(kw.lower())

    candidates = set()
    candidates.update(re.findall(r"\b([A-Z][A-Z0-9]{1,5})\b", jd_text))
    candidates.update(re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", jd_text))
    candidates.update(re.findall(r"\b([A-Z][a-z]{2,})\b", jd_text))
    candidates.update(re.findall(r"\b(C\+\+|C#|\.NET|F#)\b", jd_text))

    gaps = sorted(
        c for c in candidates
        if c.lower() not in known and c not in _GAP_STOPWORDS
    )
    return gaps


# ── LaTeX escaping ─────────────────────────────────────────────────────────────
def latex_escape(text):
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        return text
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# ── Template rendering ─────────────────────────────────────────────────────────
def build_project_context(proj):
    """Build template context for a project."""
    tech_parts = []
    for key in ("languages", "frameworks", "infrastructure", "patterns", "tools"):
        tech_parts.extend(proj.get("tech_stack", {}).get(key, []))
    tech_line = ", ".join(tech_parts[:8])

    bullets = proj.get("responsibilities", [])[:MAX_PROJECT_BULLETS]

    summary = proj.get("summary", proj["name"])
    seps = [" with ", " using ", " for ", " via ", " comparing ", " between "]
    cut_positions = [summary.index(s) for s in seps if s in summary]
    if cut_positions:
        summary = summary[:min(cut_positions)]
    tagline = summary[:50]
    if len(summary) > 50:
        tagline = tagline.rsplit(" ", 1)[0]

    return {
        "name": proj["name"],
        "tagline": tagline,
        "period": proj["period"],
        "tech_line": tech_line,
        "bullets": bullets,
    }


def build_role_context(role):
    """Build template context for a role."""
    return {
        "company": role["company"],
        "title": role["title"],
        "period": role["period"],
        "bullets": role.get("bullets", [])[:MAX_EXP_BULLETS],
    }


def estimate_line_count(experience, projects):
    """Rough estimate of content lines to check 1-page fit."""
    lines = 0
    lines += 3  # header
    lines += 4  # education (2 entries)
    lines += 2  # section headers for exp
    for role in experience:
        bullet_lines = sum(
            math.ceil(len(b) / CHARS_PER_BULLET_LINE) for b in role["bullets"]
        )
        lines += 1 + bullet_lines + 1
    lines += 2  # section header for projects
    for proj in projects:
        bullet_lines = sum(
            math.ceil(len(b) / CHARS_PER_BULLET_LINE) for b in proj["bullets"]
        )
        tech_lines = math.ceil(len(proj.get("tech_line", "")) / CHARS_PER_BULLET_LINE) if proj.get("tech_line") else 1
        lines += 1 + tech_lines + bullet_lines + 1
    lines += 5  # skills section
    lines += 3  # certifications + spacing
    return lines


def render_resume(company, role, profile, education, experience, projects, skill_lines, certifications):
    """Render the LaTeX resume from the Jinja2 template."""
    env = Environment(
        loader=FileSystemLoader(str(PKG_DIR)),
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((#",
        comment_end_string="#))",
    )
    env.filters["latex_escape"] = latex_escape

    template = env.get_template("template.tex.j2")
    return template.render(
        company=company,
        role=role,
        profile=profile,
        education=education,
        experience=experience,
        projects=projects,
        skill_lines=skill_lines,
        certifications=certifications,
    )


def render_prompt(company, role, jd_text, experience, projects):
    """Render the LLM prompt from the Jinja2 template."""
    env = Environment(loader=FileSystemLoader(str(PKG_DIR)))
    template = env.get_template("prompt_template.md.j2")
    return template.render(
        company=company,
        role=role,
        jd_text=jd_text,
        experience=experience,
        projects=projects,
    )


# ── Match report ───────────────────────────────────────────────────────────────
def generate_report(company, role, exp_ranked, proj_ranked, skill_lines, cert_ranked, gaps=None):
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

    lines.append("")
    return "\n".join(lines)


# ── Visualization ──────────────────────────────────────────────────────────────
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


def generate_visualization(
    out_dir, company, role,
    projects, proj_scores, sel_proj_ids,
    roles, role_scores, sel_role_ids,
    all_skill_scores,
    certs, cert_scores, sel_cert_ids,
):
    """Generate a 4-panel match visualization PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Match Analysis: {company} – {role}", fontsize=16, fontweight="bold", y=0.98)
    plt.subplots_adjust(hspace=0.35, wspace=0.45, top=0.93, bottom=0.04, left=0.18, right=0.96)

    from matplotlib.patches import Patch

    # Panel 1: Projects
    ax = axes[0, 0]
    ranked = sorted(zip(projects, proj_scores), key=lambda x: x[1])
    names = [p["name"] for p, _ in ranked]
    scores = [s for _, s in ranked]
    colors = [SEL_COLOR if p["id"] in sel_proj_ids else UNSEL_COLOR for p, _ in ranked]
    bars = ax.barh(names, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Hybrid Score", fontsize=9)
    ax.set_title("Projects", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
    ax.tick_params(axis="y", labelsize=8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=7, color="#374151")
    ax.legend(
        handles=[Patch(facecolor=SEL_COLOR, label="Selected"), Patch(facecolor=UNSEL_COLOR, label="Not selected")],
        loc="lower right", fontsize=8, framealpha=0.9,
    )

    # Panel 2: Experience
    ax = axes[0, 1]
    ranked = sorted(zip(roles, role_scores), key=lambda x: x[1])
    names = [f"{r['title']}\n({r['company']})" for r, _ in ranked]
    scores = [s for _, s in ranked]
    colors = [SEL_COLOR if r["id"] in sel_role_ids else UNSEL_COLOR for r, _ in ranked]
    bars = ax.barh(names, scores, color=colors, edgecolor="white", linewidth=0.5)
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
    bars = ax.barh(s_names, s_scores, color=s_colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Hybrid Score", fontsize=9)
    ax.set_title(f"Skills (Top {len(sorted_skills)})", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(s_scores) * 1.15 if s_scores else 1)
    ax.tick_params(axis="y", labelsize=7)
    for bar, score in zip(bars, s_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=6, color="#374151")
    unique_cats = list(dict.fromkeys(s[1] for s in sorted_skills))
    ax.legend(
        handles=[Patch(facecolor=CAT_COLORS.get(c, "#9CA3AF"), label=c) for c in unique_cats],
        loc="lower right", fontsize=6, framealpha=0.9, ncol=1,
    )

    # Panel 4: Certifications
    ax = axes[1, 1]
    ranked = sorted(zip(certs, cert_scores), key=lambda x: x[1])
    names = [f"{c['name']}\n({c['issuer']})" for c, _ in ranked]
    scores = [s for _, s in ranked]
    colors = [SEL_COLOR if c["id"] in sel_cert_ids else UNSEL_COLOR for c, _ in ranked]
    bars = ax.barh(names, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.25, color=THRESH_COLOR, linestyle="--", linewidth=1, alpha=0.7, label="Threshold (0.25)")
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


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tailor a resume to a job description")
    parser.add_argument("--jd", type=str, help="Path to job description text file")
    parser.add_argument("--company", required=False, help="Company name (used in output path)")
    parser.add_argument("--role", required=False, help="Role name (used in output path)")
    parser.add_argument("--output", type=str, help="Output directory (default: output/{company}-{role})")
    parser.add_argument("--index", type=str, default="index", help="Path to YAML index directory (default: index/)")
    parser.add_argument("--profile", type=str, default=None, help="Path to profile.yaml (default: profile.yaml next to index dir)")
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
    skills_data, projects_data, exp_data, certs_data = load_index(index_dir)

    categories = skills_data["categories"]
    projects = projects_data["projects"]
    roles = exp_data["roles"]
    education = exp_data["education"][:2]
    certs = certs_data["certifications"]

    # ── Coverage gap detection ──
    gaps = detect_coverage_gaps(jd_text, skills_data, projects_data, exp_data, certs_data)
    if gaps:
        print(f"  Warning: {len(gaps)} JD term(s) not in index: {', '.join(gaps)}")

    # ── Load model & embed ──
    print("Loading embedding model (first run downloads ~22MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding job description...")
    jd_chunks = chunk_text(jd_text)
    jd_embeddings = model.encode(jd_chunks)

    # ── Score everything (hybrid: semantic + keyword + recency) ──
    print("Scoring candidates...")

    role_texts = [build_role_text(r) for r in roles]
    role_embeddings = model.encode(role_texts)
    role_scores = [
        (score_against_jd(jd_embeddings, e)
         + compute_keyword_bonus(r.get("ats_tags", []), jd_lower, jd_text)
         + compute_tag_overlap_bonus(r.get("ats_tags", []), jd_lower, jd_text))
        * recency_multiplier(r["period"])
        for r, e in zip(roles, role_embeddings)
    ]

    proj_texts = [build_project_text(p) for p in projects]
    proj_embeddings = model.encode(proj_texts)
    proj_scores = [
        (score_against_jd(jd_embeddings, e)
         + compute_keyword_bonus(p.get("ats_tags", []), jd_lower, jd_text)
         + compute_tag_overlap_bonus(p.get("ats_tags", []), jd_lower, jd_text))
        * recency_multiplier(p["period"])
        for p, e in zip(projects, proj_embeddings)
    ]

    cert_texts = [build_cert_text(c) for c in certs]
    cert_embeddings = model.encode(cert_texts)
    cert_scores = [
        score_against_jd(jd_embeddings, e) + compute_keyword_bonus(c.get("ats_keywords", []), jd_lower, jd_text)
        for c, e in zip(certs, cert_embeddings)
    ]

    # ── Select ──
    print("Selecting best content...")
    sel_roles, sel_role_scores = select_experience(roles, role_scores)
    sel_projects, sel_proj_scores = select_projects(projects, proj_scores)
    skill_lines, all_skill_scores = select_skills(categories, jd_embeddings, model, jd_lower, jd_text)
    sel_certs = select_certifications(certs, cert_scores)

    # ── Budget check ──
    exp_ctx = [build_role_context(r) for r in sel_roles]
    proj_ctx = [build_project_context(p) for p in sel_projects]

    line_est = estimate_line_count(exp_ctx, proj_ctx)
    max_lines = 55

    while line_est > max_lines and len(proj_ctx) > MIN_PROJECTS:
        proj_ctx.pop()
        sel_projects.pop()
        sel_proj_scores.pop()
        line_est = estimate_line_count(exp_ctx, proj_ctx)

    if line_est > max_lines:
        for p in proj_ctx:
            if len(p["bullets"]) > 2:
                p["bullets"] = p["bullets"][:2]
        line_est = estimate_line_count(exp_ctx, proj_ctx)

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
    exp_ranked = sorted(zip(roles, role_scores), key=lambda x: x[1], reverse=True)
    sel_proj_ids = {p["id"] for p in sel_projects}
    proj_ranked = sorted(
        [(p, s, p["id"] in sel_proj_ids) for p, s in zip(projects, proj_scores)],
        key=lambda x: x[1],
        reverse=True,
    )
    sel_cert_ids = {c["id"] for c in sel_certs}
    cert_ranked = [
        (c, s, c["id"] in sel_cert_ids)
        for c, s in sorted(zip(certs, cert_scores), key=lambda x: x[1], reverse=True)
    ]

    report = generate_report(args.company, args.role, exp_ranked, proj_ranked, skill_lines, cert_ranked, gaps)
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
