"""Candidate selection and page budget estimation."""

import math

import numpy as np

from .scoring import (
    build_skill_text,
    compute_keyword_bonus,
    cosine_sim,
    score_against_jd,
)

MAX_EXPERIENCE = 3
MAX_PROJECTS = 4
MIN_PROJECTS = 4
MAX_SKILL_LINES = 4
MAX_PROJECT_BULLETS = 3  # per project
MAX_EXP_BULLETS = 3  # per role
CHARS_PER_BULLET_LINE = 80  # approx chars before LaTeX wraps a bullet

PROF_RANK = {"advanced": 0, "intermediate": 1, "familiar": 2}


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
            kw = compute_keyword_bonus(
                skill.get("ats_keywords", []), jd_lower, jd_text)
            score = sem + kw
            skill_rankings.append((skill, score))
            all_skill_scores.append((skill["name"], cat["name"], score))

        skill_rankings.sort(
            key=lambda x: (PROF_RANK.get(
                x[0].get("proficiency", "familiar"), 2), -x[1])
        )
        top5_scores = sorted([s for _, s in skill_rankings], reverse=True)[:5]
        avg_score = np.mean(top5_scores) if top5_scores else 0

        if cat["name"] == "Programming Languages":
            top_skills = [s["name"]
                          for s, sc in skill_rankings[:12] if sc > 0.1]
            if not top_skills:
                top_skills = [s["name"] for s, _ in skill_rankings[:6]]
            lang_line = {"category": "Languages",
                         "skills": ", ".join(top_skills)}
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


def estimate_line_count(experience, projects, summary=None):
    """Rough estimate of content lines to check 1-page fit."""
    lines = 0
    lines += 3  # header
    if summary:
        lines += 2  # section header + summary text
        lines += len(summary.get("highlights", []))  # one line per highlight
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
        tech_lines = math.ceil(len(proj.get("tech_line", "")) /
                               CHARS_PER_BULLET_LINE) if proj.get("tech_line") else 1
        lines += 1 + tech_lines + bullet_lines + 1
    lines += 5  # skills section
    lines += 3  # certifications + spacing
    return lines
