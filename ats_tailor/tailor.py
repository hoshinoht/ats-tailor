#!/usr/bin/env python3
"""ATS Resume Tailoring Tool.

Takes a job description, matches it against the YAML index using semantic
embeddings, selects the best content, and outputs a 1-page A4 LaTeX resume.

Usage:
    python -m ats_tailor.tailor --jd path/to/jd.txt --company acme --role backend
    python -m ats_tailor.tailor --profile profile.yaml --index index/ --jd jd.txt --company acme --role backend
"""

from sentence_transformers import SentenceTransformer
from jinja2 import Environment, FileSystemLoader
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# ── Paths ──────────────────────────────────────────────────────────────────────
PKG_DIR = Path(__file__).resolve().parent

# ── Page budget (approximate line counts for 1 A4 page at 10pt) ────────────
MAX_EXPERIENCE = 3
MAX_PROJECTS = 4
MIN_PROJECTS = 4
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
    matches = sum(1 for kw in keywords if keyword_in_text(
        kw, jd_lower, jd_text))
    return 0.06 * math.log2(1 + matches) if matches else 0.0


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
    summary_path = index_dir / "summary.yaml"
    if summary_path.exists():
        with open(summary_path) as f:
            summary_data = yaml.safe_load(f)
    else:
        summary_data = None
    return skills_data, projects_data, exp_data, certs_data, summary_data


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


def score_against_jd_multi(jd_embeddings, item_embeddings, jd_weights=None):
    """Fixed 60/40 metadata-anchored scoring.

    metadata_score (idx 0) always contributes 60% of the semantic score,
    mean of bullet scores contributes 40%. This keeps the overall project
    relevance dominant regardless of how many bullets an item has.
    """
    if jd_weights is None:
        jd_weights = np.ones(len(jd_embeddings))

    def _max_sim(emb):
        sims = np.array([cosine_sim(jd_emb, emb) for jd_emb in jd_embeddings])
        return float(np.max(sims * jd_weights))

    meta_score = _max_sim(item_embeddings[0])
    if len(item_embeddings) == 1:
        return meta_score
    bullet_scores = [_max_sim(emb) for emb in item_embeddings[1:]]
    return 0.6 * meta_score + 0.4 * float(np.mean(bullet_scores))


def build_item_vectors(item, kind, model):
    """Encode item as multiple vectors: metadata text + each bullet separately."""
    if kind == "project":
        meta = build_project_text(item)
        bullets = item.get("responsibilities", []) + item.get("impact", [])
    else:  # role
        meta = build_role_text(item)
        bullets = item.get("bullets", [])
    texts = [meta] + [b for b in bullets if b.strip()]
    return model.encode(texts)


def compute_jd_weights(chunks):
    """Return per-chunk importance weights. Requirement-like sections get 1.5x."""
    req_pattern = re.compile(
        r"requirement|qualification|responsibilities|what you.ll|must have|skills",
        re.IGNORECASE,
    )
    weights = []
    in_req_section = False
    for chunk in chunks:
        is_header = (
            chunk.endswith(":") or chunk.startswith("#")
            or (len(chunk.split()) <= 6 and chunk[0:1].isupper())
        )
        if is_header and req_pattern.search(chunk):
            in_req_section = True
        elif is_header and not req_pattern.search(chunk):
            in_req_section = False
        weights.append(1.5 if in_req_section else 1.0)
    return np.array(weights)


# ── Embedding cache ───────────────────────────────────────────────────────────
def _yaml_hash(data):
    """SHA-256 hash of YAML-serializable data for cache keying."""
    raw = yaml.dump(data, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _text_hash(text):
    """SHA-256 hash of a string."""
    return hashlib.sha256(text.encode()).hexdigest()


def load_cached_embeddings(cache_dir, key):
    """Load cached embeddings if they exist on disk."""
    path = cache_dir / f"{key}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_cached_embeddings(cache_dir, key, embeddings):
    """Save embeddings to disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{key}.npy", embeddings)


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
    "Lead", "Leadership", "Build", "Drive", "Create", "Develop", "Design", "Manage",
    "Support", "Ensure", "Provide", "Maintain", "Implement", "Deliver",
    "Collaborate", "Contribute", "Communicate", "Analyse", "Analyze",
    "Review", "Apply", "Experience", "Ability", "Skills", "Team",
    "Role", "Position", "Company", "Location", "Singapore", "Remote",
    "Hybrid", "Office", "Department", "Level", "Senior", "Junior",
    "Principal", "Staff", "Intern", "Associate", "Manager", "Director",
    "Engineer", "Developer", "Analyst", "Scientist", "Architect",
    "Consultant", "Specialist", "Coordinator", "Administrator",
    # Process & methodology
    "Application", "Process", "Requirements", "Responsibilities",
    "Technical", "Knowledge", "Domain", "Modelling", "Modeling",
    "Methodology", "Practice", "Practices", "Approach", "Framework",
    "Integration", "Delivery", "Continuous", "Driven", "Oriented",
    # Action verbs & gerunds
    "Championing", "Collaborating", "Empathising", "Empathizing",
    "Understanding", "Understand", "Writing", "Learning", "Applying",
    "Developing", "Building", "Solving", "Working", "Using",
    "Including", "Following", "Conducting", "Performing", "Executing",
    # Recruitment & JD boilerplate
    "Interview", "Resume", "Call", "Round", "Mandatory", "Introductory",
    "Exercise", "Candidate", "Candidates", "Attachment", "Period",
    "Gauge", "Proficiency", "Exposure", "Gain", "Gained",
    "Experiences", "Proactive", "Pre", "Minimally",
    # Misc generic
    "PDF", "Roles", "Certain", "Various", "Different", "Least",
    "Appropriate", "Meaningful", "Challenging", "Complex",
    "Alongside", "Order", "Right", "Industry", "Leading",
    "Business", "Stakeholders", "Products", "Solutions", "Value",
}


def expand_jd_with_llm(jd_text, model_name):
    """Call ollama REST API to expand JD text into implied keywords/technologies."""
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    prompt = (
        "Given this job description, output a JSON array of 10-25 specific technical "
        "keywords that are strongly implied but not explicitly stated. "
        "Include ONLY: programming languages, frameworks, libraries, tools, platforms, "
        "protocols, and concrete technical concepts. "
        "Exclude: soft skills, generic terms (e.g. 'Testing', 'Documentation', "
        "'Communication', 'Problem Solving', 'Leadership'), and job titles. "
        "Output ONLY the JSON array.\n\n"
        f"Job description:\n{jd_text}"
    )
    payload = json.dumps(
        {"model": model_name, "prompt": prompt, "stream": False,
         "options": {"num_predict": 512},
         "think": False}).encode()
    try:
        req = Request("http://localhost:11434/api/generate", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=300) as resp:
            raw = json.loads(resp.read())["response"]
        # Extract JSON array (model may wrap it in markdown fences or thinking tags)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            print(f"  Warning: could not parse LLM response as JSON array",
                  file=sys.stderr)
            return ""
        terms = json.loads(match.group())
        if not isinstance(terms, list):
            return ""
        terms = [str(t) for t in terms if isinstance(t, str)]
        return "\n".join(terms)
    except URLError:
        print("  Warning: cannot reach ollama at localhost:11434 — is it running?", file=sys.stderr)
        return ""
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  Warning: failed to parse LLM response: {e}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  Warning: LLM expansion failed: {e}", file=sys.stderr)
        return ""


def detect_coverage_gaps(jd_text, skills_data, projects_data, exp_data, certs_data,
                         extra_known=None):
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

    # Split compound terms so "CI/CD" also marks "CI" and "CD" as known
    known_words = set()
    for term in known:
        for word in re.split(r"[\s/\-]+", term):
            if len(word) > 1:
                known_words.add(word)
    known.update(known_words)

    # Extra known terms (e.g. company/role names from CLI args)
    if extra_known:
        for term in extra_known:
            known.add(term.lower())
            for word in re.split(r"[\s/\-]+", term.lower()):
                if len(word) > 1:
                    known.add(word)

    candidates = set()
    candidates.update(re.findall(r"\b([A-Z][A-Z0-9]{1,5})\b", jd_text))
    candidates.update(re.findall(
        r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", jd_text))
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


def render_resume(company, role, profile, education, experience, projects, skill_lines, certifications, summary=None):
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
        summary=summary,
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
    fig.suptitle(f"Match Analysis: {company} – {role}",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.subplots_adjust(hspace=0.35, wspace=0.45, top=0.93,
                        bottom=0.04, left=0.18, right=0.96)

    from matplotlib.patches import Patch

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
    parser.add_argument("--llm", nargs="?", const="qwen3.5:9b", default=None,
                        help="Expand JD keywords via ollama LLM (default model: qwen3.5:9b)")
    parser.add_argument("--rerank", action="store_true", default=False,
                        help="Re-score top candidates with a cross-encoder for precision")
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
        expanded = expand_jd_with_llm(jd_text, args.llm)
        if expanded:
            llm_terms = [t.strip() for t in expanded.split("\n") if t.strip()]
            print(f"  LLM expanded terms: {', '.join(llm_terms)}")
            jd_text = jd_text + "\n" + expanded
            jd_lower = jd_text.lower()

    # ── Load model & embed ──
    print("Loading embedding model (first run downloads ~22MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

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
        cache_key = f"role_{r['id']}_{_yaml_hash(r)[:16]}_mv"
        cached = load_cached_embeddings(emb_cache_dir, cache_key)
        if cached is not None:
            r_embeds = cached
        else:
            r_embeds = build_item_vectors(r, "role", model)
            save_cached_embeddings(emb_cache_dir, cache_key, r_embeds)
        sem = score_against_jd_multi(jd_embeddings, r_embeds, jd_weights)
        kw = compute_keyword_bonus(r.get("ats_tags", []), jd_lower, jd_text)
        tag = compute_tag_overlap_bonus(r.get("ats_tags", []), jd_lower, jd_text)
        role_scores.append((sem + kw + tag) * recency_multiplier(r["period"]))

    proj_scores = []
    for p in projects:
        cache_key = f"proj_{p['id']}_{_yaml_hash(p)[:16]}_mv"
        cached = load_cached_embeddings(emb_cache_dir, cache_key)
        if cached is not None:
            p_embeds = cached
        else:
            p_embeds = build_item_vectors(p, "project", model)
            save_cached_embeddings(emb_cache_dir, cache_key, p_embeds)
        sem = score_against_jd_multi(jd_embeddings, p_embeds, jd_weights)
        kw = compute_keyword_bonus(p.get("ats_tags", []), jd_lower, jd_text)
        tag = compute_tag_overlap_bonus(p.get("ats_tags", []), jd_lower, jd_text)
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
        kw = compute_keyword_bonus(c.get("ats_keywords", []), jd_lower, jd_text)
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
                        evidence_bonus[ref] = evidence_bonus.get(ref, 0) + bonus
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
        ce_pairs = [(jd_text, build_project_text(projects[i])) for i in ranked_indices]
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
    max_lines = 72

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
