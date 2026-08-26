"""LaTeX escape, template context builders, and Jinja2 rendering."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..config import MAX_EXP_BULLETS, MAX_PROJECT_BULLETS, MIN_PROJECT_BULLETS

#: Where the Jinja2 templates live — the ``render`` package, next to this module.
PKG_DIR = Path(__file__).resolve().parent


def latex_escape(text):
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        return text
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_href_escape(text):
    """Escape strings used inside LaTeX href targets."""
    return latex_escape(text)


def _dedupe_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        key = item.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def build_project_context(proj):
    """Build template context for a project."""
    tech_parts = []
    for key in ("languages", "frameworks", "infrastructure", "patterns", "tools"):
        tech_parts.extend(proj.get("tech_stack", {}).get(key, []))
    tech_line = ", ".join(tech_parts[:8])

    responsibilities = proj.get("responsibilities", [])
    impact = proj.get("impact", [])
    if impact and MAX_PROJECT_BULLETS == 1:
        bullets = impact[:1]
    elif impact:
        bullets = responsibilities[:max(MAX_PROJECT_BULLETS - 1, 0)] + impact[:1]
    else:
        bullets = responsibilities[:MAX_PROJECT_BULLETS]

    # Fill any remaining budget, then pad to the configured minimum if possible.
    bullets = _dedupe_keep_order(bullets)
    for candidate in responsibilities + impact:
        if len(bullets) >= MAX_PROJECT_BULLETS:
            break
        bullets = _dedupe_keep_order(bullets + [candidate])
    if len(bullets) < MIN_PROJECT_BULLETS:
        for candidate in impact + responsibilities:
            if len(bullets) >= MIN_PROJECT_BULLETS:
                break
            bullets = _dedupe_keep_order(bullets + [candidate])

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
    bullets = role.get("bullets", [])[:MAX_EXP_BULLETS]
    return {
        "company": role["company"],
        "title": role["title"],
        "period": role["period"],
        "bullets": bullets,
    }


def render_resume(company, role, profile, education, experience, projects, skill_lines,
                  certifications, summary=None, target_title=None):
    """Render the LaTeX resume from the Jinja2 template.

    ``target_title`` appends an honest target statement to the summary
    (``Targeting: <title>.``), which is how the JD's job title earns its one
    appearance on the page — see ``docs/v2-design.md`` §2. It is appended once:
    a summary that already carries the statement (a variant that names the role,
    a re-render inside the page-fit loop) is left alone, because the same words
    twice is the stuffing the ATS check warns about.
    """
    if target_title and summary and summary.get("summary"):
        existing = str(summary["summary"]).rstrip()
        statement = f"Targeting: {target_title}."
        if statement.lower() not in existing.lower():
            summary = {**summary, "summary": f"{existing} {statement}"}
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
    env.filters["latex_href_escape"] = latex_href_escape

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
