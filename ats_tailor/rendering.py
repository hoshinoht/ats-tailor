"""LaTeX escape, template context builders, and Jinja2 rendering."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .config import MAX_EXP_BULLETS, MAX_PROJECT_BULLETS

PKG_DIR = Path(__file__).resolve().parent


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
