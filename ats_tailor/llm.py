"""LLM keyword expansion and coverage gap detection."""

import json
import re
import sys

from . import config

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


def _ollama_generate(model_name, prompt, label="LLM"):
    """Send a single prompt to ollama and return the raw response text."""
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    payload = json.dumps(
        {"model": model_name, "prompt": prompt, "stream": False,
         "options": {"num_predict": 512, "num_ctx": config.LLM_NUM_CTX},
         "think": False}).encode()
    req = Request("http://localhost:11434/api/generate", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
        raw = data["response"]
        p_tok = data.get("prompt_eval_count", 0)
        g_tok = data.get("eval_count", 0)
        print(f"  {label} tokens: {p_tok} prompt + {g_tok} generated "
              f"= {p_tok + g_tok} / {config.LLM_NUM_CTX} ctx",
              file=sys.stderr)
    return raw


def _parse_json_array(raw):
    """Extract a JSON array from LLM output (may be wrapped in fences/tags)."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        terms = json.loads(match.group())
        if isinstance(terms, list):
            return [str(t) for t in terms if isinstance(t, str)]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def expand_jd_with_llm(jd_text, model_name, categories=None):
    """Call ollama REST API to expand JD text into implied keywords/technologies.

    When categories are provided, they are included in the prompt to bias
    expansion toward the candidate's actual skill areas.

    When ATS_LLM_TWO_PASS=true, uses a two-pass strategy: first infers
    relevant domains, then expands keywords scoped to those domains.
    """
    from urllib.error import URLError

    try:
        if categories and config.LLM_TWO_PASS:
            return _expand_two_pass(jd_text, model_name, categories)
        return _expand_single_pass(jd_text, model_name, categories)
    except URLError:
        print("  Warning: cannot reach ollama at localhost:11434 — is it running?",
              file=sys.stderr)
        return ""
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  Warning: failed to parse LLM response: {e}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  Warning: LLM expansion failed: {e}", file=sys.stderr)
        return ""


def _expand_single_pass(jd_text, model_name, categories=None):
    """Single-pass expansion with optional category hint."""
    cat_hint = ""
    if categories:
        cat_hint = (
            "The candidate has skills in these areas: "
            f"{', '.join(categories)}. "
            "Prioritize keywords relevant to these domains, but also include "
            "important terms outside them.\n\n"
        )

    prompt = (
        f"{cat_hint}"
        "Given this job description, output a JSON array of 10-25 specific technical "
        "keywords that are strongly implied but not explicitly stated. "
        "Include ONLY: programming languages, frameworks, libraries, tools, platforms, "
        "protocols, and concrete technical concepts. "
        "Exclude: soft skills, generic terms (e.g. 'Testing', 'Documentation', "
        "'Communication', 'Problem Solving', 'Leadership'), and job titles. "
        "Output ONLY the JSON array.\n\n"
        f"Job description:\n{jd_text}"
    )
    raw = _ollama_generate(model_name, prompt, label="LLM")
    terms = _parse_json_array(raw)
    if not terms:
        print("  Warning: could not parse LLM response as JSON array",
              file=sys.stderr)
        return ""
    return "\n".join(terms)


def _expand_two_pass(jd_text, model_name, categories):
    """Two-pass domain-aware expansion.

    Pass 1: Infer 3-5 relevant domains from the candidate's skill categories.
    Pass 2: Expand keywords scoped to those domains + an 'Other' bucket.
    Falls back to single-pass on failure.
    """
    cat_list = ", ".join(categories)

    # ── Pass 1: domain inference ──
    p1_prompt = (
        "A candidate has skills in these technical domains:\n"
        f"{cat_list}\n\n"
        "Given this job description, output a JSON array of the 3-5 most relevant "
        "domains from the list above. Output ONLY the JSON array.\n\n"
        f"Job description:\n{jd_text}"
    )
    raw1 = _ollama_generate(model_name, p1_prompt, label="Pass 1 (domains)")
    domains = _parse_json_array(raw1)
    if not domains:
        print("  Warning: pass 1 failed, falling back to single-pass",
              file=sys.stderr)
        return _expand_single_pass(jd_text, model_name, categories)

    valid = {c.lower() for c in categories}
    domains = [d for d in domains if d.lower() in valid]
    if not domains:
        print("  Warning: pass 1 returned no valid domains, falling back",
              file=sys.stderr)
        return _expand_single_pass(jd_text, model_name, categories)

    print(f"  Inferred domains: {', '.join(domains)}", file=sys.stderr)

    # ── Pass 2: domain-scoped expansion ──
    domain_list = ", ".join(domains)
    p2_prompt = (
        f"The candidate's relevant skill domains are: {domain_list}\n\n"
        "Given this job description, output a JSON object where each key is one of "
        "the domains above (plus an \"Other\" key for anything outside those domains). "
        "Each value is an array of 3-8 specific technical keywords implied but not "
        "explicitly stated in the JD for that domain. "
        "Include ONLY: programming languages, frameworks, libraries, tools, platforms, "
        "protocols, and concrete technical concepts. "
        "Exclude: soft skills, generic terms, and job titles. "
        "Output ONLY the JSON object.\n\n"
        f"Job description:\n{jd_text}"
    )
    raw2 = _ollama_generate(model_name, p2_prompt, label="Pass 2 (keywords)")

    match = re.search(r"\{.*\}", raw2, re.DOTALL)
    if not match:
        terms = _parse_json_array(raw2)
        if terms:
            return "\n".join(terms)
        print("  Warning: pass 2 failed, falling back to single-pass",
              file=sys.stderr)
        return _expand_single_pass(jd_text, model_name, categories)

    obj = json.loads(match.group())
    if not isinstance(obj, dict):
        return _expand_single_pass(jd_text, model_name, categories)

    all_terms = []
    for domain, terms in obj.items():
        if isinstance(terms, list):
            domain_terms = [str(t) for t in terms if isinstance(t, str)]
            if domain_terms:
                print(f"    {domain}: {', '.join(domain_terms)}", file=sys.stderr)
                all_terms.extend(domain_terms)

    if not all_terms:
        return _expand_single_pass(jd_text, model_name, categories)
    return "\n".join(all_terms)


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
