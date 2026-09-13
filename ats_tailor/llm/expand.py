"""v1's keyword expansion, and the coverage-gap detector both engines share.

``--llm`` on the v1 engine means what it has always meant: ask a local model
for technologies a posting implies but never spells out, and append them to the
JD text before scoring. The v2 engine gives ``--llm`` a different meaning (use
the configured provider for *requirement extraction*), which is why the two
live in separate modules rather than in one function with a flag.

Re-exported from :mod:`ats_tailor.llm` so every existing import site is
unchanged.
"""

import json
import re
import sys
from urllib.error import URLError

from .. import config
from .backends import _generate, _parse_json_array

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


#: Function words that stay lowercase in ordinary prose. A *capitalised* one
#: partway through a segment is the tell that a posting is Title-Casing a
#: sentence — "You Won't Be Confined To a Single Layer Of The Stack" — and in
#: such a segment every capital is English, not a product. A plain
#: capitalised-ratio rule would say the same about "Red Hat OpenShift", which
#: is why the tell is a function word wearing a capital and nothing else.
_TITLE_PROSE_MARKERS = frozenset(
    "A An The Of To At As In On For And Or But With By From Is Are Be Will "
    "Our Your You We It Its This That".split()
)

#: What a capitalised word is doing when it is *listed* rather than starting a
#: sentence: "- Terraform, Ansible", "Terraform (IaC)", "Go and Java". Matched
#: from the end of the candidate, so it never anchors at the line start.
_LIST_SHAPED_RE = re.compile(r"\s*(?:[,/(]|and\b|or\b)")

#: A posting quoting itself — a motto, a slogan. Capitals inside are emphasis:
#: "Digital to the Core, and Serve with Heart."
_QUOTED_RE = re.compile(r'"[^"\n]*"|\u201c[^\u201d\n]*\u201d')

#: A sentence ends here, closing punctuation and all.
_SENTENCE_END_RE = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\s+")

#: The first letter of a word — used to find where a sentence actually starts,
#: past a bullet glyph or leading whitespace.
_WORD_START_RE = re.compile(r"[^\W\d_]")
_WORD_RE = re.compile(r"[^\W\d_]+")

#: The shapes a gap candidate can take. ALL-CAPS acronyms, CamelCase names
#: and punctuated names are exempt from the positional rules that assume a
#: capital is a choice: an acronym at the start of a line is still an acronym,
#: and no amount of Title-Casing turns English into "OpenShift".
_CANDIDATE_SHAPES = (
    (re.compile(r"\b([A-Z][A-Z0-9]{1,5})\b"), True),
    (re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"), True),
    (re.compile(r"\b([A-Z][a-z]{2,})\b"), False),
    (re.compile(r"\b(C\+\+|C#|\.NET|F#)\b"), True),
)


def _label_end(line):
    """Where a leading ``Label:`` prefix ends, or 0 when the line has none.

    "Have a Security-First Mindset:", "Collaborate Across Disciplines:",
    "Job Title:" — a posting Title-Cases these by convention, so a capital
    inside one is evidence of nothing.
    """
    colon = line.find(":")
    if colon < 0:
        return 0
    prefix = line[:colon]
    if len(prefix.split()) > 6 or any(ch in prefix for ch in ".!?"):
        return 0
    return colon


def _sentence_starts(line, label_end):
    """Offsets where a capital is merely the first word of something."""
    starts = set()
    first = _WORD_START_RE.search(line)
    if first:
        starts.add(first.start())
    for match in _SENTENCE_END_RE.finditer(line):
        nxt = _WORD_START_RE.search(line, match.end())
        if nxt:
            starts.add(nxt.start())
    if label_end:
        nxt = _WORD_START_RE.search(line, label_end + 1)
        if nxt:
            starts.add(nxt.start())
    return starts


def _prose_spans(line):
    """Segments of the line that are Title-Cased English rather than names.

    Segments are delimited by commas and sentence punctuation, and a segment's
    own first word is exempt — "As part of the Singpass team" opens with a
    capitalised "As" and is still ordinary prose naming a real product.
    """
    spans = []
    start = 0
    boundaries = [m.start() for m in re.finditer(r"[,.;!?]", line)] + [len(line)]
    for boundary in boundaries:
        words = _WORD_RE.findall(line[start:boundary])
        if any(word in _TITLE_PROSE_MARKERS for word in words[1:]):
            spans.append((start, boundary))
        start = boundary + 1
    return spans


class _LineShape:
    """One posting line, pre-marked with the places a capital means nothing."""

    def __init__(self, line):
        self.line = line
        self.label_end = _label_end(line)
        self.quoted = [m.span() for m in _QUOTED_RE.finditer(line)]
        self.sentence_starts = _sentence_starts(line, self.label_end)
        self.prose = _prose_spans(line)

    def is_clean(self, start, end, exempt=False):
        """Is this occurrence of a capitalised word evidence of a real name?"""
        if start < self.label_end:
            return False
        if any(a <= start < b for a, b in self.quoted):
            return False
        if self.line[end:end + 1] in ("'", "\u2019"):
            return False
        if exempt:
            return True
        if (start in self.sentence_starts
                and not _LIST_SHAPED_RE.match(self.line, end)):
            return False
        return not any(a <= start < b for a, b in self.prose)


def expand_jd_with_llm(jd_text, model_name, categories=None):
    """Call ollama REST API to expand JD text into implied keywords/technologies.

    When categories are provided, they are included in the prompt to bias
    expansion toward the candidate's actual skill areas.

    When ATS_LLM_TWO_PASS=true, uses a two-pass strategy: first infers
    relevant domains, then expands keywords scoped to those domains.
    """
    try:
        if categories and config.LLM_TWO_PASS:
            return _expand_two_pass(jd_text, model_name, categories)
        return _expand_single_pass(jd_text, model_name, categories)
    except URLError:
        backend = config.LLM_BACKEND
        if backend == "lmstudio":
            hint = f"LM Studio at {config.LMSTUDIO_URL}"
        elif backend == "ollama":
            hint = "Ollama at localhost:11434"
        else:
            hint = "LLM backend (MLX/LM Studio/Ollama)"
        print(f"  Warning: cannot reach {hint} — is it running?",
              file=sys.stderr)
        return ""
    except ImportError:
        print("  Warning: LLM backend not available (missing mlx-lm or ollama)",
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

    print("Running single-pass LLM expansion...", file=sys.stderr)

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
    raw = _generate(model_name, prompt, label="LLM",
                    temperature=config.LLM_TEMPERATURE)
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

    print("  Running two-pass LLM expansion...", file=sys.stderr)
    # ── Pass 1: domain inference ──
    p1_prompt = (
        "A candidate has skills in these technical domains:\n"
        f"{cat_list}\n\n"
        "Given this job description, output a JSON array of the 3-5 most relevant "
        "domains from the list above. Output ONLY the JSON array.\n\n"
        f"Job description:\n{jd_text}"
    )
    raw1 = _generate(model_name, p1_prompt, label="Pass 1 (domains)",
                     temperature=config.LLM_TEMPERATURE_P1)
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
        "Each value is an array of 3-8 specific technical keywords that are strongly "
        "IMPLIED but NOT explicitly written in the job description. "
        "Do NOT repeat any term that already appears verbatim in the JD text. "
        "Focus on related tools, libraries, frameworks, and protocols that someone "
        "in this role would realistically use. "
        "Include ONLY: programming languages, frameworks, libraries, tools, platforms, "
        "protocols, and concrete technical concepts. "
        "Exclude: soft skills, generic terms, and job titles. "
        "Output ONLY the JSON object.\n\n"
        f"Job description:\n{jd_text}"
    )
    raw2 = _generate(model_name, p2_prompt, label="Pass 2 (keywords)",
                     temperature=config.LLM_TEMPERATURE_P2)

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
    seen = set()
    for domain, terms in obj.items():
        if isinstance(terms, list):
            domain_terms = [str(t) for t in terms if isinstance(t, str)]
            if domain_terms:
                print(f"    {domain}: {', '.join(domain_terms)}",
                      file=sys.stderr)
                for t in domain_terms:
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        all_terms.append(t)

    if not all_terms:
        return _expand_single_pass(jd_text, model_name, categories)
    return "\n".join(all_terms)


def detect_coverage_gaps(jd_text, skills_data, projects_data, exp_data, certs_data,
                         extra_known=None):
    """Find tech-looking terms in the JD that aren't in any index file.

    A term qualifies on shape *and* position. Shape alone — any capitalised
    word — is what a posting in Title Case ("Are Versatile Problem Solvers:")
    defeats, and the terms it invents do not stop at the report: they become
    weighted requirements. So a capital counts only where the posting chose it:
    outside a ``Label:`` prefix, not opening a sentence unless it is being
    listed, not inside a quoted motto, and not in a segment of Title-Cased
    prose. A word the posting also spells lowercase is English by its own
    admission. ALL-CAPS acronyms, CamelCase and punctuated names keep their
    positional freedom — an acronym at the start of a line is still an acronym,
    and no amount of Title-Casing turns an English word into "OpenShift".
    """
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

    # A word is a gap only where it is capitalised *by choice*. Harvesting on
    # shape alone reads a posting written in Title Case ("We View Our Engineers
    # As Holistic Problem Solvers") as three dozen technologies, and every one
    # of them goes on to become a weighted requirement.
    lowercase_words = set(re.findall(r"\b[a-z]{2,}\b", jd_text))
    clean = set()
    for line in jd_text.splitlines():
        shape = _LineShape(line)
        for pattern, exempt in _CANDIDATE_SHAPES:
            for match in pattern.finditer(line):
                word = match.group(1)
                if word in clean:
                    continue
                if shape.is_clean(match.start(1), match.end(1), exempt):
                    clean.add(word)

    return sorted(
        word for word in clean
        if word.lower() not in known
        and word not in _GAP_STOPWORDS
        # The posting spelling it lowercase somewhere settles it: an English
        # word that happened to be capitalised, not a name.
        and word.lower() not in lowercase_words
    )
