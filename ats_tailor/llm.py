"""LLM keyword expansion and coverage gap detection."""

import json
import re
import sys

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
        with urlopen(req, timeout=600) as resp:
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
