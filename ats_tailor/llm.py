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


def _ollama_generate(model_name, prompt, label="LLM", temperature=None):
    """Send a single prompt to ollama and return the raw response text."""
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    opts = {"num_predict": 512, "num_ctx": config.LLM_NUM_CTX}
    if temperature is not None:
        opts["temperature"] = temperature
    payload = json.dumps(
        {"model": model_name, "prompt": prompt, "stream": False,
         "options": opts, "think": False}).encode()
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


# Module-level cache for MLX model/tokenizer (avoid reloading between passes)
_mlx_model_cache = {}


def _mlx_generate(model_name, prompt, label="LLM", temperature=None):
    """Generate text using mlx-lm. Raises ImportError if mlx-lm is not installed."""
    import mlx_lm

    if model_name not in _mlx_model_cache:
        print(f"  Loading MLX model {model_name}...", file=sys.stderr)
        model, tokenizer = mlx_lm.load(model_name)
        _mlx_model_cache[model_name] = (model, tokenizer)

    model, tokenizer = _mlx_model_cache[model_name]
    kwargs = {"prompt": prompt, "max_tokens": 512, "verbose": False}
    if temperature is not None:
        kwargs["temp"] = temperature
    response = mlx_lm.generate(model, tokenizer, **kwargs)
    # Estimate token counts for logging
    p_tok = len(tokenizer.encode(prompt))
    g_tok = len(tokenizer.encode(response))
    print(f"  {label} tokens: {p_tok} prompt + {g_tok} generated",
          file=sys.stderr)
    return response


def _lmstudio_generate(model_name, prompt, label="LLM", temperature=None):
    """Send a prompt to LM Studio's OpenAI-compatible API."""
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048, "temperature": temperature if temperature is not None else 0.3}
    if model_name:
        body["model"] = model_name
    payload = json.dumps(body).encode()
    url = f"{config.LMSTUDIO_URL}/v1/chat/completions"
    req = Request(url, data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
        msg = data["choices"][0]["message"]
        # Prefer separated content (LM Studio reasoning_content setting)
        raw = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not raw and not reasoning:
            raw = ""
        elif not raw and reasoning:
            # reasoning_content not separated — content has everything
            pass
        # Strip <think>…</think> if content wasn't separated
        if "<think>" in raw:
            if "</think>" in raw:
                raw = re.sub(r"<think>.*?</think>\s*",
                             "", raw, flags=re.DOTALL)
            else:
                raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        usage = data.get("usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        g_tok = usage.get("completion_tokens", 0)
        thinking_note = f" ({g_tok - len(raw.split())} thinking)" if reasoning else ""
        print(f"  {label} tokens: {p_tok} prompt + {g_tok} generated{thinking_note}",
              file=sys.stderr)
    return raw


def _generate(model_name, prompt, label="LLM", temperature=None):
    """Dispatch to MLX, LM Studio, or Ollama based on config.LLM_BACKEND."""
    backend = config.LLM_BACKEND

    if backend == "mlx":
        return _mlx_generate(config.MLX_MODEL, prompt, label, temperature)
    elif backend == "lmstudio":
        return _lmstudio_generate(model_name, prompt, label, temperature)
    elif backend == "ollama":
        return _ollama_generate(model_name, prompt, label, temperature)
    else:  # auto
        from urllib.error import URLError
        try:
            return _mlx_generate(config.MLX_MODEL, prompt, label, temperature)
        except ImportError:
            print("  MLX not available, trying LM Studio",
                  file=sys.stderr)
        except Exception as e:
            print(f"  MLX failed ({e}), trying LM Studio",
                  file=sys.stderr)
        try:
            return _lmstudio_generate(model_name, prompt, label, temperature)
        except (URLError, OSError):
            print("  LM Studio not available, falling back to Ollama",
                  file=sys.stderr)
        except Exception as e:
            print(f"  LM Studio failed ({e}), falling back to Ollama",
                  file=sys.stderr)
        return _ollama_generate(model_name, prompt, label, temperature)


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
