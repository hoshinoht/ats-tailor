"""User-configurable constants. Override via .env file or environment variables.

Loads .env from the ats-tailor repo root (parent of this package directory).
Environment variables take precedence over .env file values.
"""

import os
from pathlib import Path

_CONFIG_PARSE_ERRORS = []

# Load .env from ats-tailor repo root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _key, _val = _key.strip(), _val.strip()
            # Don't override existing env vars (CLI/shell exports win)
            if _key not in os.environ:
                os.environ[_key] = _val


def _int(key, default):
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        _CONFIG_PARSE_ERRORS.append((key, f"{key} must be an integer"))
        return default


def _float(key, default):
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        _CONFIG_PARSE_ERRORS.append((key, f"{key} must be a number"))
        return default


def _str(key, default):
    return os.environ.get(key, default)


def _rerank_mode(key, default="false"):
    """``ATS_RERANK`` as ``none`` | ``cross`` | ``llm``.

    It used to be a plain boolean and still reads as one; ``llm`` is the third
    value Phase 3c adds, so the setting cannot go through :func:`_bool` without
    reporting the user's own documented value as a parse error.
    """
    raw = str(os.environ.get(key, default) or "").strip().lower()
    if raw in ("1", "true", "yes", "on", "cross", "cross-encoder"):
        return "cross"
    if raw == "llm":
        return "llm"
    if raw in ("0", "false", "no", "off", "none", ""):
        return "none"
    _CONFIG_PARSE_ERRORS.append((
        key, f"{key} must be false, true (cross-encoder) or llm"))
    return "none"


def _bool(key, default=False):
    v = os.environ.get(key)
    if v is None:
        return default
    val = v.lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    _CONFIG_PARSE_ERRORS.append((
        key,
        f"{key} must be boolean: 1/true/yes/on or 0/false/no/off",
    ))
    return default


# Embedding & LLM
EMBED_MODEL = _str("ATS_EMBED_MODEL", "all-MiniLM-L12-v2")
LLM_EXPAND = _bool("ATS_LLM_EXPAND")
LLM_MODEL = _str("ATS_LLM_MODEL", "qwen3.5:4b")
LLM_BACKEND = _str("ATS_LLM_BACKEND", "auto")  # auto | mlx | lmstudio | ollama
MLX_MODEL = _str("ATS_MLX_MODEL", "mlx-community/Qwen3-8B-4bit")
LMSTUDIO_URL = _str("ATS_LMSTUDIO_URL", "http://localhost:1234")
OLLAMA_URL = _str("ATS_OLLAMA_URL", "http://localhost:11434")
LLM_NUM_CTX = _int("ATS_LLM_NUM_CTX", 4096)
LLM_TEMPERATURE = _float("ATS_LLM_TEMPERATURE", 0.3)
LLM_TEMPERATURE_P1 = _float("ATS_LLM_TEMPERATURE_P1", 0.2)
LLM_TEMPERATURE_P2 = _float("ATS_LLM_TEMPERATURE_P2", 0.5)
LLM_TWO_PASS = _bool("ATS_LLM_TWO_PASS")
#: Seconds a generation call may take. Long by design: a local 20B model on a
#: laptop answers a requirement-extraction prompt in minutes, not seconds.
LLM_TIMEOUT = _int("ATS_LLM_TIMEOUT", 600)
#: Seconds a *health probe* may take. Short by design, and unrelated to the
#: generation timeout: `auto` asks three hosts whether they are there at all,
#: and a laptop with nothing running must not pay 600s to find that out.
LLM_PROBE_TIMEOUT = _float("ATS_LLM_PROBE_TIMEOUT", 1.5)

# ── Providers (Phase 3c) ─────────────────────────────────────────────────────
#: Who answers `extract` / `rerank` / `propose_variant` for the v2 engine.
#: auto | ollama | lmstudio | mlx | none | external. `auto` probes mlx →
#: lmstudio → ollama and falls back to `none` when nothing answers.
LLM_PROVIDER = _str("ATS_LLM_PROVIDER", "auto")
PROVIDERS = ("auto", "ollama", "lmstudio", "mlx", "none", "external")
#: Bumped whenever `jd/prompt_extract.md.j2` changes meaning: it is part of the
#: extraction cache key, so an old profile is never served for a new prompt.
PROMPT_VERSION = _str("ATS_PROMPT_VERSION", "extract-1")
#: Requirements kept from one posting, by weight. A posting with more than this
#: many distinct asks is mostly prose about itself.
MAX_REQUIREMENTS = _int("ATS_MAX_REQUIREMENTS", 40)
#: Weight given to a deterministic requirement the extractor did not cover.
#: Half: it is a safety net, not a second opinion.
FALLBACK_REQ_WEIGHT = _float("ATS_FALLBACK_REQ_WEIGHT", 0.5)

# Reranking. "false" off, "true" the cross-encoder, "llm" the provider.
RERANK_MODE = _rerank_mode("ATS_RERANK", "false")
RERANK = RERANK_MODE != "none"
RERANK_MODEL = _str("ATS_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = _int("ATS_RERANK_TOP_K", 5)

# Selection engine: "v1" (item ranking) or "v2" (requirement coverage).
# v2 is the default from Phase 3c; `--engine v1` still runs the legacy path.
ENGINE = _str("ATS_ENGINE", "v2")
#: Append the posting's own job title to the Summary as a target statement
#: ("... Targeting: Site Reliability Engineer Intern."). v2 only.
TARGET_TITLE = _bool("ATS_TARGET_TITLE", True)
# v2 objective knobs (ats_match.Params): soft-pool temperature, redundancy
# penalty, keyword-in-context bonus.
SELECT_TAU = _float("ATS_TAU", 0.1)
SELECT_MU = _float("ATS_MU", 0.3)
SELECT_BETA = _float("ATS_BETA", 3.0)
SELECT_REDUNDANCY_IN_SKILLS = _bool("ATS_REDUNDANCY_IN_SKILLS", False)
SELECT_BREADTH_FIRST = _bool("ATS_BREADTH_FIRST", True)
# Weight of a requirement that names nothing the index or the posting's own
# vocabulary can be matched against ("Works well independently and with others").
SOFT_REQ_WEIGHT = _float("ATS_SOFT_REQ_WEIGHT", 0.25)
# Gain a skill must earn to be listed. Skills cost no lines, so without a floor
# a skills line fills up with whatever has the highest proficiency prior.
# Measured over the labelled JDs: 0.0-0.01 removes nothing, 0.02-0.03 removes
# the tail without touching coverage, 0.05 starts deleting skills the posting
# actually asked for (must 75.0% -> 74.1%, context 51.4% -> 43.2%).
SKILL_MIN_GAIN = _float("ATS_SKILL_MIN_GAIN", 0.03)
# Feed each entry's header line (title/organisation, or name/tagline plus the
# Technologies line) to the matcher as a claim that arrives free with its group.
# Off by default, on evidence: measured over the labelled JDs it costs 13.5
# points of context_rate (51.4% -> 37.9%) without recovering the must-term the
# idea was meant to recover. A header's coverage saturates the requirements a
# keyword-bearing bullet would otherwise have been picked for, and the keyword
# check does not read header lines as bullets. See docs/v2-design.md §8.
HEADER_CLAIMS = _bool("ATS_HEADER_CLAIMS", False)
# Skill categories that always get a line, whatever they score. v1 always
# rendered a Languages line and a resume without one reads as incomplete.
PIN_SKILL_GROUPS = _str("ATS_PIN_SKILL_GROUPS", "Programming Languages")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

# Runtime store (SQLite: embeddings, JD profiles, run records). Empty means
# "beside the index directory" — see ats_tailor.runs.store.data_dir.
DATA_DIR = _str("ATS_DATA_DIR", "")

# Page budget
MAX_EXPERIENCE = _int("ATS_MAX_EXPERIENCE", 3)
MAX_PROJECTS = _int("ATS_MAX_PROJECTS", 4)
MIN_PROJECTS = _int("ATS_MIN_PROJECTS", 2)
MIN_EXPERIENCE = _int("ATS_MIN_EXPERIENCE", 2)
MAX_SKILL_LINES = _int("ATS_MAX_SKILL_LINES", 4)
MAX_PROJECT_BULLETS = _int("ATS_MAX_PROJECT_BULLETS", 3)
MIN_PROJECT_BULLETS = _int("ATS_MIN_PROJECT_BULLETS", 2)
MAX_EXP_BULLETS = _int("ATS_MAX_EXP_BULLETS", 3)
MIN_EXP_BULLETS = _int("ATS_MIN_EXP_BULLETS", 2)
# "experience" or "projects" — the higher-priority section is trimmed/dropped last
SECTION_PRIORITY = _str("ATS_SECTION_PRIORITY", "experience")
MIN_CERTIFICATIONS = _int("ATS_MIN_CERTIFICATIONS", 3)
MAX_CERTIFICATIONS = _int("ATS_MAX_CERTIFICATIONS", 3)
CERT_THRESHOLD = _float("ATS_CERT_THRESHOLD", 0.20)
CHARS_PER_BULLET_LINE = _int("ATS_CHARS_PER_BULLET_LINE", 80)
MAX_PAGE_LINES = _int("ATS_MAX_PAGE_LINES", 72)


def validate_config(ignore_parse_keys=None):
    """Validate environment-derived configuration before running the pipeline."""
    ignore_parse_keys = set(ignore_parse_keys or [])
    errors = [msg for key, msg in _CONFIG_PARSE_ERRORS if key not in ignore_parse_keys]

    if LLM_BACKEND not in {"auto", "mlx", "lmstudio", "ollama"}:
        errors.append("ATS_LLM_BACKEND must be one of: auto, mlx, lmstudio, ollama")
    if LLM_PROVIDER not in set(PROVIDERS):
        errors.append(f"ATS_LLM_PROVIDER must be one of: {', '.join(PROVIDERS)}")
    if not str(PROMPT_VERSION).strip():
        errors.append("ATS_PROMPT_VERSION must not be empty")
    if LLM_PROBE_TIMEOUT <= 0:
        errors.append("ATS_LLM_PROBE_TIMEOUT must be positive")
    if LLM_TIMEOUT <= 0:
        errors.append("ATS_LLM_TIMEOUT must be positive")
    if MAX_REQUIREMENTS <= 0:
        errors.append("ATS_MAX_REQUIREMENTS must be positive")
    if RERANK_TOP_K <= 0:
        errors.append("ATS_RERANK_TOP_K must be positive")
    if SECTION_PRIORITY not in {"experience", "projects"}:
        errors.append("ATS_SECTION_PRIORITY must be 'experience' or 'projects'")
    if ENGINE not in {"v1", "v2"}:
        errors.append("ATS_ENGINE must be 'v1' or 'v2'")
    if SELECT_TAU <= 0:
        errors.append("ATS_TAU must be positive")
    for name, value in (
        ("ATS_MU", SELECT_MU),
        ("ATS_BETA", SELECT_BETA),
        ("ATS_SOFT_REQ_WEIGHT", SOFT_REQ_WEIGHT),
        ("ATS_FALLBACK_REQ_WEIGHT", FALLBACK_REQ_WEIGHT),
        ("ATS_SKILL_MIN_GAIN", SKILL_MIN_GAIN),
    ):
        if value < 0:
            errors.append(f"{name} must be zero or positive")

    for name, value in [
        ("ATS_MAX_EXPERIENCE", MAX_EXPERIENCE),
        ("ATS_MAX_PROJECTS", MAX_PROJECTS),
        ("ATS_MAX_SKILL_LINES", MAX_SKILL_LINES),
        ("ATS_MAX_PROJECT_BULLETS", MAX_PROJECT_BULLETS),
        ("ATS_MAX_EXP_BULLETS", MAX_EXP_BULLETS),
        ("ATS_CHARS_PER_BULLET_LINE", CHARS_PER_BULLET_LINE),
        ("ATS_MAX_PAGE_LINES", MAX_PAGE_LINES),
        ("ATS_MAX_CERTIFICATIONS", MAX_CERTIFICATIONS),
    ]:
        if value <= 0:
            errors.append(f"{name} must be positive")

    for name, value in [
        ("ATS_MIN_PROJECTS", MIN_PROJECTS),
        ("ATS_MIN_EXPERIENCE", MIN_EXPERIENCE),
        ("ATS_MIN_PROJECT_BULLETS", MIN_PROJECT_BULLETS),
        ("ATS_MIN_EXP_BULLETS", MIN_EXP_BULLETS),
        ("ATS_MIN_CERTIFICATIONS", MIN_CERTIFICATIONS),
    ]:
        if value < 0:
            errors.append(f"{name} must be zero or positive")

    if MIN_PROJECTS > MAX_PROJECTS:
        errors.append("ATS_MIN_PROJECTS cannot exceed ATS_MAX_PROJECTS")
    if MIN_EXPERIENCE > MAX_EXPERIENCE:
        errors.append("ATS_MIN_EXPERIENCE cannot exceed ATS_MAX_EXPERIENCE")
    if MIN_PROJECT_BULLETS > MAX_PROJECT_BULLETS:
        errors.append("ATS_MIN_PROJECT_BULLETS cannot exceed ATS_MAX_PROJECT_BULLETS")
    if MIN_EXP_BULLETS > MAX_EXP_BULLETS:
        errors.append("ATS_MIN_EXP_BULLETS cannot exceed ATS_MAX_EXP_BULLETS")
    if MIN_CERTIFICATIONS > MAX_CERTIFICATIONS:
        errors.append("ATS_MIN_CERTIFICATIONS cannot exceed ATS_MAX_CERTIFICATIONS")
    if CERT_THRESHOLD < 0:
        errors.append("ATS_CERT_THRESHOLD must be zero or positive")

    # Only a typo is worth catching here: the directory is created on demand,
    # and a relative one is resolved against the working directory on purpose.
    if os.environ.get("ATS_DATA_DIR") is not None and not DATA_DIR.strip():
        errors.append("ATS_DATA_DIR must name a directory, or be left unset")
    elif DATA_DIR.strip() and Path(DATA_DIR).expanduser().exists() \
            and not Path(DATA_DIR).expanduser().is_dir():
        errors.append(f"ATS_DATA_DIR must be a directory, got {DATA_DIR!r}")

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))
