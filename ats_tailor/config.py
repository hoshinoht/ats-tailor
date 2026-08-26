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
LLM_NUM_CTX = _int("ATS_LLM_NUM_CTX", 4096)
LLM_TEMPERATURE = _float("ATS_LLM_TEMPERATURE", 0.3)
LLM_TEMPERATURE_P1 = _float("ATS_LLM_TEMPERATURE_P1", 0.2)
LLM_TEMPERATURE_P2 = _float("ATS_LLM_TEMPERATURE_P2", 0.5)
LLM_TWO_PASS = _bool("ATS_LLM_TWO_PASS")
RERANK = _bool("ATS_RERANK")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

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
    if SECTION_PRIORITY not in {"experience", "projects"}:
        errors.append("ATS_SECTION_PRIORITY must be 'experience' or 'projects'")

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

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))
