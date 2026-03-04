"""User-configurable constants. Override via .env file or environment variables.

Loads .env from the ats-tailor repo root (parent of this package directory).
Environment variables take precedence over .env file values.
"""

import os
from pathlib import Path

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
    return int(v) if v is not None else default


def _float(key, default):
    v = os.environ.get(key)
    return float(v) if v is not None else default


def _str(key, default):
    return os.environ.get(key, default)


# Embedding & LLM
EMBED_MODEL = _str("ATS_EMBED_MODEL", "all-MiniLM-L12-v2")
LLM_MODEL = os.environ.get("ATS_LLM_MODEL")  # None = disabled
LLM_BACKEND = _str("ATS_LLM_BACKEND", "auto")  # auto | mlx | lmstudio | ollama
MLX_MODEL = _str("ATS_MLX_MODEL", "mlx-community/Qwen3-8B-4bit")
LMSTUDIO_URL = _str("ATS_LMSTUDIO_URL", "http://localhost:1234")
LLM_NUM_CTX = _int("ATS_LLM_NUM_CTX", 4096)
LLM_TEMPERATURE = _float("ATS_LLM_TEMPERATURE", 0.3)
LLM_TEMPERATURE_P1 = _float("ATS_LLM_TEMPERATURE_P1", 0.2)
LLM_TEMPERATURE_P2 = _float("ATS_LLM_TEMPERATURE_P2", 0.6)
LLM_TWO_PASS = os.environ.get("ATS_LLM_TWO_PASS", "").lower() in ("1", "true", "yes")
RERANK = os.environ.get("ATS_RERANK", "").lower() in ("1", "true", "yes")
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
CERT_THRESHOLD = _float("ATS_CERT_THRESHOLD", 0.20)
CHARS_PER_BULLET_LINE = _int("ATS_CHARS_PER_BULLET_LINE", 80)
MAX_PAGE_LINES = _int("ATS_MAX_PAGE_LINES", 72)
