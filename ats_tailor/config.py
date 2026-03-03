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


def _str(key, default):
    return os.environ.get(key, default)


# Embedding & LLM
EMBED_MODEL = _str("ATS_EMBED_MODEL", "all-MiniLM-L12-v2")
LLM_MODEL = os.environ.get("ATS_LLM_MODEL")  # None = disabled
LLM_NUM_CTX = _int("ATS_LLM_NUM_CTX", 4096)
LLM_TWO_PASS = os.environ.get("ATS_LLM_TWO_PASS", "").lower() in ("1", "true", "yes")
RERANK = os.environ.get("ATS_RERANK", "").lower() in ("1", "true", "yes")

# Page budget
MAX_EXPERIENCE = _int("ATS_MAX_EXPERIENCE", 3)
MAX_PROJECTS = _int("ATS_MAX_PROJECTS", 4)
MIN_PROJECTS = _int("ATS_MIN_PROJECTS", 4)
MAX_SKILL_LINES = _int("ATS_MAX_SKILL_LINES", 4)
MAX_PROJECT_BULLETS = _int("ATS_MAX_PROJECT_BULLETS", 3)
MAX_EXP_BULLETS = _int("ATS_MAX_EXP_BULLETS", 3)
CHARS_PER_BULLET_LINE = _int("ATS_CHARS_PER_BULLET_LINE", 80)
MAX_PAGE_LINES = _int("ATS_MAX_PAGE_LINES", 72)
