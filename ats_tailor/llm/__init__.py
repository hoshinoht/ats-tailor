"""Everything that reaches a model, behind one import site.

- :mod:`.backends` — Ollama, LM Studio and MLX, plus the health probes.
- :mod:`.expand` — v1's keyword expansion, re-exported here unchanged so
  ``from ats_tailor.llm import expand_jd_with_llm`` keeps working.
- :mod:`.provider` — the v2 :class:`Provider` interface and ``get_provider``.
"""

from .expand import detect_coverage_gaps, expand_jd_with_llm
from .provider import (
    ExternalProvider,
    LMStudioProvider,
    MLXProvider,
    NoneProvider,
    OllamaProvider,
    Provider,
    get_provider,
)

__all__ = [
    "ExternalProvider",
    "LMStudioProvider",
    "MLXProvider",
    "NoneProvider",
    "OllamaProvider",
    "Provider",
    "detect_coverage_gaps",
    "expand_jd_with_llm",
    "get_provider",
]
