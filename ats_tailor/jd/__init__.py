"""Everything that reads a posting: gates, extraction, and the profile itself.

The posting is *data* — never an instruction, whoever or whatever is reading it
(``docs/v2-design.md`` §7). That rule is enforced in three places: the prompt's
trust boundary, pydantic validation of whatever comes back, and the
deterministic parser that runs whether a model did or not.
"""

from .extract import (
    deterministic_profile,
    detect_archetype,
    detect_title,
    extract_requirements,
    index_vocabulary,
    normalise_term,
    parse_profile,
    render_extract_prompt,
    save_edited,
    term_alias_map,
)
from .gates import cross_check, detect_gates, gate_dict
from .hashing import jd_hash
from .models import Gate, Gates, JDProfile, RequirementSpec
from .requirements import REQUIREMENT_VERB_RE, requirements_from_jd

__all__ = [
    "Gate",
    "Gates",
    "JDProfile",
    "REQUIREMENT_VERB_RE",
    "RequirementSpec",
    "cross_check",
    "deterministic_profile",
    "detect_archetype",
    "detect_gates",
    "detect_title",
    "extract_requirements",
    "gate_dict",
    "index_vocabulary",
    "jd_hash",
    "normalise_term",
    "parse_profile",
    "render_extract_prompt",
    "requirements_from_jd",
    "save_edited",
    "term_alias_map",
]
