"""Everything that reads a posting: requirement extraction today, gates later."""

from .requirements import REQUIREMENT_VERB_RE, requirements_from_jd

__all__ = ["REQUIREMENT_VERB_RE", "requirements_from_jd"]
