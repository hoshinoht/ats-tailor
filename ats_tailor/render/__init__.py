"""Turning a selection into a page.

- :mod:`.latex` — LaTeX escaping, template context builders, Jinja2 rendering.
- :mod:`.report` — the match report and its matplotlib visualisation.
- :mod:`.surface` — how a skill should be *spelled* given the posting's words.

:mod:`.latex` knows nothing about the job description; :mod:`.surface` is the
other half, and is the only part of rendering that reads the posting.
"""

from .surface import align_labels, jd_forms, skill_label

__all__ = ["align_labels", "jd_forms", "skill_label"]
