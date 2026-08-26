"""Rendering decisions that depend on the *posting*, not just the index.

:mod:`ats_tailor.rendering` turns a selection into LaTeX and knows nothing about
the job description. This package is the other half: how a skill should be
*spelled* on the page given the words the posting used.
"""

from .surface import align_labels, jd_forms, skill_label

__all__ = ["align_labels", "jd_forms", "skill_label"]
