"""Spelling a skill the way the posting spells it.

An ATS matches strings. The index says ``Kubernetes``; a posting that only ever
writes ``K8s`` scores the resume as not mentioning it, and
``ats-tailor check`` reports the term as ``synonym-only`` — present through an
alias, absent in the words that were asked for. Fixing that is a *rendering*
decision, not a matching one: the claim is unchanged, only its label moves.

Two rules keep this honest:

- **Both forms, once.** Every aligned label carries the index's name *and* the
  posting's word — ``Kubernetes (K8s)``, ``Amazon Web Services (AWS)``,
  ``Docker (Containerization)`` — so neither the posting's reader nor its
  parser has to know they are the same thing, and the canonical technology
  keyword never leaves the page to make room for the posting's synonym. Which
  form leads is the only difference between the branches: the acronym goes in
  the brackets, because the expansion is what a human reads. Nothing is
  repeated a third time; ``ats-tailor check``'s stuffing warning is a real
  constraint.
- **Bullets are never touched.** A bullet is a sentence the user wrote about
  something they did. Rewording it to hit a keyword is Phase 5's problem, with
  a metric guardrail in front of it; this module rewrites labels on the skills
  line and nothing else.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from ..atscheck import count_in_text

#: An acronym-shaped form: short and upper-case, possibly with digits ("K8s"
#: counts — one lower-case letter inside an otherwise upper-case token).
MAX_ACRONYM_CHARS = 6


def is_acronym_like(form: str) -> bool:
    """Whether ``form`` reads as an acronym rather than a word.

    ``K8s`` and ``AWS`` do; ``Kubernetes`` and ``Amazon Web Services`` do not.
    The test is shape, not a dictionary: at most six characters, no spaces, and
    the letters that are not digits are mostly capitals.
    """
    text = str(form or "").strip()
    if not text or len(text) > MAX_ACRONYM_CHARS or " " in text:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return sum(1 for ch in letters if ch.isupper()) >= max(1, len(letters) - 1)


def names_it(name: str, form: str) -> bool:
    """Whether ``name`` already spells ``form`` out, as a whole word."""
    name, form = str(name or "").strip(), str(form or "").strip()
    if not name or not form:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(form)}(?!\w)", name, re.IGNORECASE))


def jd_forms(jd_text: str, index: Any) -> dict[str, str]:
    """``skill id → the surface form the posting used``.

    Most occurrences wins; a tie goes to the earlier form, which is the index's
    own order (name, then aliases, then acronym) — so a posting that mentions
    both spellings once is rendered in the canonical one and nothing changes.
    Skills the posting never names are absent from the result.
    """
    text = str(jd_text or "")
    lowered = text.lower()
    chosen: dict[str, str] = {}
    for skill in _skills(index):
        best_form, best_count = None, 0
        for form in skill.surface_forms():
            count = count_in_text(form, lowered, text)
            if count > best_count:
                best_form, best_count = form, count
        if best_form is not None:
            chosen[skill.id] = best_form
    return chosen


def skill_label(skill: Any, jd_form: str | None = None) -> str:
    """How this skill should be printed, given the posting's own wording."""
    name = str(getattr(skill, "name", "") or "").strip()
    form = str(jd_form or "").strip()
    if not form or form.lower() == name.lower():
        return name
    if names_it(name, form):
        # The name already contains the posting's word — "Git / Version
        # Control" says "Git", "LSTM / GRU" says "LSTM". Adding the bracket
        # would print the same token twice on one line, which is the stuffing
        # the alignment exists to avoid, not a second surface form.
        return name

    acronym = str(getattr(skill, "acronym", "") or "").strip()
    known = {f.strip().lower() for f in _forms(skill)}
    if form.lower() not in known:
        return name  # not a form of this skill at all; nothing to align to

    # The posting used the short form of a name the index writes out.
    if (acronym and form.lower() == acronym.lower()) or (
        is_acronym_like(form) and not is_acronym_like(name)
    ):
        return f"{name} ({form})"
    # The posting wrote out a name the index abbreviates.
    if is_acronym_like(name) and len(form) > len(name):
        return f"{form} ({name})"
    # Any other alias: the index's name leads, the posting's word follows. The
    # bare JD form was measured first and rejected — rendering CPF's
    # "Containerization" in place of "Docker" satisfies the posting at the cost
    # of the better-known keyword, and an ATS that matches one would have
    # matched both.
    return f"{name} ({form})"


def align_labels(
    index: Any, forms: Mapping[str, str] | None = None
) -> dict[str, str]:
    """``skill id → label`` for every skill, aligned where the posting differs."""
    forms = dict(forms or {})
    return {skill.id: skill_label(skill, forms.get(skill.id)) for skill in _skills(index)}


def label_parts(label: str) -> list[str]:
    """The distinct words a label puts on the page.

    ``PostgreSQL (SQL)`` prints two terms, not one, and both have to count when
    deciding whether the line already carries something — otherwise aligning a
    ``SQL`` skill onto the posting's "PostgreSQL" leaves the page saying
    PostgreSQL twice, once per taxonomy entry.
    """
    text = str(label or "").strip()
    if not text:
        return []
    outside = re.sub(r"\([^)]*\)", " ", text)
    inside = re.findall(r"\(([^)]*)\)", text)
    return [part.strip() for part in [outside, *inside] if part.strip()]


def dedupe_labels(labels: Iterable[str], seen: set[str] | None = None) -> list[str]:
    """Drop a label naming something already printed, case-insensitively.

    Two taxonomy entries can align onto the same word (a ``SQL`` skill and a
    ``postgresql`` skill when the posting writes "PostgreSQL"), and printing it
    twice is exactly the stuffing ``ats-tailor check`` warns about. ``seen`` is
    carried across lines by the caller: the constraint is the section's, not the
    line's.
    """
    seen = set() if seen is None else seen
    out: list[str] = []
    for label in labels:
        parts = label_parts(label)
        keys = {part.lower() for part in parts}
        if not keys or keys & seen:
            continue
        seen |= keys
        out.append(str(label).strip())
    return out


def _skills(index: Any) -> Sequence[Any]:
    if hasattr(index, "all_skills"):
        return index.all_skills()
    return []


def _forms(skill: Any) -> list[str]:
    if hasattr(skill, "surface_forms"):
        return list(skill.surface_forms())
    name = str(getattr(skill, "name", "") or "")
    return [name] if name else []
