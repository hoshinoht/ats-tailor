"""Turn a posting into weighted requirements, deterministically.

This is the fallback extractor: no model, no network, same answer every time.
Phase 3c puts an LLM in front of it; this stays as what runs when the model is
absent, refuses, or returns something that fails validation.

The tiering is the one ``atscheck.classify_jd_sections`` already computes for
the keyword check, so a term counted as must-have by the report and a
requirement weighted as must-have by the selection agree by construction.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from ats_match import Requirement

from .. import config
from ..ats.check import classify_jd_sections, count_in_text
from ..engine.legacy.scoring import chunk_text

#: A line with no known vocabulary in it is still a requirement if it *asks*
#: for something. Without this, "Able to work independently in a small team"
#: would be dropped and its responsibility weight lost.
REQUIREMENT_VERB_RE = re.compile(
    r"experience|knowledge|familiar|proficien|understand|ability|able to|"
    r"skills?|degree|build|design|develop|deploy|maintain|support|work",
    re.IGNORECASE,
)

#: Section tier → (requirement kind, weight).
TIER_KINDS: dict[str, tuple[str, float]] = {
    "must": ("must", 1.0),
    "nice": ("nice", 1.0),
    "body": ("responsibility", 0.5),
}
_KIND_RANK = {"responsibility": 0, "nice": 1, "must": 2}
MIN_WORDS = 4


def requirements_from_jd(
    jd_text: str,
    term_forms: Mapping[str, Sequence[str]] | None = None,
    gaps: Sequence[str] = (),
    soft_weight: float | None = None,
) -> list[Requirement]:
    """Requirements for ``jd_text``, tiered by the section each line sits under.

    ``term_forms`` is the index vocabulary (canonical term → surface forms);
    ``gaps`` are the posting's technology-looking words the index has no claim
    for, which are kept as requirement terms precisely so a requirement the
    index cannot cover still shows up as one.

    A chunk under an ``ignore`` heading (culture, benefits, how to apply) is
    dropped outright rather than tiered down — see
    :func:`~ats_tailor.ats.check.classify_jd_sections`.

    A line that names *nothing* — no index vocabulary, no gap term, kept only
    because it contains a requirement verb — is a soft requirement and takes
    ``soft_weight`` whatever heading it sits under (``ATS_SOFT_REQ_WEIGHT``,
    default 0.25). "Works well independently and with others" is a real thing
    the posting asks for and belongs in the report, but it should not compete
    for page space with "a solid understanding of Kubernetes": every claim
    answers it about equally well, so the weight it carries is spent on noise.
    """
    term_forms = dict(term_forms or {})
    soft_weight = config.SOFT_REQ_WEIGHT if soft_weight is None else float(soft_weight)
    gap_list = [str(term) for term in gaps if str(term).strip()]

    chunks = chunk_text(jd_text)
    tiers = classify_jd_sections(chunks)

    merged: dict[str, Requirement] = {}
    for chunk, tier in zip(chunks, tiers):
        text = chunk.strip()
        # What a posting says about itself is not a weaker ask, it is not an
        # ask: "Mission-Driven Culture: we are Digital to the Core" would
        # otherwise arrive as a must-have and compete for page space with
        # "a solid understanding of Kubernetes".
        if tier == "ignore":
            continue
        if len(text.split()) < MIN_WORDS or not any(ch.isalpha() for ch in text):
            continue
        terms = _terms_in(text, term_forms, gap_list)
        if not terms and not REQUIREMENT_VERB_RE.search(text):
            continue
        kind, weight = TIER_KINDS.get(tier, TIER_KINDS["body"])
        if not terms:
            weight = soft_weight
        key = text.lower()
        existing = merged.get(key)
        if existing is None:
            merged[key] = Requirement(
                id="", text=text, kind=kind, weight=weight, terms=terms
            )
        elif _KIND_RANK[kind] > _KIND_RANK[existing.kind]:
            # The same sentence under two headings takes the stronger tier.
            combined = tuple(dict.fromkeys(existing.terms + terms))
            merged[key] = Requirement(
                id="", text=existing.text, kind=kind,
                weight=weight if combined else soft_weight, terms=combined,
            )

    return [
        Requirement(
            id=f"req-{i}", text=r.text, kind=r.kind, weight=r.weight, terms=r.terms
        )
        for i, r in enumerate(merged.values(), start=1)
    ]


def _terms_in(
    text: str, term_forms: Mapping[str, Sequence[str]], gaps: Sequence[str]
) -> tuple[str, ...]:
    """Canonical index terms named in ``text``, plus any gap words it uses."""
    lowered = text.lower()
    found: list[str] = []
    for term, forms in term_forms.items():
        surface = list(forms) or [term]
        if any(count_in_text(form, lowered, text) for form in surface):
            found.append(term)
    found.extend(gap for gap in gaps if count_in_text(gap, lowered, text))
    return tuple(dict.fromkeys(found))
