"""Reading a posting: the model in front, the parser behind, the store between.

Three things are true at once here, and the shape of this module is what makes
them all hold:

1. **A model is optional.** ``NoneProvider`` returns nothing and the run is
   exactly the deterministic one Phase 3a shipped — same requirements, same
   ids, same weights. There is no "degraded" mode; there is one mode with an
   optional first opinion.
2. **The parser always runs anyway.** Even when a model answers, the
   deterministic requirements are computed and the ones the model's terms do
   not cover are appended at ``ATS_FALLBACK_REQ_WEIGHT``. A model that
   summarises a posting into eight tidy lines has thrown away the ninth thing
   it asked for, and the safety net is what stops that being invisible.
3. **A human edit outranks both.** ``edited_json`` in the store wins over any
   extraction, including a fresh one — the user who corrected a profile is not
   asking to be corrected back.

The cache key is ``(jd_hash, provider, prompt_version)``. Model extractions are
cached under the provider that produced them; the deterministic profile is
stored under ``fallback`` and *never* read back as a cache — it costs
milliseconds to recompute, and a cached copy would silently outlive the parser
that made it, which is the one way an evaluation can lie about a code change.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader

from .. import config
from ..ats.check import count_in_text, gap_terms, index_term_forms
from .gates import cross_check, detect_gates
from .hashing import jd_hash
from .models import JDProfile, RequirementSpec, WEIGHT_RANGE, squash
from .requirements import requirements_from_jd

PROMPT_TEMPLATE = "prompt_extract.md.j2"
PKG_DIR = Path(__file__).resolve().parent

#: The provider name a deterministically parsed profile is stored under.
FALLBACK_PROVIDER = "fallback"
#: Providers whose stored ``extracted_json`` is a cache worth reading back.
#: ``none`` and ``fallback`` are not on it on purpose — see the module docstring.
CACHEABLE = ("ollama", "lmstudio", "mlx", "external")

#: Words that make a line a job title. Used only by the deterministic title
#: guess, which is why the list is nouns and not a taxonomy.
TITLE_NOUNS = (
    "engineer", "developer", "scientist", "analyst", "architect", "consultant",
    "specialist", "researcher", "administrator", "programmer", "designer",
    "intern", "internship", "manager", "lead",
)
#: A heading is a title when it *ends* in a role noun ("Site Reliability
#: Engineer Intern"), not merely when it contains one — "Internship Learning
#: Outcomes" contains "Internship" and is a section heading.
_TITLE_TAIL_RE = re.compile(
    r"\b(?:" + "|".join(TITLE_NOUNS) + r")s?\b[\s)\]]*$", re.IGNORECASE)
#: "Job Title: Site Reliability Engineer", "Position: Backend Developer".
_LABELLED_TITLE_RE = re.compile(
    r"^\W*(?:job\s+)?(?:title|position|role)\s*[:\-]\s*(?P<title>.+)$",
    re.IGNORECASE | re.MULTILINE,
)
#: "...seeking a talented Robotics Software Engineer Intern to join us"
_INLINE_TITLE_RE = re.compile(
    r"(?i:\b(?:seeking|hiring|looking\s+for|recruiting|as\s+an?|for\s+an?|"
    r"join\s+us\s+as\s+an?)\s+(?:a\s+|an\s+|our\s+|talented\s+|experienced\s+)*)"
    r"(?P<title>(?:[A-Z][\w/&+.-]*\s+){0,4}"
    r"(?:Engineer|Developer|Scientist|Analyst|Architect|Consultant|Specialist|"
    r"Researcher|Intern|Programmer|Designer)"
    r"(?:\s+Intern(?:ship)?)?)"
)
#: A bracketed prefix a posting puts before its own title:
#: "[2026 H2 Term-time Internship] Site Reliability Engineer Intern".
_BRACKET_PREFIX_RE = re.compile(r"^\s*[\[(][^\]\)]{0,60}[\])]\s*")
#: How far into the posting a heading-shaped title is still a title.
TITLE_SCAN_LINES = 40
MAX_TITLE_WORDS = 12


# ── The prompt ───────────────────────────────────────────────────────────────
def render_extract_prompt(
    jd_text: str,
    categories: Sequence[str] = (),
    languages: Sequence[str] = (),
    today: date | None = None,
) -> str:
    """The extraction prompt, posting included, trust boundary and all."""
    env = Environment(loader=FileSystemLoader(str(PKG_DIR)), autoescape=False)
    template = env.get_template(PROMPT_TEMPLATE)
    return template.render(
        jd_text=str(jd_text or "").strip(),
        categories=list(categories),
        languages=list(languages),
        max_requirements=config.MAX_REQUIREMENTS,
        today=(today or date.today()).isoformat(),
    )


# ── Validating what came back ────────────────────────────────────────────────
def term_alias_map(index: Any) -> dict[str, str]:
    """Lowercased surface form → the canonical index term it stands for.

    Not :func:`ats_tailor.engine.adapt.alias_map`, which answers with a *skill id*.
    Requirement terms are matched against claim terms, and those are surface
    forms, so the canonical *name* is what a posting's "K8s" has to become.
    """
    mapping: dict[str, str] = {}
    for term, forms in index_term_forms(index).items():
        canonical = term.strip()
        for form in [term, *forms]:
            key = str(form).strip().lower()
            if key and (key not in mapping or key == canonical.lower()):
                mapping[key] = canonical
    return mapping


def parse_profile(
    raw_json: Any,
    alias_map: Mapping[str, str] | None = None,
    provider: str = "external",
    prompt_version: str = "",
) -> JDProfile | None:
    """Validate and normalise one extraction. ``None`` when it is unusable.

    Normalisation, in order: schema, then terms through the alias map (unknown
    terms stay as the posting wrote them — a requirement the index cannot cover
    is still a requirement), then dedupe by normalised text, then the weight
    band, then a cap at ``ATS_MAX_REQUIREMENTS`` by weight. Requirements with no
    letters in them are dropped: an extractor that emits "3+" has emitted noise
    with an id.
    """
    if raw_json is None:
        return None
    payload = dict(raw_json) if isinstance(raw_json, Mapping) else None
    if payload is None:
        return None
    payload.pop("provider", None)
    payload.pop("prompt_version", None)
    payload.pop("cached", None)
    try:
        profile = JDProfile.model_validate(payload)
    except Exception as exc:
        print(f"  Warning: extraction failed validation ({exc})", file=sys.stderr)
        return None

    aliases = {str(k).strip().lower(): str(v) for k, v in (alias_map or {}).items()}
    low, high = WEIGHT_RANGE
    kept: dict[str, RequirementSpec] = {}
    for spec in profile.requirements:
        text = squash(spec.text)
        if not any(ch.isalpha() for ch in text):
            continue
        terms = _normalise_terms(spec.terms, aliases)
        key = text.lower()
        candidate = spec.model_copy(update={
            "text": text,
            "terms": terms,
            "weight": min(max(float(spec.weight), low), high),
        })
        existing = kept.get(key)
        if existing is None:
            kept[key] = candidate
        elif candidate.weight > existing.weight:
            kept[key] = candidate.model_copy(update={
                "terms": _dedupe([*existing.terms, *candidate.terms])
            })
        else:
            kept[key] = existing.model_copy(update={
                "terms": _dedupe([*existing.terms, *candidate.terms])
            })

    ordered = list(kept.values())
    if len(ordered) > config.MAX_REQUIREMENTS:
        # Stable by weight: ties keep the posting's own order, so a cap never
        # silently reshuffles what survived it.
        keep = sorted(
            range(len(ordered)), key=lambda i: (-ordered[i].weight, i)
        )[: config.MAX_REQUIREMENTS]
        ordered = [ordered[i] for i in sorted(keep)]

    return profile.model_copy(update={
        "requirements": _renumber(ordered),
        "provider": provider,
        "prompt_version": prompt_version,
    })


def _normalise_terms(terms: Iterable[str], aliases: Mapping[str, str]) -> list[str]:
    return _dedupe(aliases.get(squash(term).lower(), squash(term)) for term in terms)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = squash(value)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _renumber(specs: Sequence[RequirementSpec], start: int = 1) -> list[RequirementSpec]:
    return [
        spec.model_copy(update={"id": spec.id or f"req-{i}"})
        for i, spec in enumerate(specs, start=start)
    ]


# ── The deterministic reading ────────────────────────────────────────────────
def detect_title(jd_text: str) -> str | None:
    """The job title the posting advertises, guessed conservatively.

    Three passes, cheapest first: an explicit ``Job Title:`` label, a
    heading-shaped line near the top that names a role noun, and finally a
    sentence that introduces the role ("seeking a Robotics Software Engineer
    Intern"). Guessing wrong puts a wrong target statement on a resume, so each
    pass is narrow and the answer is allowed to be ``None``.
    """
    text = str(jd_text or "")
    labelled = _LABELLED_TITLE_RE.search(text)
    if labelled:
        title = _clean_title(labelled.group("title"))
        if title:
            return title

    for line in [l for l in text.splitlines() if l.strip()][:TITLE_SCAN_LINES]:
        stripped = line.strip()
        if stripped.endswith((".", "!", "?", ",", ";", ":")):
            continue
        candidate = _clean_title(stripped)
        if not candidate or len(candidate.split()) > MAX_TITLE_WORDS:
            continue
        if _TITLE_TAIL_RE.search(candidate) and _looks_like_a_title(candidate):
            return candidate

    inline = _INLINE_TITLE_RE.search(text)
    if inline:
        return _clean_title(inline.group("title"))
    return None


def _clean_title(text: str) -> str | None:
    title = squash(_BRACKET_PREFIX_RE.sub("", str(text or "")))
    title = re.sub(r"\s*[|/–—-]\s*(?:full|part)[- ]time.*$", "", title, flags=re.I)
    title = title.strip(" -–—|,:;")
    return title or None


def _looks_like_a_title(text: str) -> bool:
    """Mostly-capitalised words and no sentence punctuation inside."""
    words = text.split()
    if not words or any(ch in text for ch in ".!?"):
        return False
    capitalised = sum(1 for word in words if word[:1].isupper() or word[:1].isdigit())
    return capitalised / len(words) > 0.6


def detect_archetype(jd_text: str, index: Any) -> str | None:
    """The candidate skill category the posting leans on hardest.

    Counted, not judged: whichever category's surface forms the posting spells
    out most often. It picks the ``summaries.variants`` key, so being wrong
    costs one sentence of the summary and nothing else.
    """
    if not hasattr(index, "skills"):
        return None
    lowered = str(jd_text or "").lower()
    best, best_score = None, 0
    for category in index.skills:
        score = 0
        for skill in category.skills:
            for form in skill.surface_forms():
                score += count_in_text(form, lowered, jd_text)
        if score > best_score:
            best, best_score = category.category, score
    return best


def deterministic_profile(
    jd_text: str,
    index: Any,
    term_forms: Mapping[str, Sequence[str]] | None = None,
    gaps: Sequence[str] | None = (),
    languages: Sequence[str] = (),
    today: date | None = None,
) -> JDProfile:
    """What the posting says, with no model involved.

    The requirements are exactly :func:`requirements_from_jd`'s — same ids, same
    tiers, same weights — so a ``none`` run is bit-for-bit the Phase 3b run.
    """
    requirements = requirements_from_jd(jd_text, term_forms, gaps or ())
    return JDProfile(
        title=detect_title(jd_text),
        archetype=detect_archetype(jd_text, index),
        gates=detect_gates(jd_text, languages, today=today),
        requirements=[
            RequirementSpec(
                id=requirement.id, text=requirement.text, kind=requirement.kind,
                weight=requirement.weight, terms=list(requirement.terms),
                category=requirement.category,
            )
            for requirement in requirements
        ],
        provider=FALLBACK_PROVIDER,
        prompt_version="",
    )


# ── The safety net ───────────────────────────────────────────────────────────
def merge_fallback(
    extracted: JDProfile,
    fallback: JDProfile,
    weight: float | None = None,
    soft_weight: float | None = None,
) -> JDProfile:
    """Append the deterministic requirements the extraction does not cover.

    "Covered" means every term the parsed requirement names is already named by
    some extracted requirement. A parsed requirement that names *nothing* (kept
    only because it contains a requirement verb) is never covered by that test,
    so it comes in at ``ATS_SOFT_REQ_WEIGHT`` — where it belongs: in the report,
    and out of the competition for page space.
    """
    weight = config.FALLBACK_REQ_WEIGHT if weight is None else float(weight)
    soft_weight = config.SOFT_REQ_WEIGHT if soft_weight is None else float(soft_weight)

    covered = {
        term.strip().lower()
        for spec in extracted.requirements for term in spec.terms if term.strip()
    }
    seen_text = {spec.text.strip().lower() for spec in extracted.requirements}

    additions: list[RequirementSpec] = []
    for spec in fallback.requirements:
        if spec.text.strip().lower() in seen_text:
            continue
        terms = [t for t in spec.terms if t.strip()]
        if terms and all(t.strip().lower() in covered for t in terms):
            continue
        additions.append(spec.model_copy(update={
            "id": "",
            "weight": weight if terms else soft_weight,
        }))

    if not additions:
        return extracted
    # The extraction keeps its own ids — a `reweight` in an overrides.yaml
    # names them — and the appended ones say in the id where they came from.
    appended = [
        spec.model_copy(update={"id": f"req-net-{i}"})
        for i, spec in enumerate(additions, start=1)
    ]
    return extracted.model_copy(update={
        "requirements": [*extracted.requirements, *appended]})


# ── The whole thing ──────────────────────────────────────────────────────────
def extract_requirements(
    jd_text: str,
    index: Any,
    provider,
    store=None,
    *,
    jd_path: str | Path | None = None,
    prompt_version: str | None = None,
    reextract: bool = False,
    extra_known: Sequence[str] = (),
    languages: Sequence[str] = (),
    term_forms: Mapping[str, Sequence[str]] | None = None,
    gaps: Sequence[str] | None = None,
    today: date | None = None,
    say=print,
) -> JDProfile:
    """The posting as a profile: cache, then model, then parser, then net.

    ``term_forms`` and ``gaps`` are the index vocabulary and the posting's
    unknown technology words; the pipeline already computes both for the report
    and passes them in rather than paying for them twice.
    """
    prompt_version = config.PROMPT_VERSION if prompt_version is None else prompt_version
    digest = jd_hash(jd_text)
    name = getattr(provider, "name", "none")
    row_provider = FALLBACK_PROVIDER if name in {"none", FALLBACK_PROVIDER} else name
    # The deterministic reading does not vary with a prompt, so its row is not
    # keyed on one: a prompt edit must not orphan the edit a user made to it.
    row_version = "" if row_provider == FALLBACK_PROVIDER else prompt_version

    term_forms = index_term_forms(index) if term_forms is None else term_forms
    gaps = gap_terms(jd_text, index, extra_known=list(extra_known)) if gaps is None else gaps
    fallback = deterministic_profile(
        jd_text, index, term_forms, gaps, languages, today=today)

    stored = _stored(store, digest, row_provider, row_version)
    edited = JDProfile.from_json(stored.edited) if stored is not None else None
    if edited is not None:
        edited = edited.model_copy(update={
            "provider": stored.provider, "prompt_version": prompt_version,
            "cached": True,
        })

    cached = None
    if stored is not None and not reextract and row_provider in CACHEABLE:
        cached = JDProfile.from_json(stored.extracted)
        if cached is not None:
            cached = cached.model_copy(update={
                "provider": stored.provider, "prompt_version": prompt_version,
                "cached": True,
            })

    profile = cached
    if profile is None:
        answer = None
        if name not in {"none", FALLBACK_PROVIDER}:
            answer = provider.extract(
                jd_text, [category.category for category in getattr(index, "skills", [])],
                prompt_version,
            )
        # The safety net, the gate cross-check and the title fallback apply to a
        # *fresh* extraction only: a cached profile was merged when it was made,
        # and doing it twice would append the same requirements to itself.
        profile = None if answer is None else _reconcile(answer, fallback)

    if profile is None:
        profile = fallback

    _persist(store, digest, jd_text, profile, jd_path,
             "" if profile.provider == FALLBACK_PROVIDER else prompt_version)

    if edited is not None:
        if reextract:
            say("  Note: a stored edit outranks this re-extraction; "
                "the fresh reading was saved as `extracted`")
        # The edit wins on requirements, but gates are a safety mechanism and
        # get the same treatment as a model's: a `pass` the posting contradicts
        # is downgraded to `flag` with the quote, whoever typed it.
        return edited.model_copy(update={
            "gates": cross_check(edited.gates, fallback.gates),
            "title": edited.title or profile.title,
            "archetype": edited.archetype or profile.archetype,
            "requirements": _renumber(edited.requirements),
        })
    return profile


def _reconcile(extracted: JDProfile, fallback: JDProfile) -> JDProfile:
    """A model's reading, checked against the parser's: net, gates, title."""
    profile = merge_fallback(extracted, fallback)
    return profile.model_copy(update={
        "title": profile.title or fallback.title,
        "archetype": profile.archetype or fallback.archetype,
        "gates": cross_check(profile.gates, fallback.gates),
    })


def _stored(store: Any, digest: str, provider: str, prompt_version: str):
    if store is None:
        return None
    try:
        return store.get_jd_profile(digest, provider=provider,
                                    prompt_version=prompt_version)
    except Exception:
        return None


def _persist(store, digest, jd_text, profile: JDProfile, jd_path, prompt_version) -> None:
    if store is None or profile.cached:
        return
    try:
        store.put_jd_profile(
            digest, jd_text,
            provider=profile.provider or FALLBACK_PROVIDER,
            source_path=str(jd_path) if jd_path else None,
            prompt_version=prompt_version,
            extracted=profile.to_json(),
        )
    except Exception as exc:  # a store that will not write must not stop a run
        print(f"  Warning: could not record the JD profile ({exc})", file=sys.stderr)


def save_edited(store, jd_text: str, profile: JDProfile, prompt_version: str | None = None,
                jd_path: str | Path | None = None) -> None:
    """Write a hand-edited profile back, where it outranks any extraction."""
    provider = profile.provider or FALLBACK_PROVIDER
    if prompt_version is None:
        prompt_version = "" if provider == FALLBACK_PROVIDER else config.PROMPT_VERSION
    store.put_jd_profile(
        jd_hash(jd_text), jd_text,
        provider=provider,
        source_path=str(jd_path) if jd_path else None,
        prompt_version=prompt_version,
        edited=profile.to_json(),
    )
