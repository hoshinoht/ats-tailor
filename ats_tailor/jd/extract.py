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
from ..engine.legacy.scoring import AMBIGUOUS_KEYWORDS
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

#: How much of the prompt the index's vocabulary is allowed to take. ~1 token
#: per 3.5 characters, so 7000 characters is under the 2k-token budget the
#: extraction prompt can spare next to a posting.
VOCABULARY_CHARS = 7000

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
    vocabulary: Sequence[str] = (),
) -> str:
    """The extraction prompt, posting included, trust boundary and all.

    ``vocabulary`` is :func:`index_vocabulary` — the names the index can match.
    Showing it is what turns "Azure Kubernetes Services (AKS)" into terms the
    scoring can use; the posting's own spelling is asked for separately, in
    ``surface``, so nothing is lost by asking for the canonical one.
    """
    env = Environment(loader=FileSystemLoader(str(PKG_DIR)), autoescape=False)
    template = env.get_template(PROMPT_TEMPLATE)
    return template.render(
        jd_text=str(jd_text or "").strip(),
        categories=list(categories),
        languages=list(languages),
        vocabulary=list(vocabulary),
        max_requirements=config.MAX_REQUIREMENTS,
        today=(today or date.today()).isoformat(),
    )


# ── The index's own words ────────────────────────────────────────────────────
def term_alias_map(index: Any) -> dict[str, str]:
    """Lowercased surface form → the canonical index term it stands for.

    Not :func:`ats_tailor.engine.adapt.alias_map`, which answers with a *skill id*.
    Requirement terms are matched against claim terms, and those are surface
    forms, so the canonical *name* is what a posting's "K8s" has to become.
    """
    return alias_map_from_forms(index_term_forms(index))


def alias_map_from_forms(term_forms: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """:func:`term_alias_map` from an already-computed ``index_term_forms``.

    The pipeline computes the vocabulary once per run and hands it around; this
    is the same map without paying for the index walk a second time.
    """
    mapping: dict[str, str] = {}
    for term, forms in term_forms.items():
        canonical = str(term).strip()
        for form in [term, *(forms or ())]:
            key = str(form).strip().lower()
            if key and (key not in mapping or key == canonical.lower()):
                mapping[key] = canonical
    return mapping


def index_vocabulary(
    index: Any, limit: int = VOCABULARY_CHARS, term_forms=None
) -> list[str]:
    """One line per skill — ``Kubernetes (K8s, Container Orchestration)``.

    This is what the extraction prompt shows the model so it answers in words
    the index can actually match. Skills only, not every item keyword: the
    keyword list is four times longer and mostly repeats the skill names, and a
    prompt that spends two thousand tokens on vocabulary has spent them on
    something other than the posting. Truncated at ``limit`` characters, in
    index order, so a much larger index degrades by dropping the tail rather
    than by blowing the context window.
    """
    lines: list[str] = []
    if hasattr(index, "all_skills"):
        for skill in index.all_skills():
            forms = [str(f).strip() for f in skill.surface_forms()]
            name = forms[0] if forms else str(skill.name).strip()
            others = [f for f in forms[1:] if f.lower() != name.lower()]
            lines.append(f"{name} ({', '.join(others)})" if others else name)
    else:  # a v1 index has no canonical/alias distinction; list the terms
        lines = [str(term) for term in (term_forms or index_term_forms(index))]

    kept, spent = [], 0
    for line in lines:
        if spent + len(line) + 1 > limit:
            break
        kept.append(line)
        spent += len(line) + 1
    return kept


# ── Speaking the index's language ────────────────────────────────────────────
def _spans(form: str, text: str, text_lower: str) -> list[tuple[int, int]]:
    """Where ``form`` occurs in ``text``, under :func:`count_in_text` semantics.

    Same three branches, same answers — punctuated names ("C++", "CI/CD") match
    as substrings, ``C``/``Go``/``R`` match case-sensitively against the
    original, everything else on a lowercased word boundary — but returning the
    spans rather than a count, because a longer form has to be able to swallow
    the shorter ones sitting inside it.
    """
    kw = form.lower()
    if not kw.replace(" ", "").replace("-", "").isalnum():
        pattern, target = re.escape(kw), text_lower
    elif kw in AMBIGUOUS_KEYWORDS or (len(kw) <= 2 and kw.isalpha()):
        pattern, target = r"\b" + re.escape(form) + r"\b", text
    else:
        pattern, target = r"\b" + re.escape(kw) + r"\b", text_lower
    return [m.span() for m in re.finditer(pattern, target)]


def normalise_term(phrase: str, aliases: Mapping[str, str]) -> list[str]:
    """The index terms named *inside* ``phrase``; ``[phrase]`` when none is.

    A model told to copy the posting's spelling answers "Azure Kubernetes
    Services (AKS)", "Infrastructure as Code (IaC)", "DevOps CI/CD workflows".
    Resolving those by exact lookup finds nothing, and a requirement whose terms
    resolve to nothing scores through the embedding channel alone — the BM25 and
    term-overlap channels of ``ats_match.build_matrix`` go dark. So the phrase is
    *scanned* instead of looked up: every surface form the index knows that
    occurs in it with word boundaries becomes the canonical term it stands for.

    Longest first, and a form whose every occurrence sits inside one already
    taken is dropped — "Amazon Web Services" answers ``AWS`` once, not ``AWS``
    plus whatever "Services" happens to be an alias of. A phrase no form matches
    is kept exactly as it came: that is a gap term, and a gap the index cannot
    cover still has to be visible as one.
    """
    text = squash(phrase)
    if not text or not aliases:
        return [text] if text else []
    lowered = text.lower()

    found: list[tuple[int, int, str]] = []
    taken: list[tuple[int, int]] = []
    for key in sorted(aliases, key=lambda k: (-len(k), k)):
        canonical = str(aliases[key]).strip()
        if not canonical:
            continue
        # The map is keyed lowercase; C/Go/R match case-sensitively, and the
        # canonical name is the one spelling of the key that kept its case.
        form = canonical if canonical.lower() == key else key
        fresh = [
            span for span in _spans(form, text, lowered)
            if not any(start <= span[0] and span[1] <= end for start, end in taken)
        ]
        if not fresh:
            continue
        taken.extend(fresh)
        found.append((fresh[0][0], -len(key), canonical))

    if not found:
        return [text]
    return _dedupe(canonical for _, _, canonical in sorted(found))


def normalise_profile_terms(
    profile: JDProfile, aliases: Mapping[str, str]
) -> JDProfile:
    """:func:`normalise_term` over every requirement of a whole profile.

    Applied to a profile read back from the cache as well as to a fresh one:
    the rows in the store were written by an earlier prompt that asked for the
    posting's spelling, and normalising at read time is what lets them benefit
    without being re-extracted.
    """
    return profile.model_copy(update={"requirements": [
        spec.model_copy(update={
            "terms": _normalise_terms(spec.terms, aliases),
            "surface": _dedupe(spec.surface or spec.terms),
        })
        for spec in profile.requirements
    ]})


def valid_archetype(value: Any, categories: Sequence[str]) -> str | None:
    """``value`` when it names one of ``categories``, otherwise ``None``.

    An extractor asked for "one of the candidate categories" answers "Intern"
    or "Student", and an invented archetype overrides the deterministic one —
    which is counted from the posting rather than imagined, and picks the
    summary variant. With no categories to check against, nothing is claimed
    and the value stands.
    """
    text = squash(value)
    if not text or not categories:
        return text or None
    for category in categories:
        if squash(category).lower() == text.lower():
            return squash(category)
    return None


def parse_profile(
    raw_json: Any,
    alias_map: Mapping[str, str] | None = None,
    provider: str = "external",
    prompt_version: str = "",
    categories: Sequence[str] = (),
) -> JDProfile | None:
    """Validate and normalise one extraction. ``None`` when it is unusable.

    Normalisation, in order: schema, then the archetype against ``categories``
    (an invented one becomes ``None`` so the deterministic count backfills it),
    then terms through :func:`normalise_term` (a phrase no index form appears in
    stays as the posting wrote it — a requirement the index cannot cover is
    still a requirement), then dedupe by normalised text, then the weight band,
    then a cap at ``ATS_MAX_REQUIREMENTS`` by weight. Requirements with no
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
        candidate = spec.model_copy(update={
            "text": text,
            "terms": _normalise_terms(spec.terms, aliases),
            # What the posting called them, kept for the report. An extractor
            # that answered with none is one whose `terms` *are* the posting's
            # words, so those stand in — before normalisation moves them onto
            # the index's.
            "surface": _dedupe(spec.surface or spec.terms),
            "weight": min(max(float(spec.weight), low), high),
        })
        key = text.lower()
        existing = kept.get(key)
        if existing is None:
            kept[key] = candidate
            continue
        winner = candidate if candidate.weight > existing.weight else existing
        kept[key] = winner.model_copy(update={
            "terms": _dedupe([*existing.terms, *candidate.terms]),
            "surface": _dedupe([*existing.surface, *candidate.surface]),
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
        "archetype": valid_archetype(profile.archetype, categories),
        "requirements": _renumber(ordered),
        "provider": provider,
        "prompt_version": prompt_version,
    })


def _normalise_terms(terms: Iterable[str], aliases: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    for term in terms:
        out.extend(normalise_term(term, aliases))
    return _dedupe(out)


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
    categories = [category.category for category in getattr(index, "skills", [])]

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
            # A stored profile was written by whatever prompt was current when
            # it was made, so it is re-normalised on the way out rather than
            # trusted: the vocabulary alignment is post-processing, and a row
            # that predates it should not be the one run that misses out.
            cached = normalise_profile_terms(
                cached, alias_map_from_forms(term_forms)
            ).model_copy(update={
                "provider": stored.provider, "prompt_version": prompt_version,
                "archetype": valid_archetype(cached.archetype, categories)
                             or fallback.archetype,
                "cached": True,
            })

    profile = cached
    if profile is None:
        answer = None
        if name not in {"none", FALLBACK_PROVIDER}:
            answer = provider.extract(jd_text, categories, prompt_version)
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
