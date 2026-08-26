"""The metric guardrail: what a proposed rephrase is not allowed to change.

Phase 5 lets a model or an agent *propose* wording. This module is the reason
that is safe. Every proposal — from a local model, from the driving agent
through the MCP, or typed by the user in the TUI — passes through
:func:`check_variant` before it can be stored, and nothing reaches a PDF that
has not been accepted by a human *after* passing it.

The rules are the honesty rules of ``docs/v2-design.md`` §2 made executable:

``metrics``
    every number the source bullet recorded still appears in the proposal.
``technology``
    every technology the source bullet *said* is still said.
``new_technology``
    the proposal names no technology that is neither in the source bullet nor
    in the item's own ``tech_stack`` / ``skills`` / ``keywords``.
``numbers``
    the proposal invents no number the source did not carry.
``length`` / ``style`` / ``person``
    house style: at most :data:`ats_tailor.index.models.MAX_BULLET_CHARS`
    characters, no trailing full stop, no first person.

Everything here is a pure function of the source bullet, its item and the
proposed string, so a rule can be tested in both directions without a store, a
model or an index on disk. ``index`` is optional and only widens what counts as
a *known* technology: without it, a term is judged against the item alone.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .ats.check import count_in_text, gap_terms, index_term_forms
from .index.migrate import find_metrics
from .index.models import MAX_BULLET_CHARS, Bullet

#: Rule names, in the order :meth:`GuardrailReport.summary` prints them.
RULES: tuple[str, ...] = (
    "metrics", "technology", "new_technology", "numbers",
    "length", "style", "person",
)

#: Words that make a sentence first person. A resume bullet has no subject —
#: "Architected …", not "I architected …" — and a model asked to make a line
#: warmer will reach for one.
FIRST_PERSON: tuple[str, ...] = (
    "i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine", "myself",
    "we", "we're", "we've", "us", "our", "ours", "ourselves",
)
_FIRST_PERSON_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in FIRST_PERSON) + r")\b",
    re.IGNORECASE,
)

#: Any number, with thousands separators and decimals. Deliberately simpler
#: than :data:`ats_tailor.index.migrate.METRIC_RE`: this one asks "is there a
#: quantity here at all", and a unit it failed to recognise would let an
#: invented figure through.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WHITESPACE_RE = re.compile(r"\s+")

#: ``tech_stack`` keys a project may carry. Everything under them is a
#: technology the item is entitled to name.
TECH_STACK_KEYS: tuple[str, ...] = (
    "languages", "frameworks", "infrastructure", "patterns", "tools",
)


class Violation(BaseModel):
    """One broken rule, in the words the person deciding needs."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    detail: str


class GuardrailReport(BaseModel):
    """The verdict on one proposal, with the diff that produced it."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    violations: list[Violation] = Field(default_factory=list)
    #: Unified diff between the index's wording and the proposal.
    diff: str = ""
    #: What was compared, so a stored verdict is readable years later.
    source: str = ""
    proposed: str = ""

    def summary(self) -> str:
        """One line: ``ok`` or the rules that failed, in rule order."""
        if self.ok:
            return "ok"
        broken = [rule for rule in RULES
                  if any(v.rule == rule for v in self.violations)]
        extra = [v.rule for v in self.violations if v.rule not in RULES]
        return "failed: " + ", ".join([*broken, *dict.fromkeys(extra)])

    def rules_broken(self) -> list[str]:
        return list(dict.fromkeys(v.rule for v in self.violations))


# ── Text helpers ─────────────────────────────────────────────────────────────
def squash(text: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _numbers(text: str) -> list[str]:
    """Every number in ``text``, normalised so ``5,100`` and ``5100`` match."""
    return [match.group(0).replace(",", "") for match in _NUMBER_RE.finditer(text)]


def mentions_metric(text: str, metric: str) -> bool:
    """Whether ``text`` still carries ``metric``.

    Whitespace-insensitive on the literal ("15 %" satisfies "15%"), and when
    the literal is gone the metric's *numbers* must all still be there — so a
    rephrase may move a unit around but may not turn 15% into 20%.
    """
    haystack = squash(text).lower()
    needle = squash(metric).lower()
    if needle and _strip_spaces(needle) in _strip_spaces(haystack):
        return True
    numbers = _numbers(metric)
    if not numbers:
        return False
    present = _numbers(haystack)
    return all(number in present for number in numbers)


def _strip_spaces(text: str) -> str:
    return text.replace(" ", "")


#: A word, keeping the punctuation technology names are spelled with (``C++``,
#: ``.NET``, ``CI/CD``) and dropping the punctuation sentences are.
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#./-]*")


def words(text: str) -> list[str]:
    """``text`` as bare words, so a known-vocabulary check is not defeated by
    the comma after "TLS"."""
    return [word.strip(".-/") or word for word in _WORD_RE.findall(str(text or ""))]


# ── Vocabulary ───────────────────────────────────────────────────────────────
def item_terms(item: Any, index: Any = None) -> dict[str, list[str]]:
    """Canonical term → surface forms the *item* is entitled to name.

    Its ``keywords``, every surface form of every skill it or its bullets cite,
    and — for a project — everything in its ``tech_stack``. This is the set the
    "no new technology" rule measures a proposal against: a bullet may reach
    for another of its own project's technologies, and may not reach outside
    it.
    """
    forms: dict[str, list[str]] = {}

    def add(term: str, surface: Sequence[str] = ()) -> None:
        term = str(term).strip()
        if not term:
            return
        merged = forms.setdefault(term, [])
        for value in [term, *surface]:
            value = str(value).strip()
            if value and value.lower() not in {m.lower() for m in merged}:
                merged.append(value)

    if item is None:
        return forms

    skills = index.skill_map() if hasattr(index, "skill_map") else {}
    skill_ids = list(getattr(item, "skills", []) or [])
    for bullet in getattr(item, "bullets", []) or []:
        skill_ids.extend(getattr(bullet, "skills", []) or [])
    for skill_id in dict.fromkeys(skill_ids):
        skill = skills.get(skill_id)
        if skill is not None:
            add(skill.name, skill.surface_forms())
        else:
            add(str(skill_id))
    for keyword in getattr(item, "keywords", []) or []:
        add(keyword)
    tech_stack = getattr(item, "tech_stack", {}) or {}
    if isinstance(tech_stack, Mapping):
        for key in [*TECH_STACK_KEYS, *tech_stack]:
            for value in tech_stack.get(key, []) or []:
                add(value)
    return forms


def terms_in(text: str, forms: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Canonical term → the surface form ``text`` spells out, for every hit."""
    lowered = str(text or "").lower()
    found: dict[str, str] = {}
    for term, surfaces in forms.items():
        for surface in [term, *surfaces]:
            if surface and count_in_text(str(surface), lowered, str(text or "")):
                found[term] = str(surface)
                break
    return found


def _merged_forms(*maps: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for mapping in maps:
        for term, surfaces in mapping.items():
            row = merged.setdefault(term, [])
            for surface in [term, *surfaces]:
                if surface and str(surface).lower() not in {r.lower() for r in row}:
                    row.append(str(surface))
    return merged


def unknown_technologies(
    proposed: str, source_text: str, item: Any, index: Any = None
) -> list[str]:
    """Capitalised, technology-shaped words in the proposal that nothing knows.

    :func:`ats_tailor.ats.check.gap_terms` is the same detector the coverage
    report uses to say "the posting asks for this and the index has never
    heard of it"; pointed at a proposal instead of a posting, it answers "the
    model made this up". Everything the source bullet already said, everything
    the item may say, and the proposal's *first* word are known — a sentence
    starts with a capital because it is a sentence, and a technology in that
    position is caught by the alias-map rule when the index knows it at all.
    """
    if index is None:
        return []
    known: list[str] = words(source_text)
    for term, surfaces in item_terms(item, index).items():
        known.extend([term, *surfaces])
    known.extend(words(proposed)[:1])
    for attribute in ("name", "company", "title", "id"):
        value = getattr(item, attribute, None)
        if value:
            known.append(str(value))
    try:
        return gap_terms(proposed, index, extra_known=known)
    except Exception:  # a detector that fails must not block a proposal
        return []


# ── The rules ────────────────────────────────────────────────────────────────
def source_metrics(source: Bullet) -> list[str]:
    """What the bullet claims: its recorded ``metrics`` plus what its text says.

    ``metrics`` is filled in by the migration and by ``index_put``, but a
    hand-edited bullet may carry a figure nobody recorded.
    :func:`ats_tailor.index.migrate.find_metrics` is the same reader the
    migration used, so a number is guarded whether or not it made it into the
    field.
    """
    recorded = [str(m) for m in (getattr(source, "metrics", []) or []) if str(m).strip()]
    seen = {_strip_spaces(m.lower()) for m in recorded}
    for found in find_metrics(str(source.text or "")):
        key = _strip_spaces(found.lower())
        if key not in seen:
            seen.add(key)
            recorded.append(found)
    return recorded


def check_metrics(source: Bullet, proposed: str) -> list[Violation]:
    """Every metric the source claims survives the rephrase."""
    return [
        Violation(rule="metrics",
                  detail=f"the source bullet claims {metric!r} and the proposal "
                         f"does not")
        for metric in source_metrics(source)
        if not mentions_metric(proposed, metric)
    ]


def check_numbers(source: Bullet, proposed: str) -> list[Violation]:
    """No number appears that the source did not carry."""
    allowed = set(_numbers(source.text))
    for metric in getattr(source, "metrics", []) or []:
        allowed.update(_numbers(metric))
    invented = [n for n in dict.fromkeys(_numbers(proposed)) if n not in allowed]
    return [
        Violation(rule="numbers",
                  detail=f"the proposal introduces the number {number!r}, which is "
                         f"in neither the source bullet nor its metrics")
        for number in invented
    ]


def check_technology(
    source: Bullet, item: Any, proposed: str, index: Any = None
) -> list[Violation]:
    """Every technology the source *says* is still said."""
    forms = _known_forms(item, index)
    said = terms_in(source.text, forms)
    violations: list[Violation] = []
    for term, surface in said.items():
        if not terms_in(proposed, {term: forms.get(term, [term])}):
            violations.append(Violation(
                rule="technology",
                detail=f"the source bullet names {surface!r} and the proposal "
                       f"names it by no form"))
    return violations


def check_new_technology(
    source: Bullet, item: Any, proposed: str, index: Any = None
) -> list[Violation]:
    """No technology the source bullet and its item do not have."""
    forms = _known_forms(item, index)
    allowed = set(item_terms(item, index))
    in_source = set(terms_in(source.text, forms))
    violations: list[Violation] = []
    for term in terms_in(proposed, forms):
        if term in in_source or term in allowed:
            continue
        violations.append(Violation(
            rule="new_technology",
            detail=f"the proposal names {term!r}, which is in neither the source "
                   f"bullet nor the item's technologies"))
    for term in unknown_technologies(proposed, source.text, item, index):
        violations.append(Violation(
            rule="new_technology",
            detail=f"the proposal names {term!r}, which nothing in the index "
                   f"knows about"))
    return violations


def source_technologies(text: str, item: Any, index: Any = None) -> list[str]:
    """The surface forms ``text`` spells out, as the ``technology`` rule sees them.

    Exported for the prompt: a model told "keep FastAPI" is told exactly what
    the check will look for, rather than a paraphrase of it.
    """
    return list(terms_in(str(text or ""), _known_forms(item, index)).values())


def check_style(proposed: str) -> list[Violation]:
    """House style: a length, no trailing stop, and no first person."""
    text = str(proposed or "")
    violations: list[Violation] = []
    if not text.strip():
        violations.append(Violation(rule="style", detail="the proposal is empty"))
        return violations
    if len(text.strip()) > MAX_BULLET_CHARS:
        violations.append(Violation(
            rule="length",
            detail=f"the proposal is {len(text.strip())} characters "
                   f"(the index allows {MAX_BULLET_CHARS})"))
    if text.strip().endswith("."):
        violations.append(Violation(
            rule="style",
            detail="index bullets do not end in a full stop"))
    person = _FIRST_PERSON_RE.search(text)
    if person:
        violations.append(Violation(
            rule="person",
            detail=f"the proposal is first person ({person.group(0)!r}); resume "
                   f"bullets have no subject"))
    return violations


def _known_forms(item: Any, index: Any = None) -> dict[str, list[str]]:
    """Every canonical term either the index or the item can name."""
    maps: list[Mapping[str, Sequence[str]]] = []
    if index is not None:
        try:
            maps.append(index_term_forms(index))
        except Exception:  # an index that cannot be read is one we do not use
            pass
    maps.append(item_terms(item, index))
    return _merged_forms(*maps)


# ── The check ────────────────────────────────────────────────────────────────
def diff_text(source: str, proposed: str) -> str:
    """A unified diff of the two wordings, for a human reading a verdict."""
    lines = difflib.unified_diff(
        [squash(source)], [squash(proposed)],
        fromfile="index", tofile="proposal", lineterm="", n=0,
    )
    return "\n".join(lines)


def check_variant(
    source: Bullet, item: Any, proposed: str, index: Any = None
) -> GuardrailReport:
    """Whether ``proposed`` may stand in for ``source`` — and why not.

    ``item`` is the role or project the bullet belongs to; ``index`` is the
    whole index when one is available, which is what lets the "no new
    technology" rule tell "a word nobody has ever heard of" from "a technology
    that belongs to a different project".
    """
    text = str(proposed or "")
    violations = [
        *check_metrics(source, text),
        *check_technology(source, item, text, index),
        *check_new_technology(source, item, text, index),
        *check_numbers(source, text),
        *check_style(text),
    ]
    return GuardrailReport(
        ok=not violations,
        violations=violations,
        diff=diff_text(source.text, text),
        source=squash(source.text),
        proposed=squash(text),
    )


# ── Summaries ────────────────────────────────────────────────────────────────
def check_summary(
    proposed: str,
    allowed_text: Iterable[str],
    source: str = "",
    index: Any = None,
) -> GuardrailReport:
    """The same contract for the Summary: no new claims, no new numbers.

    ``allowed_text`` is everything the summary is allowed to be made of — the
    index's own summaries, its highlights, and the claims selected for this
    application. A noun phrase is approximated by its technology terms and its
    numbers, which is what a guardrail can check without a model: the sentence
    may be rearranged, and it may not introduce a technology or a figure that
    is nowhere in the index.
    """
    corpus = "\n".join(str(part) for part in allowed_text if str(part or "").strip())
    text = str(proposed or "")
    violations: list[Violation] = []

    forms: dict[str, list[str]] = {}
    if index is not None:
        try:
            forms = dict(index_term_forms(index))
        except Exception:
            forms = {}
    known = set(terms_in(corpus, forms))
    for term in terms_in(text, forms):
        if term not in known:
            violations.append(Violation(
                rule="new_technology",
                detail=f"the summary names {term!r}, which no index claim carries"))

    allowed_numbers = set(_numbers(corpus))
    for number in dict.fromkeys(_numbers(text)):
        if number not in allowed_numbers:
            violations.append(Violation(
                rule="numbers",
                detail=f"the summary introduces the number {number!r}, which is "
                       f"nowhere in the index"))

    if not text.strip():
        violations.append(Violation(rule="style", detail="the summary is empty"))
    person = _FIRST_PERSON_RE.search(text)
    if person:
        violations.append(Violation(
            rule="person",
            detail=f"the summary is first person ({person.group(0)!r})"))
    for term in unknown_summary_terms(text, corpus, index):
        violations.append(Violation(
            rule="new_technology",
            detail=f"the summary names {term!r}, which nothing in the index "
                   f"knows about"))

    return GuardrailReport(
        ok=not violations, violations=violations,
        diff=diff_text(source, text), source=squash(source), proposed=squash(text),
    )


def unknown_summary_terms(proposed: str, corpus: str, index: Any = None) -> list[str]:
    """:func:`unknown_technologies` for a summary, whose corpus is the index."""
    if index is None:
        return []
    known = [*words(corpus), *words(proposed)[:1]]
    try:
        return gap_terms(proposed, index, extra_known=known)
    except Exception:
        return []
