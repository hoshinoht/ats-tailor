"""Hard filters, read out of the posting's own words.

A gate is not a score. Citizenship, a language nobody declared, a relocation,
or a closing date that has passed decide whether the application is possible at
all, and none of them belongs anywhere near a cosine. They are computed here,
printed before the run, and quoted verbatim so the user checks the posting and
not this module's opinion of it.

Everything in this file is deterministic. The same regexes are used twice: to
build the gates when no model ran, and to *cross-check* a model that did — an
LLM that answers ``pass`` where the text plainly says "Singapore Citizens only"
is downgraded to ``flag`` with the quote it missed.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Iterable, Sequence

from .models import GATE_NAMES, Gate, Gates

#: Clauses that decide who may apply at all.
ELIGIBILITY_RE = re.compile(
    r"Singapore(?:an)?\s+(?:citizen|PR)|permanent\s+resident|\bSC/PR\b|"
    r"\bPR\b|citizenship|security\s+clearance|work\s+pass|employment\s+pass|"
    r"only\s+\w+(?:\s+\w+){0,3}\s+citizens|right\s+to\s+work|"
    r"eligible\s+to\s+work|work\s+authoriz|work\s+authoris",
    re.IGNORECASE,
)

#: Natural languages worth gating on. A whitelist rather than the bare
#: ``proficient in ([A-Z]\w+)`` of the plan: that pattern reads "Proficient in
#: Python" as a language requirement, and a gate that cries wolf on every
#: backend posting is a gate the user learns to skip.
LANGUAGES: tuple[str, ...] = (
    "Mandarin", "Chinese", "Cantonese", "Malay", "Tamil", "Bahasa",
    "Indonesian", "Japanese", "Korean", "Thai", "Vietnamese", "Hindi",
    "German", "French", "Spanish", "Portuguese", "Italian", "Dutch",
    "Russian", "Arabic", "Hebrew", "Polish", "Swedish", "Danish",
    "Norwegian", "Finnish", "Turkish", "Greek", "Czech", "Hungarian",
    "English",
)
_LANGUAGE_ALT = "|".join(sorted(LANGUAGES, key=len, reverse=True))
LANGUAGE_RE = re.compile(
    r"(?:fluent|fluency|proficien\w+|native|conversational|command|speak\w*|"
    r"bilingual|read\s+and\s+write)\b[^.\n]{0,40}?\b(?P<lang>" + _LANGUAGE_ALT + r")\b",
    re.IGNORECASE,
)
#: Languages the candidate is assumed to have without saying so. English is on
#: the list because the posting is written in it.
ASSUMED_LANGUAGES = {"english"}

LOCATION_RE = re.compile(
    r"relocat\w*|on-?site\s+in\b|based\s+in\s+[A-Z]|willing\s+to\s+travel|"
    r"must\s+be\s+(?:located|based)\b|\d+\s*days?\s+(?:a\s+week\s+)?in\s+(?:the\s+)?office",
    re.IGNORECASE,
)

DEADLINE_RE = re.compile(
    r"(?:deadline|closing\s+date|close[sd]?\s+on|apply\s+by|applications?\s+close)"
    r"[^.\n]{0,60}",
    re.IGNORECASE,
)
#: Dates a posting writes: 07/05/2026, 2026-05-07, 7 May 2026, May 7, 2026.
_DATE_PATTERNS = (
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "dmy"),
    (re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b"), "dMy"),
    (re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b"), "Mdy"),
)
_MONTHS = {
    m: i + 1 for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}

SENIORITY_RE = re.compile(
    r"\b(\d{1,2})\s*\+?\s*(?:to\s*\d{1,2}\s*)?years?(?:['’]s)?\s+"
    r"(?:of\s+)?(?:relevant\s+|professional\s+|hands-on\s+|working\s+)?experience",
    re.IGNORECASE,
)
SENIOR_TITLE_RE = re.compile(
    r"\b(?:senior|staff|principal|lead|head\s+of|director|manager|"
    r"architect)\b[^.\n]{0,30}\b(?:engineer|developer|scientist|role|position)\b",
    re.IGNORECASE,
)
#: Years of experience a student application can still plausibly answer.
JUNIOR_YEARS = 3


def _quote(match: re.Match | None, text: str, width: int = 160) -> str | None:
    """The sentence a match sits in, trimmed to something readable."""
    if match is None:
        return None
    start = max(text.rfind("\n", 0, match.start()), text.rfind(". ", 0, match.start()))
    start = 0 if start < 0 else start + 1
    end = min(len(text), match.end() + 80)
    for stop in ("\n", ". "):
        found = text.find(stop, match.end())
        if 0 <= found < end:
            end = found
    snippet = " ".join(text[start:end].split())
    return snippet[:width].strip() or None


def _parse_date(text: str) -> date | None:
    for pattern, order in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            if order == "ymd":
                return date(int(match[1]), int(match[2]), int(match[3]))
            if order == "dmy":
                # Day-first: every posting in this corpus is written for a
                # Singapore/EU audience, and a wrong guess here only ever
                # changes a `flag` into a `pass` on ambiguous dates.
                day, month, year = int(match[1]), int(match[2]), int(match[3])
                if month > 12:
                    day, month = month, day
                return date(year, month, day)
            if order == "dMy":
                month = _MONTHS.get(match[2].lower()[:3])
                return date(int(match[3]), month, int(match[1])) if month else None
            month = _MONTHS.get(match[1].lower()[:3])
            return date(int(match[3]), month, int(match[2])) if month else None
        except (ValueError, TypeError):
            continue
    return None


def detect_gates(
    jd_text: str,
    languages: Iterable[str] = (),
    today: date | None = None,
) -> Gates:
    """The deterministic reading of a posting's hard filters.

    ``languages`` is what the candidate declares (``profile.yaml``); anything
    the posting asks for and the profile does not list comes back as a failure
    the user can overrule, because the alternative is a resume sent to a
    posting that wanted Mandarin.
    """
    text = str(jd_text or "")
    today = today or date.today()
    known = {str(name).strip().lower() for name in languages if str(name).strip()}
    known |= ASSUMED_LANGUAGES

    return Gates(
        eligibility=_eligibility(text),
        language=_language(text, known),
        location=_location(text),
        deadline=_deadline(text, today),
        seniority=_seniority(text),
    )


def _eligibility(text: str) -> Gate:
    match = ELIGIBILITY_RE.search(text)
    if match is None:
        return Gate(verdict="pass", note="no citizenship or clearance clause found")
    return Gate(
        verdict="fail",
        quote=_quote(match, text),
        note="the posting restricts who may apply — check that it covers you",
    )


def _language(text: str, known: set[str]) -> Gate:
    wanted: list[tuple[str, re.Match]] = []
    for match in LANGUAGE_RE.finditer(text):
        language = match.group("lang")
        if language.strip().lower() not in known:
            wanted.append((language, match))
    if not wanted:
        return Gate(verdict="pass", note="no language beyond the posting's own")
    language, match = wanted[0]
    return Gate(
        verdict="fail",
        quote=_quote(match, text),
        note=f"asks for {language}, which the profile does not declare",
    )


def _location(text: str) -> Gate:
    match = LOCATION_RE.search(text)
    if match is None:
        return Gate(verdict="pass", note="no relocation or on-site clause found")
    return Gate(verdict="flag", quote=_quote(match, text),
                note="the posting constrains where the work happens")


def _deadline(text: str, today: date) -> Gate:
    match = DEADLINE_RE.search(text)
    if match is None:
        return Gate(verdict="unknown", note="no closing date stated")
    when = _parse_date(match.group())
    quote = _quote(match, text)
    if when is None:
        return Gate(verdict="flag", quote=quote,
                    note="a closing date is mentioned but could not be parsed")
    if when < today:
        return Gate(verdict="fail", quote=quote, date=when.isoformat(),
                    note=f"closed on {when.isoformat()}")
    return Gate(verdict="pass", quote=quote, date=when.isoformat(),
                note=f"open until {when.isoformat()}")


def _seniority(text: str) -> Gate:
    years = [int(m.group(1)) for m in SENIORITY_RE.finditer(text)]
    demanded = min(years) if years else None
    if demanded is not None and demanded > JUNIOR_YEARS:
        match = SENIORITY_RE.search(text)
        return Gate(verdict="flag", quote=_quote(match, text),
                    note=f"asks for {demanded}+ years of experience")
    title = SENIOR_TITLE_RE.search(text)
    if title is not None:
        return Gate(verdict="flag", quote=_quote(title, text),
                    note="the posting reads as a senior role")
    return Gate(verdict="pass",
                note="no seniority bar found" if demanded is None
                else f"asks for {demanded} year(s) of experience")


# ── Cross-check ──────────────────────────────────────────────────────────────
def cross_check(claimed: Gates, deterministic: Gates) -> Gates:
    """Reconcile a model's gates with the regexes'.

    One rule, and it only goes one way: where the model says ``pass`` and the
    text carries a quote saying otherwise, the verdict becomes ``flag`` and the
    quote comes along. A model is allowed to be *more* careful than the regexes
    (it can read "you must be eligible to work in Singapore" out of a sentence
    no pattern matches); it is not allowed to be less.
    """
    merged: dict[str, Gate] = {}
    for name in GATE_NAMES:
        model_gate: Gate = getattr(claimed, name)
        found: Gate = getattr(deterministic, name)
        if model_gate.verdict in {"pass", "unknown"} and found.blocking and found.quote:
            note = found.note or ""
            merged[name] = Gate(
                verdict="flag",
                quote=found.quote,
                note=(f"model said {model_gate.verdict}; the posting says otherwise"
                      + (f" — {note}" if note else "")),
                date=found.date or model_gate.date,
            )
            continue
        if model_gate.verdict == "unknown" and found.verdict != "unknown":
            merged[name] = found
            continue
        if model_gate.quote is None and found.quote:
            merged[name] = Gate(verdict=model_gate.verdict, quote=found.quote,
                                note=model_gate.note or found.note,
                                date=model_gate.date or found.date)
            continue
        merged[name] = model_gate
    return Gates(**merged)


def gate_dict(gates: Gates) -> dict[str, dict]:
    """The gates as plain JSON, for the run record and the report."""
    return {name: gate.model_dump(mode="json", exclude_none=True)
            for name, gate in gates.items()}


def parse_iso(value: str) -> date | None:
    try:
        return datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError):
        return None


def blocking_names(gates: Gates) -> Sequence[str]:
    return [name for name, _gate in gates.blocking()]
