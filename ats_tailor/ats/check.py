"""Compile a rendered resume and check that an ATS can actually read it.

Two independent questions, answered against the *compiled PDF* rather than
the LaTeX source, because what an ATS ingests is the PDF's text layer:

``parse_check``
    Structural parseability — see ``docs/v2-design.md`` §2. Contact details
    recoverable by regex, standard headings in template order, one page, and
    every entry carrying a complete ASCII-hyphen date range (en dashes and
    bare single years are a confirmed silent Workday import failure).

``keyword_coverage``
    Which JD terms actually reached the page, and *where* — a term in the
    skills line only is weaker than one that also appears in a bullet. Missing
    terms are split into "the index has it, the selection dropped it" and
    "the index has no claim for it", which is a gap to report, never to stuff.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..engine.legacy.scoring import AMBIGUOUS_KEYWORDS, chunk_text

# ── Section headings ──────────────────────────────────────────────────────────
#: Canonical region name → the heading strings accepted for it, in template
#: order. ``pdftotext`` sees the uppercased form the template renders; the
#: design's target string is the first alternative.
SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("summary", ("SUMMARY", "PROFESSIONAL SUMMARY")),
    ("education", ("EDUCATION",)),
    ("experience", ("EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE")),
    ("projects", ("PROJECTS",)),
    ("skills", ("SKILLS", "TECHNICAL SKILLS")),
    ("certifications", ("CERTIFICATIONS",)),
)

MIN_TEXT_CHARS = 500
STUFFING_LIMIT = 3

_CID_RE = re.compile(r"\(cid:\d+\)")
_REPLACEMENT = "�"
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/([A-Za-z0-9._%-]+)", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9._-]+)", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[•‣●▪*-]\s+")
_TECH_RE = re.compile(r"^\s*Technologies:", re.IGNORECASE)

#: One endpoint of a date range: ``Mon YYYY``, ``YYYY`` or ``Present``.
_ENDPOINT = r"(?:[A-Za-z]{3,9}\.?\s+\d{4}|\d{4}|Present)"
#: A range with *any* separator, so the checker can report the wrong ones.
DATE_RANGE_RE = re.compile(rf"({_ENDPOINT})\s*([-–—])\s*({_ENDPOINT})")
#: A lone date token, used to spot an entry whose range lost an endpoint.
DATE_TOKEN_RE = re.compile(rf"\b(?:[A-Za-z]{{3,9}}\.?\s+\d{{4}}|\d{{4}})\b")


# ── Results ───────────────────────────────────────────────────────────────────
@dataclass
class Check:
    check: str
    ok: bool
    detail: str = ""


@dataclass
class ParseReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.ok]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}


@dataclass
class TermCoverage:
    """One JD term and the four-state verdict of ``docs/v2-design.md`` §2."""

    term: str
    tier: str  # "must" | "nice"
    forms: list[str]
    present: bool
    count: int
    matched_forms: list[str]
    regions: list[str]
    in_skills_and_bullet: bool
    state: str  # covered | synonym-only | missing (have it) | missing (gap)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoverageReport:
    """Scores are ``None``, not zero, when the tier has no terms to cover."""

    terms: list[TermCoverage] = field(default_factory=list)
    must_coverage: float | None = None
    nice_coverage: float | None = None
    context_rate: float | None = None
    stuffing: list[tuple[str, int]] = field(default_factory=list)

    def by_state(self, state: str) -> list[TermCoverage]:
        return [term for term in self.terms if term.state == state]

    def to_dict(self) -> dict[str, Any]:
        return {
            "must_coverage": self.must_coverage,
            "nice_coverage": self.nice_coverage,
            "context_rate": self.context_rate,
            "stuffing": [{"term": t, "count": c} for t, c in self.stuffing],
            "terms": [term.to_dict() for term in self.terms],
        }


@dataclass
class CompileResult:
    pdf_path: Path
    pages: int
    log_path: Path | None = None


class CompileError(RuntimeError):
    """The LaTeX engine failed; the message carries the tail of its log."""


# ── Compilation ───────────────────────────────────────────────────────────────
#: What ``ATS_LATEX_ENGINE`` may say. ``auto`` prefers ``latexmk`` — a TeX Live
#: install is what a laptop has and what every measured baseline was produced
#: with — and falls back to Tectonic, which is the single static binary the
#: container ships instead of a two-gigabyte distribution.
LATEX_ENGINES: tuple[str, ...] = ("auto", "latexmk", "tectonic")

#: The default when neither the caller nor the environment says. Spelled here
#: rather than imported from :mod:`ats_tailor.config` so that a test can set
#: ``ATS_LATEX_ENGINE`` after import and be believed.
DEFAULT_LATEX_ENGINE = "auto"


def resolve_latex_engine(engine: str | None = None) -> str:
    """``latexmk`` or ``tectonic``: the engine a compile would actually use.

    Resolution order is the explicit argument, then ``ATS_LATEX_ENGINE``, then
    :data:`DEFAULT_LATEX_ENGINE`. ``auto`` asks the PATH; a named engine is
    returned whether or not it is installed, because "tectonic is missing" is a
    better error than a silent fall back to a different typesetter than the one
    the caller asked for.
    """
    value = str(
        engine or os.environ.get("ATS_LATEX_ENGINE") or DEFAULT_LATEX_ENGINE
    ).strip().lower()
    if value not in LATEX_ENGINES:
        raise ValueError(
            f"unknown LaTeX engine {value!r}; expected one of: "
            f"{', '.join(LATEX_ENGINES)}"
        )
    if value == "auto":
        return "latexmk" if shutil.which("latexmk") else "tectonic"
    return value


def latex_command(engine: str, tex_name: str) -> list[str]:
    """The argv for one non-interactive compile of ``tex_name`` in its own dir."""
    if engine == "latexmk":
        return ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_name]
    if engine == "tectonic":
        # ``--keep-logs`` because the page count and every compile error are read
        # out of the ``.log``; Tectonic deletes it otherwise. ``--chatter minimal``
        # keeps a service that must not print from printing.
        return ["tectonic", "--keep-logs", "--chatter", "minimal", tex_name]
    raise ValueError(f"unknown LaTeX engine {engine!r}")


def latex_available(engine: str | None = None) -> bool:
    """Whether the engine a compile would use is on the PATH."""
    try:
        return bool(shutil.which(resolve_latex_engine(engine)))
    except ValueError:
        return False


def output_filename(name: str, company: str) -> str:
    """``<First>_<Last>_Resume_<Company>.pdf`` with every non-alnum run as ``_``."""
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    first = parts[0] if parts else "Resume"
    last = parts[-1] if len(parts) > 1 else ""
    pieces = [first, last, "Resume", company]
    slug = "_".join(_alnum(p) for p in pieces if _alnum(p))
    return f"{slug}.pdf"


def _alnum(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def compile_pdf(
    tex_path: str | Path, out_name: str | None = None, *, engine: str | None = None
) -> CompileResult:
    """Compile ``tex_path``, clean the aux files, count the pages.

    ``engine`` is ``auto`` | ``latexmk`` | ``tectonic``, defaulting to
    ``ATS_LATEX_ENGINE`` (see :func:`resolve_latex_engine`). Everything after
    the compile is engine-agnostic: both write ``<stem>.pdf`` and a standard
    TeX ``<stem>.log`` beside the source, and every check downstream reads the
    PDF's text layer rather than anything the typesetter said.

    The log is copied to ``<stem>.compile.log`` *before* ``latexmk -c`` runs,
    since the clean-up takes ``.log`` with it. Tectonic leaves no aux files to
    clean, so it gets no clean-up pass.
    """
    tex_path = Path(tex_path).resolve()
    work_dir = tex_path.parent
    engine = resolve_latex_engine(engine)
    result = subprocess.run(
        latex_command(engine, tex_path.name),
        cwd=work_dir, capture_output=True, text=True,
    )
    log_path = work_dir / f"{tex_path.stem}.log"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    if result.returncode != 0:
        tail = "\n".join(
            (log_text or result.stderr or result.stdout).splitlines()[-25:])
        raise CompileError(f"{engine} failed for {tex_path}:\n{tail}")

    pdf_path = work_dir / f"{tex_path.stem}.pdf"
    if not pdf_path.exists():
        raise CompileError(f"{engine} reported success but {pdf_path} is missing")

    kept_log = work_dir / f"{tex_path.stem}.compile.log"
    if log_text:
        kept_log.write_text(log_text)
    pages = page_count(pdf_path, log_text)
    if engine == "latexmk":
        subprocess.run(["latexmk", "-c", tex_path.name], cwd=work_dir,
                       capture_output=True, text=True)

    if out_name:
        shutil.copyfile(pdf_path, work_dir / out_name)
    return CompileResult(pdf_path=pdf_path, pages=pages,
                         log_path=kept_log if kept_log.exists() else None)


def page_count(pdf_path: Path, log_text: str = "") -> int:
    """``pdfinfo`` when available, else the ``Output written on …`` log line."""
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True)
        match = re.search(r"^Pages:\s+(\d+)", info.stdout, re.MULTILINE)
        if match:
            return int(match.group(1))
    match = re.search(r"Output written on .*?\((\d+) pages?", log_text, re.DOTALL)
    return int(match.group(1)) if match else 0


def extract_text(pdf_path: str | Path) -> str:
    """The PDF's text layer via ``pdftotext -layout -enc UTF-8``.

    ``-enc UTF-8`` is mandatory: Xpdf-derived builds default to Latin-1 and
    turn every non-ASCII glyph into U+FFFD, which would make the replacement
    character check fire on the tool rather than on the PDF.
    """
    pdf_path = Path(pdf_path)
    result = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed for {pdf_path}: {result.stderr.strip()}")
    return result.stdout


# ── Regions ───────────────────────────────────────────────────────────────────
def find_headings(text: str) -> list[tuple[str, int, int]]:
    """``(region, char_offset, line_index)`` for every section heading, in order."""
    lookup = {alt: name for name, alts in SECTIONS for alt in alts}
    found: list[tuple[str, int, int]] = []
    offset = 0
    for i, line in enumerate(text.splitlines(keepends=True)):
        key = line.strip().upper()
        if key in lookup:
            found.append((lookup[key], offset, i))
        offset += len(line)
    return found


def split_regions(text: str) -> dict[str, str]:
    """Split the extracted text into named regions for hit attribution.

    ``header`` … ``certifications`` partition the page. ``bullets`` is derived
    on top of experience + projects: the entry header lines (title + date) and
    the ``Technologies:`` lines are dropped, so "appears in a bullet" means
    genuine prose, not a tech-stack listing.
    """
    headings = find_headings(text)
    lines = text.splitlines(keepends=True)
    regions: dict[str, str] = {}
    if not headings:
        return {"header": text, "bullets": ""}

    regions["header"] = text[: headings[0][1]]
    for i, (name, _offset, line_index) in enumerate(headings):
        start = line_index + 1
        end = headings[i + 1][2] if i + 1 < len(headings) else len(lines)
        regions[name] = "".join(lines[start:end])

    body = regions.get("experience", "") + regions.get("projects", "")
    regions["bullets"] = "".join(
        line + "\n"
        for line in body.splitlines()
        if line.strip()
        and not _TECH_RE.match(line)
        and not DATE_RANGE_RE.search(line)
    )
    return regions


# ── Parse checks ──────────────────────────────────────────────────────────────
def parse_check(
    text: str,
    profile: dict,
    items: Sequence[dict] | None = None,
    pages: int | None = None,
) -> ParseReport:
    """Run every structural check of ``docs/v2-design.md`` §2 over ``text``.

    ``items`` are the entries the run actually rendered — ``{"title", "period"}``
    dicts, optionally ``"kind"`` — so the per-entry date check tests what was
    selected rather than whatever the text happens to look like.
    """
    checks: list[Check] = []
    stripped = "".join(text.split())

    checks.append(Check(
        "text_layer",
        len(stripped) >= MIN_TEXT_CHARS,
        f"{len(stripped)} non-whitespace characters (need >= {MIN_TEXT_CHARS})",
    ))
    cids = _CID_RE.findall(text)
    checks.append(Check(
        "no_cid_markers", not cids,
        "none" if not cids else f"{len(cids)} marker(s): {', '.join(sorted(set(cids))[:5])}",
    ))
    bad = text.count(_REPLACEMENT)
    checks.append(Check(
        "no_replacement_chars", bad == 0,
        "none" if not bad else f"{bad} U+FFFD character(s) — font lacks a Unicode map",
    ))

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    name = str(profile.get("name", "")).strip()
    checks.append(Check(
        "name_first_line",
        bool(name) and name.lower() in first_line.lower(),
        f"first line: {first_line!r}" if first_line else "no text",
    ))

    checks.extend(_contact_checks(text, profile))
    checks.extend(_section_checks(text))
    checks.extend(_date_checks(text, items or []))

    if pages is not None:
        checks.append(Check("single_page", pages == 1, f"{pages} page(s)"))
    return ParseReport(checks)


def _contact_checks(text: str, profile: dict) -> list[Check]:
    checks: list[Check] = []
    email = str(profile.get("email", "")).strip()
    found_emails = _EMAIL_RE.findall(text)
    checks.append(Check(
        "email",
        bool(email) and any(e.lower() == email.lower() for e in found_emails),
        f"found {found_emails}" if found_emails else "no email in text layer",
    ))

    if str(profile.get("phone", "")).strip():
        digits = re.sub(r"\D", "", str(profile["phone"]))
        found = [m for m in _PHONE_RE.findall(text) if re.sub(r"\D", "", m).endswith(digits[-6:])]
        checks.append(Check(
            "phone", bool(found),
            f"found {found[0]!r}" if found else f"profile phone {profile['phone']!r} not recoverable",
        ))

    for label, pattern, key in (
        ("linkedin", _LINKEDIN_RE, "linkedin"),
        ("github", _GITHUB_RE, "github"),
    ):
        handle = str(profile.get(key, "")).strip()
        if not handle:
            continue
        found = any(h.lower() == handle.lower() for h in pattern.findall(text))
        if not found:
            # The contact block wraps: the label can end one line and the URL
            # start the next. Re-join the header only — collapsing the whole
            # page would glue the next heading onto the handle.
            found = any(h.lower() == handle.lower()
                        for h in pattern.findall(_join_header(text)))
        checks.append(Check(
            label, found,
            f"{label}.com handle {handle!r} found" if found
            else f"{label}.com/{handle} not recoverable from the text layer",
        ))
    return checks


def _join_header(text: str) -> str:
    """The contact block with its line wraps removed."""
    headings = find_headings(text)
    header = text[: headings[0][1]] if headings else text
    return re.sub(r"\s*\n\s*", "", header)


def _section_checks(text: str) -> list[Check]:
    headings = find_headings(text)
    seen = [name for name, _o, _l in headings]
    expected = [name for name, _alts in SECTIONS]
    missing = [name for name in expected if name not in seen]
    checks = [Check(
        "sections_present", not missing,
        "all present: " + ", ".join(seen) if not missing
        else f"missing: {', '.join(missing)} (found: {', '.join(seen) or 'none'})",
    )]

    order = [expected.index(name) for name in seen if name in expected]
    offsets = [offset for name, offset, _l in headings if name in expected]
    in_order = order == sorted(order) and offsets == sorted(offsets)
    checks.append(Check(
        "reading_order", in_order,
        "headings in template order" if in_order
        else f"out of order: {', '.join(seen)}",
    ))
    return checks


def _date_checks(text: str, items: Sequence[dict]) -> list[Check]:
    checks: list[Check] = []
    wrong = [
        f"{m.group(1)}{m.group(2)}{m.group(3)}"
        for m in DATE_RANGE_RE.finditer(text)
        if m.group(2) != "-"
    ]
    total = len(DATE_RANGE_RE.findall(text))
    checks.append(Check(
        "date_ranges_ascii", not wrong,
        f"{total} range(s), all ASCII-hyphenated" if not wrong
        else f"non-ASCII separator in: {'; '.join(wrong[:5])}",
    ))

    sections = {name for name, _offset, _line in find_headings(text)}
    if items or sections & {"experience", "projects"}:
        checks.append(Check(
            "entries_known", bool(items),
            f"{len(items)} entr{'y' if len(items) == 1 else 'ies'} to verify"
            if items else
            "no selection.json and no entries parsed from resume.tex — stale output?",
        ))

    if items:
        problems: list[str] = []
        lines = [_squeeze(line) for line in text.splitlines()]
        for item in items:
            title = _squeeze(str(item.get("title", "")))
            if not title:
                continue
            hits = [line for line in lines if title.lower() in line.lower()]
            if not hits:
                problems.append(f"{title!r}: not found in the text layer")
                continue
            if not any(DATE_RANGE_RE.search(line) for line in hits):
                lone = [line for line in hits if DATE_TOKEN_RE.search(line)]
                problems.append(
                    f"{title!r}: date on a separate line" if not lone
                    else f"{title!r}: incomplete range on the entry line"
                )
        checks.append(Check(
            "entry_dates", not problems,
            f"{len(items)} entries carry a complete range on the title line"
            if not problems else "; ".join(problems[:5]),
        ))
    return checks


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ── Fallback entry list ───────────────────────────────────────────────────────
_SECTION_RE = re.compile(r"\\section\{([^}]*)\}")
#: ``\entryhead{\textbf{Title} -- Organisation}{Period}``, optionally followed
#: by the ``\\`` that precedes a project's Technologies line.
_ENTRY_RE = re.compile(
    r"^\\entryhead\{(?P<head>.*)\}\{(?P<period>[^{}]*)\}(?:\\\\)?\s*$")
_TITLE_RE = re.compile(r"\\textbf\{(?P<title>[^}]*)\}")
#: Template section → the entry kind its rows describe.
_ENTRY_SECTIONS = {"education": "education", "experience": "role", "projects": "project"}
_UNESCAPE = {r"\&": "&", r"\%": "%", r"\$": "$", r"\#": "#", r"\_": "_",
             r"\{": "{", r"\}": "}"}


def entries_from_tex(tex: str) -> list[dict]:
    """Rendered education/experience/projects entries recovered from the LaTeX source.

    Only a fallback: ``check`` prefers the run's own ``selection.json``, which
    says what the pipeline chose rather than what a regex can see.
    """
    entries: list[dict] = []
    section = ""
    for line in tex.splitlines():
        heading = _SECTION_RE.search(line)
        if heading:
            section = heading.group(1).strip().lower()
            continue
        if section not in _ENTRY_SECTIONS:
            continue
        match = _ENTRY_RE.match(line.strip())
        if not match:
            continue
        title = _TITLE_RE.search(match.group("head"))
        if title:
            entries.append({
                "kind": _ENTRY_SECTIONS[section],
                "title": _unescape_tex(title.group("title")),
                "period": _unescape_tex(match.group("period")),
            })
    return entries


def _unescape_tex(text: str) -> str:
    for escaped, plain in _UNESCAPE.items():
        text = text.replace(escaped, plain)
    return text.replace(r"\textbackslash{}", "\\").strip()


# ── Keyword coverage ──────────────────────────────────────────────────────────
def count_in_text(keyword: str, text_lower: str, text_original: str = "") -> int:
    """Occurrences of ``keyword`` under ``scoring.keyword_in_text`` semantics."""
    kw = keyword.lower()
    if not kw.replace(" ", "").replace("-", "").isalnum():
        return text_lower.count(kw)
    if kw in AMBIGUOUS_KEYWORDS or (len(kw) <= 2 and kw.isalpha()):
        if not text_original:
            return len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
        return len(re.findall(r"\b" + re.escape(keyword) + r"\b", text_original))
    return len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))


def keyword_coverage(
    text: str,
    terms: dict[str, list[str]],
    index_terms: Iterable[str],
    regions: dict[str, str],
    must: Iterable[str] | None = None,
) -> CoverageReport:
    """Score which JD terms reached the page, in which regions, and why not.

    ``terms`` maps the JD's canonical surface form to every form that counts as
    the same term. A term matched only through an alias is ``synonym-only``:
    the claim is on the page but not in the words the posting used. Terms in
    ``must`` are the must-haves; everything else in ``terms`` is nice-to-have
    (with ``must`` omitted, every term is treated as a must).
    """
    must_set = {t for t in (must if must is not None else terms)}
    known = {str(t).strip().lower() for t in index_terms}

    text_lower = text.lower()
    region_lower = {name: body.lower() for name, body in regions.items()}
    results: list[TermCoverage] = []

    for term, forms in terms.items():
        forms = [f for f in forms if str(f).strip()] or [term]
        counts = {f: count_in_text(f, text_lower, text) for f in forms}
        matched = [f for f, c in counts.items() if c > 0]
        count = sum(counts.values())
        hit_regions = [
            name for name, body in region_lower.items()
            if name != "header" and any(count_in_text(f, body, regions[name]) for f in forms)
        ]
        present = bool(matched)
        exact = any(f.lower() == term.lower() for f in matched)
        if present:
            state = "covered" if exact else "synonym-only"
        elif any(str(f).strip().lower() in known for f in forms):
            state = "missing (have it)"
        else:
            state = "missing (gap)"
        results.append(TermCoverage(
            term=term,
            tier="must" if term in must_set else "nice",
            forms=forms,
            present=present,
            count=count,
            matched_forms=matched,
            regions=hit_regions,
            in_skills_and_bullet="skills" in hit_regions and "bullets" in hit_regions,
            state=state,
        ))

    report = CoverageReport(terms=results)
    report.must_coverage = _ratio([t for t in results if t.tier == "must"])
    report.nice_coverage = _ratio([t for t in results if t.tier == "nice"])
    present_must = [t for t in results if t.tier == "must" and t.present]
    report.context_rate = (
        sum(1 for t in present_must if t.in_skills_and_bullet) / len(present_must)
        if present_must else None
    )
    report.stuffing = sorted(
        ((t.term, t.count) for t in results if t.count > STUFFING_LIMIT),
        key=lambda pair: -pair[1],
    )
    return report


def _ratio(terms: list[TermCoverage]) -> float | None:
    return sum(1 for t in terms if t.present) / len(terms) if terms else None


def _pct(value: float | None, empty: str = "n/a (no terms)") -> str:
    return empty if value is None else f"{value:.0%}"


# ── Term sources ──────────────────────────────────────────────────────────────
def index_term_forms(index: Any) -> dict[str, list[str]]:
    """Canonical term → surface forms, for a v2 ``Index`` or the v1 dicts.

    A v2 skill contributes its name with aliases and acronym as forms; every
    item ``keywords`` entry is its own single-form term. A v1 index has no
    canonical/alias distinction, so each ``ats_keyword``/``ats_tag`` stands
    alone.
    """
    forms: dict[str, list[str]] = {}

    def add(term: str, surface: Sequence[str] | None = None) -> None:
        term = str(term).strip()
        if not term:
            return
        merged = forms.setdefault(term, [])
        for value in [term, *(surface or [])]:
            value = str(value).strip()
            if value and value.lower() not in {m.lower() for m in merged}:
                merged.append(value)

    if hasattr(index, "all_skills"):  # v2 Index
        for skill in index.all_skills():
            add(skill.name, skill.surface_forms())
        for item in [*index.projects, *index.roles, *index.certifications]:
            for keyword in item.keywords:
                add(keyword)
        return forms

    skills_data, projects_data, exp_data, certs_data = _legacy_parts(index)
    for category in skills_data.get("categories", []):
        for skill in category.get("skills", []):
            add(skill.get("name", ""), skill.get("ats_keywords", []))
    for project in projects_data.get("projects", []):
        for tag in project.get("ats_tags", []):
            add(tag)
    for role in exp_data.get("roles", []):
        for tag in role.get("ats_tags", []):
            add(tag)
    for cert in certs_data.get("certifications", []):
        for keyword in cert.get("ats_keywords", []):
            add(keyword)
    return forms


def _legacy_parts(index: Any) -> tuple[dict, dict, dict, dict]:
    if isinstance(index, (tuple, list)):
        return index[0], index[1], index[2], index[3]
    if isinstance(index, dict):
        return (
            index.get("skills", {}), index.get("projects", {}),
            index.get("experience", {}), index.get("certifications", {}),
        )
    raise TypeError(f"cannot read index terms from {type(index).__name__}")


def form_lookup(terms: dict[str, list[str]]) -> dict[str, list[str]]:
    """Lowercased surface form → the full form list of the entry that owns it.

    One form can belong to several taxonomy entries (a v1 index lists
    "PostgreSQL" both as its own skill and among SQL's aliases); the entry
    whose canonical name *is* that form wins, otherwise the first in index
    order.
    """
    lookup: dict[str, list[str]] = {}
    for term, forms in terms.items():
        canonical = term.strip().lower()
        for form in [term, *forms]:
            key = form.strip().lower()
            if key and (key not in lookup or key == canonical):
                lookup[key] = list(forms) or [term]
    return lookup


def resolve_terms(
    requested: Iterable[str], index_terms: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Requested term → its own spelling first, then every form the index knows.

    A label may name a term any way the JD did — the posting's surface form, an
    alias, or the canonical name — and coverage still has to test all of them
    while judging exactness against the requested spelling.
    """
    lookup = form_lookup(index_terms)
    resolved: dict[str, list[str]] = {}
    for term in requested:
        term = str(term).strip()
        if not term or any(term.lower() == seen.lower() for seen in resolved):
            continue
        others = lookup.get(term.lower(), [])
        resolved[term] = [term] + [f for f in others if f.strip().lower() != term.lower()]
    return resolved


def surface_form_set(terms: dict[str, list[str]]) -> set[str]:
    """Every form the index can express, canonical names and aliases alike.

    This is what decides ``missing (have it)`` versus ``missing (gap)``: a JD
    asking for "K8s" is supported by an index whose Kubernetes skill lists that
    alias, even though no canonical term is spelled that way.
    """
    return {form for forms in terms.values() for form in forms} | set(terms)


#: A heading names its section early. Allowing three leading words covers
#: "Job Requirements:" and "What we are looking for" while keeping a mid-
#: sentence "(PostgreSQL preferred)" from re-tiering the section it sits in.
_HEADING_PREFIX = r"^\W*(?:[\w/&'-]+\W+){0,3}"

#: Headings introducing what the posting *requires*. Checked after the
#: nice-to-have patterns, so "Preferred Qualifications" reads as nice.
MUST_HEADER_RE = re.compile(
    _HEADING_PREFIX + r"(?:requirement|qualification|looking for|look for|"
    r"what you.?ll need|what you need|must have|you have|you bring|you should have|"
    r"we require|skills|about you|who you are|your profile|tech stack)",
    re.IGNORECASE,
)
#: Headings introducing what the posting would merely *like*. "plus" only
#: counts at the end ("Bonus / a plus"), never as "X is a plus" inside a line.
NICE_HEADER_RE = re.compile(
    _HEADING_PREFIX + r"(?:nice.to.have|good.to.have|bonus|preferred|advantage|"
    r"desirable)|\bplus\s*$",
    re.IGNORECASE,
)


def classify_jd_sections(chunks: Sequence[str]) -> list[str]:
    """Label each JD chunk ``must`` / ``nice`` / ``body`` by the header above it.

    Deliberately separate from ``scoring.compute_jd_weights``: that function
    feeds the *ranking* and widening it would move every score, while this one
    only decides which tier a term lands in. It also recognises the
    "What we are looking for" phrasing that Workday postings use, which the
    scoring regex does not.
    """
    tiers: list[str] = []
    current = "body"
    for chunk in chunks:
        text = chunk.strip()
        marked = ("nice" if NICE_HEADER_RE.search(text) else
                  "must" if MUST_HEADER_RE.search(text) else None)
        if _heading_shape(text):
            if marked:
                current = marked
            elif _strong_heading(text) or _title_cased(text):
                # An unrecognised heading ends the section above it; a
                # sentence-cased fragment is left alone, since it is more
                # likely an unbulleted requirement than a new section.
                current = "body"
        tiers.append(current)
    return tiers


def _strong_heading(text: str) -> bool:
    """A trailing colon or a leading ``#`` settles it, whatever the wording."""
    return bool(text) and (text.startswith("#") or text.endswith(":"))


def _heading_shape(text: str) -> bool:
    """Short and unpunctuated enough to be a heading at all."""
    if _strong_heading(text):
        return True
    words = text.split()
    return bool(words) and len(words) <= 8 and text[-1] not in ".!?,;"


def _title_cased(text: str) -> bool:
    """Mostly capitalised words — "Who We Look For", not "Familiar with Python"."""
    words = text.split()
    if not words:
        return False
    return sum(1 for word in words if word[:1].isupper()) / len(words) > 0.6


def auto_terms(jd_text: str, index: Any) -> tuple[list[str], list[str]]:
    """Baseline term labels: taxonomy terms the JD mentions, in *its* wording.

    Each term is returned as the surface form the posting actually uses — the
    JD says "K8s", the label says ``K8s`` — because that is what
    ``keyword_coverage`` calls ``covered``; rendering "Kubernetes" instead is
    a ``synonym-only`` result the renderer should fix, and collapsing to the
    canonical name here would hide it.

    A term is a must-have when any of its forms appears under a requirements
    header (:func:`classify_jd_sections`); otherwise it is nice-to-have. This
    is what ``labels init`` pre-fills for the user to edit.
    """
    forms = index_term_forms(index)
    chunks = chunk_text(jd_text)
    tiers = classify_jd_sections(chunks)
    must_text = "\n".join(c for c, tier in zip(chunks, tiers) if tier == "must")
    must_lower = must_text.lower()
    jd_lower = jd_text.lower()

    tiered: dict[str, str] = {}   # surface form -> tier, first spelling wins
    seen: dict[str, str] = {}     # lowercased form -> the spelling kept
    for surface in forms.values():
        counts = {f: count_in_text(f, jd_lower, jd_text) for f in surface}
        present = {f: c for f, c in counts.items() if c}
        if not present:
            continue
        # The JD's own wording wins: most occurrences, ties to the earlier
        # (more canonical) form.
        best = max(present, key=lambda f: (present[f], -surface.index(f)))
        tier = ("must" if must_text and any(
            count_in_text(f, must_lower, must_text) for f in surface) else "nice")
        key = best.lower()
        if key in seen:
            # Two taxonomy entries can share a surface form; must wins.
            if tier == "must":
                tiered[seen[key]] = "must"
            continue
        seen[key] = best
        tiered[best] = tier

    must = sorted((t for t, tier in tiered.items() if tier == "must"), key=str.lower)
    nice = sorted((t for t, tier in tiered.items() if tier == "nice"), key=str.lower)
    return must, nice


#: Words ``llm.detect_coverage_gaps`` surfaces that are English or posting
#: boilerplate rather than technology. Kept here (not in ``llm``) because it is
#: tuned for label scaffolds, where a false keep costs the user one deletion
#: and a false drop hides a real gap.
GAP_STOPLIST = {
    # months and their abbreviations, weekdays
    *[m.lower() for m in (
        "January February March April May June July August September October "
        "November December Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec "
        "Monday Tuesday Wednesday Thursday Friday Saturday Sunday").split()],
    # posting metadata and section furniture
    "additional", "additionally", "allowance", "attributes", "availability",
    "available", "benefits", "case", "category", "competencies", "consideration",
    "cost", "coursework", "date", "description", "duration", "education",
    "expected", "final", "group", "h2", "here", "highlight", "internship", "job",
    "main", "minimum", "nice", "outcomes", "overview", "partner", "path",
    "personal", "posting", "preferred", "presentation", "prior", "profile",
    "promotion", "qualifications", "ref", "requirement", "requisition",
    "responsibility", "start", "stock", "students", "summary", "talent",
    "undergraduate", "app", "code", "degree", "knowledge", "strong", "good",
    "experience", "team", "ability", "requirements",
    # degrees and study
    "bachelor", "bachelors", "master", "masters", "ms", "enrolled", "graduating",
    "pursuing", "english", "dietetics", "games",
    # generic English the capitalised-word regex picks up
    "able", "artificial", "assist", "attending", "awareness", "basic", "being",
    "benchmark", "builder", "collaboration", "collaborative", "combining",
    "comfort", "consulting", "context", "creativity", "curiosity", "curious",
    "currently", "deeply", "designing", "engineers", "enhancement", "establish",
    "excellent", "exchange", "exploration", "explore", "exploring", "familiar",
    "familiarity", "family", "flexible", "foundational", "gen", "genuine",
    "headquartered", "hero", "identify", "improvement", "integrate",
    "integrating", "integrity", "intelligence", "interest", "interested",
    "internet", "inventor", "leveraging", "look", "middle", "motivation",
    "operational", "opportunities", "participate", "participating", "passionate",
    "pilot", "power", "practical", "proactivity", "problem", "product",
    "production", "professional", "proficient", "project", "prompt", "prototype",
    "refine", "sharing", "simultaneous", "smart", "solid", "stay", "studio",
    "succeed", "sustenance", "tech", "together", "transcribing", "users",
    "willingness", "works", "write", "dev", "docs", "se", "dh", "net",
    # places and organisations
    "africa", "america", "asia", "avanade", "berlin", "china", "east", "europe",
    "frankfurt", "germany", "latin", "gts", "mdax", "healthtech", "icf",
    "infocomm", "microsoft", "shenzhen", "yemeksepeti", "rail", "site",
    "digitalisation", "modernisation", "operations", "infrastructure",
}


def gap_terms(jd_text: str, index: Any, extra_known: Sequence[str] = ()) -> list[str]:
    """Tech-looking JD terms with no index claim behind them.

    ``labels init`` seeds these into ``nice_terms`` so a scaffold can produce
    ``missing (gap)`` verdicts at all — terms drawn from the taxonomy never
    can, and a gap the user never sees is a gap they cannot decide about.
    """
    from ..index.io import to_legacy
    from ..llm import detect_coverage_gaps

    if hasattr(index, "all_skills"):
        skills_data, projects_data, exp_data, certs_data, _ = to_legacy(index)
    else:
        skills_data, projects_data, exp_data, certs_data = _legacy_parts(index)

    found = detect_coverage_gaps(
        jd_text, skills_data, projects_data, exp_data, certs_data,
        extra_known=list(extra_known),
    )
    return [term for term in found
            if len(term) > 1 and term.lower() not in GAP_STOPLIST]


# ── Reports ───────────────────────────────────────────────────────────────────
def write_reports(
    out_dir: str | Path,
    parse_report: ParseReport,
    coverage_report: CoverageReport | None = None,
) -> tuple[Path, Path]:
    """Write ``ats_check.md`` and ``ats_check.json`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "ats_check.md"
    json_path = out_dir / "ats_check.json"
    md_path.write_text(render_report(parse_report, coverage_report))
    payload: dict[str, Any] = {"parse": parse_report.to_dict()}
    if coverage_report is not None:
        payload["coverage"] = coverage_report.to_dict()
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return md_path, json_path


def render_report(
    parse_report: ParseReport, coverage_report: CoverageReport | None = None
) -> str:
    lines = ["# ATS check", "", "## Parse", "", "| check | result | detail |",
             "|---|---|---|"]
    for check in parse_report.checks:
        # A detail can quote page text; an unescaped pipe would break the table.
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.check} | {'PASS' if check.ok else 'FAIL'} | {detail} |")
    lines += ["", f"**{'PASS' if parse_report.ok else 'FAIL'}** — "
              f"{len(parse_report.checks) - len(parse_report.failures)}"
              f"/{len(parse_report.checks)} checks passed."]

    if coverage_report is not None:
        lines += ["", "## Keyword coverage", "",
                  f"- must coverage: {_pct(coverage_report.must_coverage, 'n/a')} "
                  f"({sum(1 for t in coverage_report.terms if t.tier == 'must')} terms)",
                  f"- nice coverage: {_pct(coverage_report.nice_coverage, 'n/a')} "
                  f"({sum(1 for t in coverage_report.terms if t.tier == 'nice')} terms)",
                  f"- context rate (must-terms in skills *and* a bullet): "
                  f"{_pct(coverage_report.context_rate)}", ""]
        lines += ["| term | tier | state | count | regions |", "|---|---|---|---|---|"]
        for term in sorted(coverage_report.terms,
                           key=lambda t: (t.tier != "must", t.state, t.term.lower())):
            lines.append(
                f"| {term.term} | {term.tier} | {term.state} | {term.count} | "
                f"{', '.join(term.regions) or '—'} |"
            )
        if coverage_report.stuffing:
            lines += ["", f"### Stuffing warnings (> {STUFFING_LIMIT} occurrences)", ""]
            lines += [f"- {term}: {count}" for term, count in coverage_report.stuffing]
        gaps = coverage_report.by_state("missing (gap)")
        if gaps:
            lines += ["", "### Gaps (no index claim — report, never stuff)", "",
                      ", ".join(t.term for t in gaps)]
    return "\n".join(lines) + "\n"


def summarize(parse_report: ParseReport, coverage_report: CoverageReport | None) -> str:
    """The one-screen summary the CLI prints."""
    lines = [f"parse: {'PASS' if parse_report.ok else 'FAIL'} "
             f"({len(parse_report.checks) - len(parse_report.failures)}"
             f"/{len(parse_report.checks)} checks)"]
    for check in parse_report.failures:
        lines.append(f"  FAIL {check.check}: {check.detail}")
    if coverage_report is not None:
        lines.append(
            f"coverage: must {_pct(coverage_report.must_coverage)}, "
            f"nice {_pct(coverage_report.nice_coverage)}, "
            f"context {_pct(coverage_report.context_rate)}"
        )
        for state in ("covered", "synonym-only", "missing (have it)", "missing (gap)"):
            hits = coverage_report.by_state(state)
            if hits:
                shown = ", ".join(t.term for t in hits[:8])
                more = f" (+{len(hits) - 8})" if len(hits) > 8 else ""
                lines.append(f"  {state:<18} {len(hits):>3}  {shown}{more}")
        if coverage_report.stuffing:
            lines.append("  stuffing: " + ", ".join(
                f"{term} x{count}" for term, count in coverage_report.stuffing))
    return "\n".join(lines)
