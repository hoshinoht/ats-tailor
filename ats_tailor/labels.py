"""Per-JD ground truth: what a good run for this posting must contain.

One tracked YAML file per job description in the user's repo (``labels/``),
holding the judgements only a human can make — which items *must* be picked,
which must not, and which JD terms are genuinely must-have. ``ats-tailor eval``
scores the pipeline against these, so they are the fixed point that lets a
scoring change be called an improvement rather than a difference.

``labels init`` pre-fills the term lists from :func:`atscheck.auto_terms` and
leaves everything else empty: the scaffold is a starting point to edit, never
an answer key generated from the thing being measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from .index_io import slugify


class Label(BaseModel):
    """Ground truth for one job description."""

    model_config = ConfigDict(extra="forbid")

    jd: str
    company: str
    role: str
    pin_projects: list[str] = Field(default_factory=list)
    pin_roles: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    must_terms: list[str] = Field(default_factory=list)
    nice_terms: list[str] = Field(default_factory=list)
    #: optional full preference order, keyed by ``projects`` / ``roles``
    ranking: dict[str, list[str]] = Field(default_factory=dict)


def slug(jd_path: str | Path) -> str:
    """The label slug for a JD file: its stem, slugified."""
    return slugify(Path(jd_path).stem)


def label_path(labels_dir: str | Path, jd_path: str | Path) -> Path:
    return Path(labels_dir) / f"{slug(jd_path)}.yaml"


def load_labels(labels_dir: str | Path) -> dict[str, Label]:
    """Load every ``*.yaml`` in ``labels_dir``, keyed by file stem."""
    import yaml

    labels_dir = Path(labels_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")
    labels: dict[str, Label] = {}
    for path in sorted(labels_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: file must contain a mapping at the top level")
        labels[path.stem] = Label.model_validate(data)
    return labels


def render_label(label: Label, gaps: Iterable[str] = ()) -> str:
    """The scaffold text: values inline, comments saying what to fill in.

    Terms in ``gaps`` are flagged ``# not in index`` so the user can tell a
    real gap (nothing in the index claims it — decide whether to add evidence
    or drop the term) from a taxonomy term the selection merely has to reach.
    """
    flagged = {g.strip().lower() for g in gaps}

    def block(name: str, values: list[str]) -> str:
        if not values:
            return f"{name}: []"
        return f"{name}:\n" + "\n".join(
            f"  - {_quote(v)}" + ("  # not in index" if v.strip().lower() in flagged else "")
            for v in values
        )

    return "\n".join([
        f"# Ground truth for {label.jd} — hand-edited; see docs/v2-design.md §6.",
        f"jd: {_quote(label.jd)}",
        f"company: {_quote(label.company)}",
        f"role: {_quote(label.role)}",
        "",
        "# Item ids the run must select, and ids it must never select.",
        block("pin_projects", label.pin_projects),
        block("pin_roles", label.pin_roles),
        block("exclude", label.exclude),
        "",
        "# Pre-filled from the JD ∩ index taxonomy; re-tier and prune by hand.",
        block("must_terms", label.must_terms),
        block("nice_terms", label.nice_terms),
        "",
        "# Optional full preference order, e.g.  projects: [carehaven, virtualkopis]",
        "ranking: {}" if not label.ranking else "ranking:\n" + "\n".join(
            f"  {key}: [{', '.join(_quote(v) for v in values)}]"
            for key, values in label.ranking.items()
        ),
        "",
    ])


def _quote(value: str) -> str:
    """Quote only when YAML would otherwise mis-read the scalar."""
    text = str(value)
    if text == "" or text[0] in "-?:,[]{}#&*!|>'\"%@`" or ": " in text or text.endswith(":"):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def init_label(
    jd_path: str | Path,
    company: str,
    role: str,
    labels_dir: str | Path,
    index: object | None = None,
    jd_ref: str | None = None,
    force: bool = False,
) -> Path:
    """Write a scaffold label file for ``jd_path``; returns the path written."""
    from .atscheck import auto_terms, gap_terms

    jd_path = Path(jd_path)
    path = label_path(labels_dir, jd_path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (use --force to overwrite)")

    must: list[str] = []
    nice: list[str] = []
    gaps: list[str] = []
    if index is not None:
        jd_text = jd_path.read_text()
        must, nice = auto_terms(jd_text, index)
        known = {term.lower() for term in [*must, *nice]}
        gaps = [term for term in gap_terms(jd_text, index, extra_known=[company, role])
                if term.lower() not in known]
        nice = sorted([*nice, *gaps], key=str.lower)

    label = Label(
        jd=jd_ref or str(jd_path),
        company=company,
        role=role,
        must_terms=must,
        nice_terms=nice,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_label(label, gaps))
    return path
