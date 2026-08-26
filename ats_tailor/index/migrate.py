"""Migrate a v1 index directory (four flat YAML files) to the v2 layout.

The migration is lossless by construction: ``ats_tags`` are partitioned into
skill references and free-text keywords, ``responsibilities``/``impact``/
``bullets`` become structured :class:`~ats_tailor.index.models.Bullet` objects with
pre-filled metrics, and period strings are parsed into structured periods.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .io import save_index_v2, save_role, slugify
from .loaders import load_index
from .models import (
    Bullet,
    Certification,
    Education,
    Evidence,
    Index,
    Period,
    Project,
    Role,
    Skill,
    SkillCategory,
    Summaries,
)
from ..engine.legacy.scoring import keyword_in_text

SKILL_ID_OVERRIDES = {
    "c#": "csharp",
    "c++": "cpp",
    ".net": "dotnet",
    "javascript / node.js": "javascript",
}

HIDDEN_COMMENT = "hidden: no bullets yet"

# Numbers with an optional unit / percent / multiplier / trailing plus:
# "15%", "5,100+", "2M+", "47 ms", "1.000", "5.4x", "27.96 MiB".
# A unit written with a space must look like a unit (contains a capital, or is a
# known lowercase one) so that plain English words are not swallowed.
_NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"
_UNIT_ATTACHED = r"%|[A-Za-z]{1,4}\b"
_UNIT_SPACED = r"(?:[a-zA-Z]{0,3}[A-Z][a-zA-Z]{0,3}|ms|s|x|fps|rps|qps|req|loc|kb|mb|gb|tb)\b"
METRIC_RE = re.compile(
    rf"(?<![\w.,]){_NUMBER}(?:(?:{_UNIT_ATTACHED})|\s(?:{_UNIT_SPACED}))?\+?"
)

_ACRONYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9./+-]*$")
_BARE_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


# ── Report ────────────────────────────────────────────────────────────────────
@dataclass
class ItemReport:
    kind: str
    id: str
    tags: int = 0
    to_skills: int = 0
    to_keywords: int = 0
    bullets: int = 0
    metrics: int = 0
    hidden: bool = False


@dataclass
class MigrationReport:
    src: Path
    dst: Path
    items: list[ItemReport] = field(default_factory=list)
    skills: int = 0
    categories: int = 0
    hidden_roles: list[str] = field(default_factory=list)
    unmapped_tags: list[str] = field(default_factory=list)
    id_collisions: list[str] = field(default_factory=list)
    absorbed_skills_used: dict[str, list[str]] = field(default_factory=dict)

    def render(self) -> str:
        rows = [
            "| kind | id | tags | →skills | →keywords | bullets | metrics | hidden |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for item in self.items:
            rows.append(
                f"| {item.kind} | {item.id} | {item.tags} | {item.to_skills} | "
                f"{item.to_keywords} | {item.bullets} | {item.metrics} | "
                f"{'yes' if item.hidden else ''} |"
            )
        lines = [
            f"Migrated {self.src} → {self.dst}",
            "",
            *rows,
            "",
            f"Skills: {self.skills} across {self.categories} categories",
            f"Hidden roles: {', '.join(self.hidden_roles) or 'none'}",
            f"Id collisions: {', '.join(self.id_collisions) or 'none'}",
            f"Unmapped tags ({len(self.unmapped_tags)}): "
            f"{', '.join(self.unmapped_tags) or 'none'}",
        ]
        if self.absorbed_skills_used:
            lines.append("skills_used folded into role tags:")
            for role_id, values in self.absorbed_skills_used.items():
                lines.append(f"  {role_id}: {', '.join(values)}")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────
def skill_id_for(name: str) -> str:
    """Stable slug for a skill name, with overrides for symbol-heavy names."""
    override = SKILL_ID_OVERRIDES.get(name.strip().lower())
    if override:
        return override
    slug = slugify(name)
    if not slug:
        raise ValueError(f"cannot derive a skill id from name {name!r}")
    return slug


def pick_acronym(aliases: list[str]) -> str | None:
    """Shortest all-caps alias of at most 6 characters, if any."""
    candidates = [
        alias
        for alias in aliases
        if len(alias) <= 6 and alias == alias.upper() and _ACRONYM_RE.match(alias)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda alias: (len(alias), aliases.index(alias)))


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        key = value.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result


def find_metrics(text: str) -> list[str]:
    """Numbers with optional units that a rewrite guardrail must preserve.

    Bare four-digit years are dropped: "…Awards for Machine Learning 2026" is a
    date, not a quantity worth guarding.
    """
    return _dedupe(
        match.group(0).strip()
        for match in METRIC_RE.finditer(text)
        if not _BARE_YEAR_RE.match(match.group(0).strip())
    )


def _skills_in_text(text: str, forms: list[tuple[str, str]]) -> list[str]:
    """Skill ids whose name/alias/acronym occurs in ``text`` (word-boundary)."""
    lowered = text.lower()
    found: list[str] = []
    for form, skill_id in forms:
        if skill_id in found:
            continue
        if keyword_in_text(form, lowered, text):
            found.append(skill_id)
    return found


# ── Skills ────────────────────────────────────────────────────────────────────
def _build_skills(
    skills_data: dict, item_kinds: dict[str, str], report: MigrationReport
) -> tuple[list[SkillCategory], dict[str, str], list[tuple[str, str]]]:
    categories: list[SkillCategory] = []
    by_name: dict[str, str] = {}
    by_acronym: dict[str, str] = {}
    by_alias: dict[str, str] = {}
    forms: list[tuple[str, str]] = []
    seen_ids: dict[str, str] = {}

    for category in skills_data.get("categories", []):
        skills: list[Skill] = []
        for raw in category.get("skills", []):
            name = raw["name"]
            skill_id = skill_id_for(name)
            if skill_id in seen_ids:
                report.id_collisions.append(
                    f"{skill_id}: '{name}' collides with '{seen_ids[skill_id]}'"
                )
            seen_ids[skill_id] = name

            keywords = list(raw.get("ats_keywords", []))
            aliases = _dedupe(
                kw for kw in keywords if kw.strip().lower() != name.strip().lower()
            )
            acronym = pick_acronym(aliases)

            evidence = []
            for ev in raw.get("evidence", []):
                ref = ev["ref"]
                evidence.append(
                    Evidence(
                        ref=ref,
                        type=ev.get("type") or item_kinds.get(ref, "project"),
                        detail=ev.get("detail"),
                    )
                )

            skills.append(
                Skill(
                    id=skill_id,
                    name=name,
                    proficiency=raw["proficiency"],
                    aliases=aliases,
                    acronym=acronym,
                    evidence=evidence,
                )
            )

            by_name.setdefault(name.strip().lower(), skill_id)
            if acronym:
                by_acronym.setdefault(acronym.strip().lower(), skill_id)
            for alias in aliases:
                by_alias.setdefault(alias.strip().lower(), skill_id)
            for form in [name, *aliases, *([acronym] if acronym else [])]:
                forms.append((form, skill_id))

        categories.append(SkillCategory(category=category["name"], skills=skills))

    if report.id_collisions:
        raise ValueError("skill id collisions:\n- " + "\n- ".join(report.id_collisions))

    lookup = {**by_alias, **by_acronym, **by_name}
    return categories, lookup, forms


def _partition_tags(
    tags: list[str], lookup: dict[str, str]
) -> tuple[list[str], list[str]]:
    skill_ids: list[str] = []
    keywords: list[str] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            continue
        skill_id = lookup.get(tag.strip().lower())
        if skill_id is not None:
            if skill_id not in skill_ids:
                skill_ids.append(skill_id)
        else:
            keywords.append(tag.strip())
    return skill_ids, _dedupe(keywords)


def _make_bullets(
    item_id: str, texts: list[str], kind: str, suffix: str, forms: list[tuple[str, str]]
) -> list[Bullet]:
    return [
        Bullet(
            id=f"{item_id}.{suffix}{i}",
            kind=kind,
            text=text,
            skills=_skills_in_text(text, forms),
            metrics=find_metrics(text),
        )
        for i, text in enumerate(texts, start=1)
        if isinstance(text, str) and text.strip()
    ]


# ── Migration ─────────────────────────────────────────────────────────────────
def build_v2_index(src_dir: str | Path) -> tuple[Index, MigrationReport]:
    """Load a v1 index and project it onto the v2 model without writing anything."""
    src_dir = Path(src_dir)
    skills_data, projects_data, exp_data, certs_data, summary_data = load_index(src_dir)
    report = MigrationReport(src=src_dir, dst=Path())

    item_kinds: dict[str, str] = {}
    for role in exp_data.get("roles", []):
        item_kinds[role["id"]] = "role"
    for project in projects_data.get("projects", []):
        item_kinds[project["id"]] = "project"
    for cert in certs_data.get("certifications", []):
        item_kinds[cert["id"]] = "cert"

    categories, lookup, forms = _build_skills(skills_data, item_kinds, report)
    report.categories = len(categories)
    report.skills = sum(len(category.skills) for category in categories)

    unmapped: list[str] = []

    projects: list[Project] = []
    for raw in projects_data.get("projects", []):
        item_id = raw["id"]
        tags = list(raw.get("ats_tags", []))
        skill_ids, keywords = _partition_tags(tags, lookup)
        unmapped.extend(keywords)
        bullets = _make_bullets(
            item_id, list(raw.get("responsibilities", [])), "responsibility", "r", forms
        ) + _make_bullets(item_id, list(raw.get("impact", [])), "impact", "i", forms)
        projects.append(
            Project(
                id=item_id,
                name=raw["name"],
                type=raw.get("type"),
                repo=raw.get("repo"),
                period=Period.parse(raw["period"]),
                summary=raw.get("summary", ""),
                tech_stack={
                    key: list(values)
                    for key, values in (raw.get("tech_stack") or {}).items()
                },
                skills=skill_ids,
                keywords=keywords,
                bullets=bullets,
            )
        )
        report.items.append(
            ItemReport(
                kind="project",
                id=item_id,
                tags=len(tags),
                to_skills=len(skill_ids),
                to_keywords=len(keywords),
                bullets=len(bullets),
                metrics=sum(len(b.metrics) for b in bullets),
            )
        )

    roles: list[Role] = []
    for raw in exp_data.get("roles", []):
        item_id = raw["id"]
        tags = list(raw.get("ats_tags", []))
        extra = [
            value
            for value in raw.get("skills_used", [])
            if isinstance(value, str)
            and value.strip().lower() not in {t.strip().lower() for t in tags}
        ]
        if extra:
            report.absorbed_skills_used[item_id] = extra
        skill_ids, keywords = _partition_tags(tags + extra, lookup)
        unmapped.extend(keywords)
        bullets = _make_bullets(
            item_id, list(raw.get("bullets", [])), "responsibility", "b", forms
        )
        hidden = not bullets
        if hidden:
            report.hidden_roles.append(item_id)
        roles.append(
            Role(
                id=item_id,
                company=raw["company"],
                title=raw["title"],
                type=raw["type"],
                period=Period.parse(raw["period"]),
                hidden=hidden,
                skills=skill_ids,
                keywords=keywords,
                bullets=bullets,
            )
        )
        report.items.append(
            ItemReport(
                kind="role",
                id=item_id,
                tags=len(tags),
                to_skills=len(skill_ids),
                to_keywords=len(keywords),
                bullets=len(bullets),
                metrics=sum(len(b.metrics) for b in bullets),
                hidden=hidden,
            )
        )

    certifications: list[Certification] = []
    for raw in certs_data.get("certifications", []):
        tags = list(raw.get("ats_keywords", []))
        skill_ids, keywords = _partition_tags(tags, lookup)
        unmapped.extend(keywords)
        certifications.append(
            Certification(
                id=raw["id"],
                name=raw["name"],
                issuer=raw["issuer"],
                date=raw.get("date"),
                skills=skill_ids,
                keywords=keywords,
                relevance=list(raw.get("relevance", [])),
            )
        )
        report.items.append(
            ItemReport(
                kind="cert",
                id=raw["id"],
                tags=len(tags),
                to_skills=len(skill_ids),
                to_keywords=len(keywords),
            )
        )

    education = [
        Education(
            institution=raw["institution"],
            degree=raw["degree"],
            field=raw["field"],
            period=Period.parse(raw["period"]),
        )
        for raw in exp_data.get("education", [])
    ]

    summaries = Summaries(
        default=(summary_data or {}).get("summary", "") or "",
        variants={},
        highlights=list((summary_data or {}).get("highlights", []) or []),
    )

    report.unmapped_tags = _dedupe(unmapped)

    index = Index(
        skills=categories,
        projects=projects,
        roles=roles,
        certifications=certifications,
        education=education,
        summaries=summaries,
    )
    return index, report


def migrate_v1_to_v2(
    src_dir: str | Path,
    dst_dir: str | Path,
    profile_path: str | Path | None = None,
    force: bool = False,
) -> MigrationReport:
    """Convert the v1 index at ``src_dir`` into the v2 layout at ``dst_dir``."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if dst_dir.exists() and any(dst_dir.iterdir()) and not force:
        raise ValueError(f"{dst_dir} is not empty; pass --force to write into it")

    index, report = build_v2_index(src_dir)
    report.dst = dst_dir

    dst_dir.mkdir(parents=True, exist_ok=True)
    save_index_v2(index, dst_dir)
    for role in index.roles:
        if role.hidden:
            save_role(role, dst_dir, eol_comments={"hidden": HIDDEN_COMMENT})

    if profile_path is not None:
        source = Path(profile_path)
        if source.exists():
            (dst_dir / "profile.yaml").write_text(source.read_text())

    return report
