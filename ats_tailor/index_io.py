"""Read and write the v2 per-entity index, and project it back to v1 dicts.

The v2 layout (``docs/v2-design.md`` §3)::

    index/
      .schema/<kind>.schema.json   # generated; targets of the editor $schema hints
      education.yaml
      summaries.yaml
      certifications.yaml
      skills/<category-slug>.yaml
      roles/<id>.yaml
      projects/<id>.yaml

Writes go through ``ruamel.yaml`` in round-trip mode so hand-written comments
and key order survive edits made by the CLI or the UI.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from .models import (
    Certification,
    CertificationsFile,
    Education,
    EducationFile,
    Index,
    Project,
    Role,
    Skill,
    SkillCategory,
    Summaries,
)

#: Schemas live inside the index so an index directory is self-contained and an
#: editor resolves the hint without anything else being checked out alongside it.
SCHEMA_DIRNAME = ".schema"
SCHEMA_COMMENT = "# yaml-language-server: $schema={rel}" + SCHEMA_DIRNAME + "/{kind}.schema.json"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_HINT_RE = re.compile(r"^# yaml-language-server:.*$", re.MULTILINE)

#: kind -> the model describing one whole file of that kind.
SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "index": Index,
    "project": Project,
    "role": Role,
    "skills": SkillCategory,
    "certifications": CertificationsFile,
    "education": EducationFile,
    "summaries": Summaries,
}

_BARE_YEAR_RE = re.compile(r"^\d{4}$")


def write_schemas(out_dir: str | Path) -> list[Path]:
    """Write one JSON Schema file per top-level kind into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, model in SCHEMA_MODELS.items():
        schema = {"$schema": JSON_SCHEMA_DIALECT, **model.model_json_schema()}
        path = out_dir / f"{kind}.schema.json"
        path.write_text(json.dumps(schema, indent=2) + "\n")
        written.append(path)
    return written


def write_index_schemas(index_dir: str | Path) -> list[Path]:
    """Write the schemas into ``<index_dir>/.schema/`` so the index is self-contained."""
    return write_schemas(Path(index_dir) / SCHEMA_DIRNAME)


def _yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096  # never re-wrap long bullet text
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def slugify(text: str) -> str:
    """Lowercase, ``/`` and whitespace to ``-``, drop other punctuation."""
    slug = text.strip().lower().replace("/", "-")
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^a-z0-9-]+", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


# ── Layout detection and loading ──────────────────────────────────────────────
def detect_layout(index_dir: str | Path) -> str:
    """Return ``"v2"`` if ``index_dir/projects/`` is a directory, else ``"v1"``."""
    return "v2" if (Path(index_dir) / "projects").is_dir() else "v1"


def _read_yaml(path: Path) -> Any:
    yaml = _yaml()
    with open(path) as handle:
        data = yaml.load(handle)
    if data is None:
        raise ValueError(f"YAML file is empty: {path}")
    return data


def _validate(model: type[BaseModel], data: Any, path: Path) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: file must contain a mapping at the top level")
    try:
        return model.model_validate(_plain(data))
    except ValidationError as exc:
        raise ValueError(_format_errors(path, exc)) from exc


def _format_errors(where: Path, exc: ValidationError) -> str:
    """Readable one-line-per-problem rendering of a pydantic error."""
    messages: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        messages.extend(
            f"{location}: {line}" if location else line
            for line in message.splitlines()
        )
    return f"{where}:\n- " + "\n- ".join(messages)


def _plain(value: Any) -> Any:
    """Strip ruamel comment wrappers so pydantic sees plain containers."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def load_index_v2(index_dir: str | Path) -> Index:
    """Load a v2 index directory into a validated :class:`Index`."""
    index_dir = Path(index_dir)
    if detect_layout(index_dir) != "v2":
        raise ValueError(f"{index_dir}: not a v2 index (no projects/ directory)")

    categories: list[SkillCategory] = []
    skills_dir = index_dir / "skills"
    if skills_dir.is_dir():
        for path in sorted(skills_dir.glob("*.yaml")):
            categories.append(_validate(SkillCategory, _read_yaml(path), path))

    projects: list[Project] = []
    for path in sorted((index_dir / "projects").glob("*.yaml")):
        project = _validate(Project, _read_yaml(path), path)
        if project.id != path.stem:
            raise ValueError(
                f"{path}: filename stem must equal the entity id (id: {project.id})"
            )
        projects.append(project)

    roles: list[Role] = []
    roles_dir = index_dir / "roles"
    if roles_dir.is_dir():
        for path in sorted(roles_dir.glob("*.yaml")):
            role = _validate(Role, _read_yaml(path), path)
            if role.id != path.stem:
                raise ValueError(
                    f"{path}: filename stem must equal the entity id (id: {role.id})"
                )
            roles.append(role)

    certifications: list[Certification] = []
    certs_path = index_dir / "certifications.yaml"
    if certs_path.exists():
        certifications = _validate(
            CertificationsFile, _read_yaml(certs_path), certs_path
        ).certifications

    education: list[Education] = []
    edu_path = index_dir / "education.yaml"
    if edu_path.exists():
        education = _validate(EducationFile, _read_yaml(edu_path), edu_path).education

    summaries = Summaries()
    summaries_path = index_dir / "summaries.yaml"
    if summaries_path.exists():
        summaries = _validate(Summaries, _read_yaml(summaries_path), summaries_path)

    try:
        return Index(
            skills=categories,
            projects=projects,
            roles=roles,
            certifications=certifications,
            education=education,
            summaries=summaries,
        )
    except ValidationError as exc:
        raise ValueError(_format_errors(index_dir, exc)) from exc


# ── Writing ───────────────────────────────────────────────────────────────────
def _to_yaml_data(model: BaseModel, always: tuple[str, ...] = ()) -> Any:
    """Model → plain containers in field-declaration order, compacted.

    Fields left at their default are omitted, so a bullet with no skills,
    metrics or variants writes as just ``id``/``kind``/``text``. Names in
    ``always`` are re-added even when defaulted, so a file always carries the
    key that says what it holds.
    """
    data = model.model_dump(mode="python", exclude_none=True, exclude_defaults=True)
    for key in always:
        data.setdefault(key, getattr(model, key))
    ordered = {key: data[key] for key in type(model).model_fields if key in data}
    return _decorate(ordered)


def _decorate(value: Any) -> Any:
    """Render period mappings inline and quote bare-year endpoints."""
    if isinstance(value, dict):
        if set(value) == {"start", "end"} and all(
            isinstance(v, str) for v in value.values()
        ):
            period = CommentedMap()
            for key in ("start", "end"):
                raw = value[key]
                period[key] = (
                    DoubleQuotedScalarString(raw) if _BARE_YEAR_RE.match(raw) else raw
                )
            period.fa.set_flow_style()
            return period
        return {key: _decorate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decorate(item) for item in value]
    return value


def _merge_into(target: Any, source: Any) -> Any:
    """Write ``source`` into ``target`` in place, preserving comments and order."""
    if isinstance(target, dict) and isinstance(source, dict):
        for key in [k for k in target if k not in source]:
            del target[key]
        for key, value in source.items():
            if key in target:
                target[key] = _merge_into(target[key], value)
            else:
                target[key] = value
        return target
    if isinstance(target, list) and isinstance(source, list):
        for i, value in enumerate(source):
            if i < len(target):
                target[i] = _merge_into(target[i], value)
            else:
                target.append(value)
        while len(target) > len(source):
            target.pop()
        return target
    return source


def _write_yaml(
    path: Path, data: Any, kind: str, eol_comments: dict[str, str] | None = None
) -> None:
    """Round-trip update ``path`` with ``data``, refreshing its schema hint.

    An existing file is loaded with ruamel and updated in place so comments and
    key order survive; keys absent from ``data`` are dropped. The
    ``yaml-language-server`` header is always rewritten to the current target,
    so moving the schema location fixes every file on the next write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = _yaml()
    if path.exists():
        existing = yaml.load(path.read_text())
        if existing is not None:
            data = _merge_into(existing, data)
    data = _attach_comments(data, eol_comments)

    depth = len(path.parent.relative_to(_index_root(path)).parts)
    header = SCHEMA_COMMENT.format(rel="../" * depth, kind=kind)
    buffer = io.StringIO()
    yaml.dump(data, buffer)
    path.write_text(_with_schema_header(buffer.getvalue(), header))


def _with_schema_header(text: str, header: str) -> str:
    """Replace the file's schema hint, or add one if it has none."""
    if _SCHEMA_HINT_RE.search(text):
        return _SCHEMA_HINT_RE.sub(lambda _m: header, text, count=1)
    return f"{header}\n\n{text}"


def _attach_comments(data: Any, eol_comments: dict[str, str] | None) -> Any:
    """Attach end-of-line comments to top-level keys, without clobbering existing ones."""
    if not eol_comments or not isinstance(data, dict):
        return data
    if not isinstance(data, CommentedMap):
        data = CommentedMap(data)
    for key, text in eol_comments.items():
        if key in data and key not in getattr(data.ca, "items", {}):
            data.yaml_add_eol_comment(text, key)
    return data


def _index_root(path: Path) -> Path:
    """The index directory a file belongs to (``skills``/``roles``/``projects`` are nested)."""
    return path.parent.parent if path.parent.name in {"skills", "roles", "projects"} else path.parent


def save_project(project: Project, index_dir: str | Path) -> Path:
    path = Path(index_dir) / "projects" / f"{project.id}.yaml"
    _write_yaml(path, _to_yaml_data(project), "project")
    return path


def save_role(
    role: Role, index_dir: str | Path, eol_comments: dict[str, str] | None = None
) -> Path:
    path = Path(index_dir) / "roles" / f"{role.id}.yaml"
    _write_yaml(path, _to_yaml_data(role), "role", eol_comments)
    return path


def save_skill_category(category: SkillCategory, index_dir: str | Path) -> Path:
    path = Path(index_dir) / "skills" / f"{slugify(category.category)}.yaml"
    _write_yaml(path, _to_yaml_data(category), "skills")
    return path


def save_certifications(
    certifications: Iterable[Certification], index_dir: str | Path
) -> Path:
    path = Path(index_dir) / "certifications.yaml"
    data = _to_yaml_data(
        CertificationsFile(certifications=list(certifications)), always=("certifications",)
    )
    _write_yaml(path, data, "certifications")
    return path


def save_education(education: Iterable[Education], index_dir: str | Path) -> Path:
    path = Path(index_dir) / "education.yaml"
    data = _to_yaml_data(EducationFile(education=list(education)), always=("education",))
    _write_yaml(path, data, "education")
    return path


def save_summaries(summaries: Summaries, index_dir: str | Path) -> Path:
    path = Path(index_dir) / "summaries.yaml"
    _write_yaml(path, _to_yaml_data(summaries, always=("default",)), "summaries")
    return path


def save_index_v2(index: Index, index_dir: str | Path, schemas: bool = True) -> None:
    """Write every entity of ``index`` into the v2 layout under ``index_dir``."""
    index_dir = Path(index_dir)
    if schemas:
        write_index_schemas(index_dir)
    # Create the entity directories even when empty: detect_layout keys off
    # projects/, so the layout it writes must be detectable as v2.
    for name in ("skills", "projects", "roles"):
        (index_dir / name).mkdir(parents=True, exist_ok=True)
    for category in index.skills:
        save_skill_category(category, index_dir)
    for project in index.projects:
        save_project(project, index_dir)
    for role in index.roles:
        save_role(role, index_dir)
    save_certifications(index.certifications, index_dir)
    save_education(index.education, index_dir)
    save_summaries(index.summaries, index_dir)


# ── v1 compatibility projection ───────────────────────────────────────────────
def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        key = value.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _tags_for(
    keywords: list[str],
    skill_ids: Iterable[str],
    bullet_skill_ids: Iterable[str],
    skills: dict[str, Skill],
) -> list[str]:
    """Legacy ``ats_tags``: keywords plus every surface form of referenced skills."""
    tags: list[str] = list(keywords)
    for skill_id in [*skill_ids, *bullet_skill_ids]:
        skill = skills.get(skill_id)
        if skill is not None:
            tags.extend(skill.surface_forms())
    return _dedupe(tags)


def to_legacy(index: Index) -> tuple[dict, dict, dict, dict, dict | None]:
    """Project a v2 :class:`Index` onto the five v1 dicts the pipeline consumes."""
    skills = index.skill_map()

    skills_data = {
        "categories": [
            {
                "name": category.category,
                "skills": [
                    {
                        "name": skill.name,
                        "proficiency": skill.proficiency,
                        "ats_keywords": skill.surface_forms(),
                        "evidence": [
                            {
                                key: value
                                for key, value in (
                                    ("ref", ev.ref),
                                    ("type", ev.type),
                                    ("detail", ev.detail),
                                )
                                if value is not None
                            }
                            for ev in skill.evidence
                        ],
                    }
                    for skill in category.skills
                ],
            }
            for category in index.skills
        ]
    }

    projects_data = {
        "projects": [
            {
                "id": project.id,
                "name": project.name,
                **({"type": project.type} if project.type else {}),
                **({"repo": project.repo} if project.repo else {}),
                "period": project.period.format(),
                "summary": project.summary,
                "tech_stack": {
                    key: list(values) for key, values in project.tech_stack.items()
                },
                "responsibilities": [
                    b.text for b in project.bullets if b.kind == "responsibility"
                ],
                "impact": [b.text for b in project.bullets if b.kind == "impact"],
                "ats_tags": _tags_for(
                    project.keywords,
                    project.skills,
                    [s for b in project.bullets for s in b.skills],
                    skills,
                ),
            }
            for project in index.projects
        ]
    }

    exp_data = {
        "roles": [
            {
                "id": role.id,
                "company": role.company,
                "title": role.title,
                "period": role.period.format(),
                "type": role.type,
                "skills_used": _dedupe(
                    skills[s].name for s in role.skills if s in skills
                ),
                "bullets": [b.text for b in role.bullets],
                "ats_tags": _tags_for(
                    role.keywords,
                    role.skills,
                    [s for b in role.bullets for s in b.skills],
                    skills,
                ),
            }
            for role in index.roles
            if not role.hidden
        ],
        "education": [
            {
                "institution": edu.institution,
                "degree": edu.degree,
                "field": edu.field,
                "period": edu.period.format(),
            }
            for edu in index.education
        ],
    }

    certs_data = {
        "certifications": [
            {
                "id": cert.id,
                "name": cert.name,
                "issuer": cert.issuer,
                **({"date": cert.date} if cert.date else {}),
                "ats_keywords": _tags_for(cert.keywords, cert.skills, [], skills),
                **({"relevance": list(cert.relevance)} if cert.relevance else {}),
            }
            for cert in index.certifications
        ]
    }

    summary_data: dict | None = None
    if index.summaries.default.strip():
        summary_data = {
            "summary": index.summaries.default,
            "highlights": list(index.summaries.highlights),
        }

    return skills_data, projects_data, exp_data, certs_data, summary_data
