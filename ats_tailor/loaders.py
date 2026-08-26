"""YAML loading, profile loading, date parsing, recency, and embedding cache."""

import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

CACHE_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_end_date(period):
    """Extract the most recent date from a period string."""
    import re
    if "present" in period.lower():
        return datetime.now()
    matches = re.findall(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b", period)
    if matches:
        month_str, year_str = matches[-1]
        month = MONTH_MAP.get(month_str.lower()[:3], 6)
        return datetime(int(year_str), month, 1)
    years = re.findall(r"(\d{4})", period)
    if years:
        return datetime(int(years[-1]), 6, 1)
    return datetime(2020, 1, 1)


def parse_start_date(period):
    """Extract the start date from a period string."""
    import re
    matches = re.findall(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b", period)
    if matches:
        month_str, year_str = matches[0]
        month = MONTH_MAP.get(month_str.lower()[:3], 6)
        return datetime(int(year_str), month, 1)
    years = re.findall(r"(\d{4})", period)
    if years:
        return datetime(int(years[0]), 6, 1)
    return datetime(2020, 1, 1)


def recency_multiplier(period):
    """Recency boost: current=1.10, <12mo=1.05, <24mo=1.02, older=1.00."""
    end_date = parse_end_date(period)
    now = datetime.now()
    months_ago = (now.year - end_date.year) * 12 + (now.month - end_date.month)
    if months_ago <= 0:
        return 1.10
    elif months_ago <= 12:
        return 1.05
    elif months_ago <= 24:
        return 1.02
    return 1.00


def load_index(index_dir):
    """Load all YAML index files from the given directory."""
    index_dir = Path(index_dir)
    skills_data = _load_yaml(index_dir / "skills.yaml")
    projects_data = _load_yaml(index_dir / "projects.yaml")
    exp_data = _load_yaml(index_dir / "experience.yaml")
    certs_data = _load_yaml(index_dir / "certifications.yaml")
    summary_path = index_dir / "summary.yaml"
    if summary_path.exists():
        summary_data = _load_yaml(summary_path)
    else:
        summary_data = None
    validate_index(skills_data, projects_data, exp_data, certs_data, summary_data)
    return skills_data, projects_data, exp_data, certs_data, summary_data


def load_profile(profile_path):
    """Load profile.yaml with personal info (name, email, phone, links)."""
    profile = _load_yaml(profile_path)
    validate_profile(profile)
    return profile


def _load_yaml(path):
    """Load a required YAML file and fail with an actionable error."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Required YAML file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML file is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at the top level: {path}")
    return data


def _require_mapping(data, key, path):
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: '{key}' must be a mapping")
    return value


def _require_list(data, key, path):
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{path}: '{key}' must be a list")
    return value


def _require_str_list(data, key, path):
    value = _require_list(data, key, path)
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path}: '{key}[{i}]' must be a non-empty string")
    return value


def _require_str(data, key, path):
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: '{key}' must be a non-empty string")
    return value


def validate_index(skills_data, projects_data, exp_data, certs_data, summary_data=None):
    """Validate index schema and cross-file evidence references."""
    categories = _require_list(skills_data, "categories", "skills.yaml")
    projects = _require_list(projects_data, "projects", "projects.yaml")
    roles = _require_list(exp_data, "roles", "experience.yaml")
    education = _require_list(exp_data, "education", "experience.yaml")
    certs = _require_list(certs_data, "certifications", "certifications.yaml")

    ids = {}
    for kind, items in (("project", projects), ("role", roles), ("cert", certs)):
        for i, item in enumerate(items):
            path = f"{kind}[{i}]"
            if not isinstance(item, dict):
                raise ValueError(f"{path}: item must be a mapping")
            item_id = _require_str(item, "id", path)
            if item_id in ids:
                raise ValueError(f"Duplicate id '{item_id}' in {path}; already used by {ids[item_id][1]}")
            ids[item_id] = (kind, path)

    for i, project in enumerate(projects):
        path = f"projects.yaml projects[{i}] ({project['id']})"
        for key in ("name", "period", "summary"):
            _require_str(project, key, path)
        tech_stack = _require_mapping(project, "tech_stack", path)
        for tech_key, tech_values in tech_stack.items():
            if not isinstance(tech_values, list):
                raise ValueError(f"{path}: tech_stack.{tech_key} must be a list")
            for ti, tech in enumerate(tech_values):
                if not isinstance(tech, str) or not tech.strip():
                    raise ValueError(f"{path}: tech_stack.{tech_key}[{ti}] must be a non-empty string")
        _require_str_list(project, "responsibilities", path)
        _require_str_list(project, "ats_tags", path)
        if "impact" in project:
            _require_str_list(project, "impact", path)

    for i, role in enumerate(roles):
        path = f"experience.yaml roles[{i}] ({role['id']})"
        for key in ("company", "title", "period", "type"):
            _require_str(role, key, path)
        if role["type"] not in {"internship", "research", "fulltime"}:
            raise ValueError(f"{path}: 'type' must be internship, research, or fulltime")
        _require_str_list(role, "skills_used", path)
        _require_str_list(role, "bullets", path)
        _require_str_list(role, "ats_tags", path)

    for i, edu in enumerate(education):
        path = f"experience.yaml education[{i}]"
        if not isinstance(edu, dict):
            raise ValueError(f"{path}: item must be a mapping")
        for key in ("institution", "degree", "field", "period"):
            _require_str(edu, key, path)

    for i, cert in enumerate(certs):
        path = f"certifications.yaml certifications[{i}] ({cert['id']})"
        for key in ("name", "issuer"):
            _require_str(cert, key, path)
        _require_str_list(cert, "ats_keywords", path)

    for ci, cat in enumerate(categories):
        path = f"skills.yaml categories[{ci}]"
        if not isinstance(cat, dict):
            raise ValueError(f"{path}: item must be a mapping")
        _require_str(cat, "name", path)
        skills = _require_list(cat, "skills", path)
        for si, skill in enumerate(skills):
            spath = f"{path} skills[{si}]"
            if not isinstance(skill, dict):
                raise ValueError(f"{spath}: item must be a mapping")
            _require_str(skill, "name", spath)
            proficiency = _require_str(skill, "proficiency", spath)
            if proficiency not in {"advanced", "intermediate", "familiar"}:
                raise ValueError(f"{spath}: proficiency must be advanced, intermediate, or familiar")
            _require_str_list(skill, "ats_keywords", spath)
            evidence = skill.get("evidence", [])
            if not isinstance(evidence, list):
                raise ValueError(f"{spath}: 'evidence' must be a list")
            for ei, ev in enumerate(evidence):
                epath = f"{spath} evidence[{ei}]"
                if not isinstance(ev, dict):
                    raise ValueError(f"{epath}: item must be a mapping")
                ref = _require_str(ev, "ref", epath)
                if ref not in ids:
                    raise ValueError(f"{epath}: unknown evidence ref '{ref}'")
                if "type" in ev:
                    if not isinstance(ev["type"], str):
                        raise ValueError(f"{epath}: 'type' must be a string")
                    if ev["type"] not in {"role", "project", "cert"}:
                        raise ValueError(f"{epath}: 'type' must be role, project, or cert")
                    if ev["type"] != ids[ref][0]:
                        raise ValueError(f"{epath}: type '{ev['type']}' does not match ref '{ref}' ({ids[ref][0]})")

    if summary_data is not None:
        _require_str(summary_data, "summary", "summary.yaml")
        if "highlights" in summary_data and not isinstance(summary_data["highlights"], list):
            raise ValueError("summary.yaml: 'highlights' must be a list")


def validate_profile(profile):
    """Validate profile.yaml before rendering contact details."""
    for key in ("name", "email"):
        _require_str(profile, key, "profile.yaml")


def _yaml_hash(data):
    """SHA-256 hash of YAML-serializable data for cache keying."""
    raw = yaml.dump(data, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _text_hash(text):
    """SHA-256 hash of a string."""
    return hashlib.sha256(text.encode()).hexdigest()


def build_cache_key(kind, model_name, content_hash, schema_version="v3"):
    """Build a model-aware, path-safe cache key.

    The key deliberately avoids raw YAML IDs or model names because both can
    contain path separators. Changing the embedding model or vector schema gets
    a separate cache namespace.
    """
    kind = re.sub(r"[^A-Za-z0-9_.-]+", "-", kind).strip(".-_") or "item"
    schema_version = re.sub(r"[^A-Za-z0-9_.-]+", "-", schema_version).strip(".-_") or "v"
    while ".." in kind:
        kind = kind.replace("..", "-")
    while ".." in schema_version:
        schema_version = schema_version.replace("..", "-")
    model_hash = _text_hash(model_name or "default")[:12]
    content_part = re.sub(r"[^A-Fa-f0-9]+", "", content_hash)[:16]
    key = f"{kind}_{schema_version}_{model_hash}_{content_part}"
    _validate_cache_key(key)
    return key


def _validate_cache_key(key):
    if not CACHE_KEY_RE.fullmatch(key):
        raise ValueError(f"Unsafe cache key: {key!r}")


def load_cached_embeddings(cache_dir, key, expected_dim=None, expected_ndim=None,
                           expected_rows=None, expected_min_rows=None):
    """Load cached embeddings if they exist on disk."""
    _validate_cache_key(key)
    path = cache_dir / f"{key}.npy"
    if path.exists():
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception:
            return None
        if expected_ndim is not None and arr.ndim != expected_ndim:
            return None
        if expected_dim is not None and (arr.ndim == 0 or arr.shape[-1] != expected_dim):
            return None
        if expected_rows is not None and (arr.ndim == 0 or arr.shape[0] != expected_rows):
            return None
        if expected_min_rows is not None and (arr.ndim == 0 or arr.shape[0] < expected_min_rows):
            return None
        return arr
    return None


def save_cached_embeddings(cache_dir, key, embeddings):
    """Save embeddings to disk cache."""
    _validate_cache_key(key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{key}.npy", embeddings, allow_pickle=False)
