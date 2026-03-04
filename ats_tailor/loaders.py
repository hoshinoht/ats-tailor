"""YAML loading, profile loading, date parsing, recency, and embedding cache."""

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_end_date(period):
    """Extract the most recent date from a period string."""
    import re
    if "present" in period.lower():
        return datetime.now()
    matches = re.findall(r"([A-Za-z]{3})\s+(\d{4})", period)
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
    matches = re.findall(r"([A-Za-z]{3})\s+(\d{4})", period)
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
    with open(index_dir / "skills.yaml") as f:
        skills_data = yaml.safe_load(f)
    with open(index_dir / "projects.yaml") as f:
        projects_data = yaml.safe_load(f)
    with open(index_dir / "experience.yaml") as f:
        exp_data = yaml.safe_load(f)
    with open(index_dir / "certifications.yaml") as f:
        certs_data = yaml.safe_load(f)
    summary_path = index_dir / "summary.yaml"
    if summary_path.exists():
        with open(summary_path) as f:
            summary_data = yaml.safe_load(f)
    else:
        summary_data = None
    return skills_data, projects_data, exp_data, certs_data, summary_data


def load_profile(profile_path):
    """Load profile.yaml with personal info (name, email, phone, links)."""
    with open(profile_path) as f:
        return yaml.safe_load(f)


def _yaml_hash(data):
    """SHA-256 hash of YAML-serializable data for cache keying."""
    raw = yaml.dump(data, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _text_hash(text):
    """SHA-256 hash of a string."""
    return hashlib.sha256(text.encode()).hexdigest()


def load_cached_embeddings(cache_dir, key):
    """Load cached embeddings if they exist on disk."""
    path = cache_dir / f"{key}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_cached_embeddings(cache_dir, key, embeddings):
    """Save embeddings to disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{key}.npy", embeddings)
