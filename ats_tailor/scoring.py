"""Keyword matching, embedding construction, cosine similarity, and JD weighting."""

import math
import re

import numpy as np

AMBIGUOUS_KEYWORDS = {"go", "c", "r"}


def keyword_in_text(keyword, text_lower, text_original=""):
    """Check if keyword appears in text with word boundaries.

    For ambiguous short keywords (Go, C, R, etc.), uses case-sensitive
    matching against the original text — JDs write languages canonically
    ("Go", "C") while English homonyms are lowercase.
    """
    kw = keyword.lower()
    # Keywords with special chars (C++, C#, .NET) — substring match
    if not kw.replace(" ", "").replace("-", "").isalnum():
        return kw in text_lower
    # Ambiguous short keywords — case-sensitive match on original text
    if kw in AMBIGUOUS_KEYWORDS or (len(kw) <= 2 and kw.isalpha()):
        if not text_original:
            return bool(re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
        return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text_original))
    # Alphanumeric keywords — word boundary match
    return bool(re.search(r"\b" + re.escape(kw) + r"\b", text_lower))


def compute_keyword_bonus(keywords, jd_lower, jd_text=""):
    """Bonus score for exact keyword matches found in JD text."""
    matches = sum(1 for kw in keywords if keyword_in_text(
        kw, jd_lower, jd_text))
    return 0.06 * math.log2(1 + matches) if matches else 0.0


def compute_tag_overlap_bonus(tags, jd_lower, jd_text=""):
    """Bonus for tag alignment with JD, blending ratio and absolute hits.

    Pure ratio (hits/total) penalises items with rich tag sets — e.g. an
    item with 2/5 tags matching scores 0.10 while 2/14 scores only 0.036,
    even though both have the same absolute relevance. Blending with an
    absolute coverage term (hits/6, capped at 1.0) prevents small tag sets
    from getting disproportionate boosts.
    """
    if not tags:
        return 0.0
    hits = sum(1 for t in tags if keyword_in_text(t, jd_lower, jd_text))
    ratio = hits / len(tags)
    coverage = min(hits / 6, 1.0)  # 6+ hits = full coverage credit
    return (0.4 * ratio + 0.6 * coverage) * 0.25


def chunk_text(text, max_len=512):
    """Split text into sentence-ish chunks for embedding."""
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(". ", ".\n").split("\n")
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences if sentences else [text[:max_len]]


def build_skill_text(skill):
    """Build a text representation of a skill for embedding."""
    keywords = " ".join(skill.get("ats_keywords", []))
    return f"{skill['name']} {keywords}"


def build_project_text(proj):
    """Build a text representation of a project for embedding."""
    tags = " ".join(proj.get("ats_tags", []))
    tech = []
    for key in ("languages", "frameworks", "infrastructure", "patterns", "tools"):
        tech.extend(proj.get("tech_stack", {}).get(key, []))
    tech_str = " ".join(tech)
    resps = " ".join(proj.get("responsibilities", []))
    return f"{proj['name']} {proj.get('summary', '')} {tags} {tech_str} {resps}"


def build_role_text(role):
    """Build a text representation of a role for embedding."""
    skills = " ".join(role.get("skills_used", []))
    tags = " ".join(role.get("ats_tags", []))
    bullets = " ".join(role.get("bullets", []))
    return f"{role['title']} {role['company']} {skills} {tags} {bullets}"


def build_cert_text(cert):
    """Build a text representation of a certification for embedding."""
    keywords = " ".join(cert.get("ats_keywords", []))
    return f"{cert['name']} {cert['issuer']} {keywords}"


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def score_against_jd(jd_embeddings, item_embedding):
    """Max-pooled cosine similarity: best match across JD sentences."""
    return max(cosine_sim(jd_emb, item_embedding) for jd_emb in jd_embeddings)


def score_against_jd_multi(jd_embeddings, item_embeddings, jd_weights=None):
    """Title-anchored 15/50/35 scoring.

    item_embeddings[0] = title, [1] = metadata, [2..] = bullets.
    title_score contributes 15%, metadata 50%, mean bullet scores 35%.
    """
    if jd_weights is None:
        jd_weights = np.ones(len(jd_embeddings))

    def _max_sim(emb):
        sims = np.array([cosine_sim(jd_emb, emb) for jd_emb in jd_embeddings])
        return float(np.max(sims * jd_weights))

    title_score = _max_sim(item_embeddings[0])
    meta_score = _max_sim(item_embeddings[1])
    if len(item_embeddings) <= 2:
        return 0.15 * title_score + 0.85 * meta_score
    bullet_scores = [_max_sim(emb) for emb in item_embeddings[2:]]
    return 0.15 * title_score + 0.50 * meta_score + 0.35 * float(np.mean(bullet_scores))


def build_item_vectors(item, kind, model):
    """Encode item as multiple vectors: title + metadata text + each bullet separately."""
    if kind == "project":
        meta = build_project_text(item)
        bullets = item.get("responsibilities", []) + item.get("impact", [])
        title = item.get("name", "")
    else:  # role
        meta = build_role_text(item)
        bullets = item.get("bullets", [])
        title = f"{item.get('title', '')} {item.get('company', '')}"
    texts = [title, meta] + [b for b in bullets if b.strip()]
    return model.encode(texts)


def compute_jd_weights(chunks):
    """Return per-chunk importance weights. Requirement-like sections get 1.5x."""
    req_pattern = re.compile(
        r"requirement|qualification|responsibilities|what you.ll|must have|skills",
        re.IGNORECASE,
    )
    weights = []
    in_req_section = False
    for chunk in chunks:
        is_header = (
            chunk.endswith(":") or chunk.startswith("#")
            or (len(chunk.split()) <= 6 and chunk[0:1].isupper())
        )
        if is_header and req_pattern.search(chunk):
            in_req_section = True
        elif is_header and not req_pattern.search(chunk):
            in_req_section = False
        weights.append(1.5 if in_req_section else 1.0)
    return np.array(weights)
