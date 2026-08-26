"""Bring the runtime store in line with the index.

``sync`` is the one direction data flows between the two: YAML in, SQLite out.
It is idempotent by construction — every entity carries the hash of its own
content, so a second run over an unchanged index reports everything unchanged
and writes nothing — and it is the only thing that has to be re-run after
editing the index by hand.

``--embed`` additionally fills the embedding cache for the configured model, so
the first tailoring run after an edit pays for the edited text only.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from ..engine.adapt import claims_from_index
from ..index.models import Index
from .store import Store

#: ``entities.id`` is namespaced by kind: a skill and a project are allowed to
#: share an id in the index (they are validated in separate namespaces), and the
#: store has one primary key for both.
KINDS = ("role", "project", "cert", "skill")


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    embedded: int = 0
    cached: int = 0
    seconds: float = 0.0

    def render(self) -> str:
        lines = [
            f"added {len(self.added)} / updated {len(self.updated)} / "
            f"unchanged {len(self.unchanged)} / pruned {len(self.pruned)}"
            + (f" / embedded {self.embedded} (cached {self.cached})"
               if self.embedded or self.cached else "")
        ]
        for label, ids in (("added", self.added), ("updated", self.updated),
                           ("pruned", self.pruned)):
            for entity_id in ids:
                lines.append(f"  {label:<9} {entity_id}")
        lines.append(f"took {self.seconds:.1f}s")
        return "\n".join(lines)


def entity_key(kind: str, entity_id: str) -> str:
    return f"{kind}:{entity_id}"


def content_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def index_entities(index: Index) -> list[tuple[str, str, dict[str, Any]]]:
    """``(kind, id, payload)`` for every role, project, certification and skill."""
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for role in index.roles:
        rows.append(("role", role.id, role.model_dump(mode="json")))
    for project in index.projects:
        rows.append(("project", project.id, project.model_dump(mode="json")))
    for cert in index.certifications:
        rows.append(("cert", cert.id, cert.model_dump(mode="json")))
    for category in index.skills:
        for skill in category.skills:
            rows.append((
                "skill", skill.id,
                {**skill.model_dump(mode="json"), "category": category.category},
            ))
    return rows


def skill_texts(index: Index) -> list[str]:
    """The text ``adapt.skill_claims`` embeds, for every skill in the index.

    ``skill_claims`` holds ``familiar`` skills back unless a posting names one;
    ``sync`` has no posting, so it embeds them all — the point is that no run
    ever has to wait for a model.
    """
    return [" ".join(skill.surface_forms()) for skill in index.all_skills()]


def cert_texts(index: Index) -> list[str]:
    from ..index.io import to_legacy
    from ..engine.legacy.scoring import build_cert_text

    _skills, _projects, _exp, certs, _summary = to_legacy(index)
    return [build_cert_text(cert) for cert in certs["certifications"]]


def sync(
    index_dir: str | Path,
    embed: bool = False,
    model: str | None = None,
    store: Store | None = None,
    index: Index | None = None,
) -> SyncReport:
    """Upsert every entity, prune what the index no longer has, optionally embed."""
    from ..index.io import load_index_v2

    started = time.monotonic()
    index_dir = Path(index_dir)
    index = index if index is not None else load_index_v2(index_dir)
    owns_store = store is None
    store = store or Store.open(index_dir)
    report = SyncReport()
    try:
        keep: list[str] = []
        for kind, entity_id, payload in index_entities(index):
            key = entity_key(kind, entity_id)
            keep.append(key)
            state = store.upsert_entity(key, kind, content_hash(payload), payload)
            getattr(report, state if state != "unchanged" else "unchanged").append(key)
        report.pruned = store.prune_entities(keep)

        if embed:
            report.embedded, report.cached = _embed(index, store, model)
    finally:
        if owns_store:
            store.close()
    report.seconds = time.monotonic() - started
    return report


def _embed(index: Index, store: Store, model: str | None) -> tuple[int, int]:
    """Fill the cache for claim, skill and certification texts; returns (new, hit)."""
    from ..engine.embedding import SentenceTransformerEmbedder, encode_store, text_hash

    embedder = SentenceTransformerEmbedder(model or config.EMBED_MODEL)
    groups = [
        ("claim", embedder.passage_prefix,
         [claim.text for claim in claims_from_index(index)]),
        ("skill", embedder.passage_prefix, skill_texts(index)),
        ("cert", "", cert_texts(index)),
    ]
    new = hit = 0
    for kind, prefix, texts in groups:
        texts = [t for t in dict.fromkeys(texts) if t.strip()]
        if not texts:
            continue
        hashes = [text_hash(text, prefix) for text in texts]
        present = store.get_embeddings(hashes, embedder.model_name)
        hit += len(present)
        new += len(set(hashes)) - len(present)
        encode_store(embedder, texts, store, kind, prefix)
    return new, hit
