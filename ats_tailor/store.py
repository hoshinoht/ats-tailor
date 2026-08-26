"""The runtime store: everything derived, per-application, or rebuildable.

The index (YAML) and the labels stay the git-tracked source of truth. This file
is the other half of that split: one SQLite database under ``data/`` holding the
entity snapshots ``sync`` writes, the embedding cache that replaces v2's
``.emb_cache/*.npy`` files, the postings a run was tailored to, and the run
records the tracker reads. Deleting it loses no content — ``ats-tailor sync``
rebuilds everything except the run history, which is per-application state that
was never in the index to begin with.

Only the stdlib ``sqlite3`` is used, and schema changes go through
:data:`MIGRATIONS` keyed on ``PRAGMA user_version``, so an older database is
upgraded in place rather than thrown away.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import config

#: The database file inside the resolved data directory.
STORE_FILENAME = "ats.db"

#: Tracker vocabulary (``docs/v2-design.md`` §7). Underscores, never spaces.
STATUSES: tuple[str, ...] = (
    "drafted",
    "applied",
    "interview",
    "offer",
    "hired",
    "rejected",
    "no_response",
    "offer_declined",
    "withdrawn",
)
#: Statuses an application does not come back from.
FINAL_STATUSES: frozenset[str] = frozenset(
    {"hired", "rejected", "no_response", "offer_declined", "withdrawn"}
)
#: The status every run starts in.
INITIAL_STATUS = "drafted"

MIGRATIONS: list[str] = [
    # ── v1 ──
    """
    CREATE TABLE IF NOT EXISTS entities (
        id           TEXT PRIMARY KEY,
        kind         TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        json         TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS entities_kind ON entities(kind);

    CREATE TABLE IF NOT EXISTS embeddings (
        content_hash TEXT NOT NULL,
        model        TEXT NOT NULL,
        kind         TEXT NOT NULL,
        dim          INTEGER NOT NULL,
        vec          BLOB NOT NULL,
        created_at   TEXT NOT NULL,
        PRIMARY KEY (content_hash, model)
    );

    CREATE TABLE IF NOT EXISTS jd_profiles (
        jd_hash        TEXT PRIMARY KEY,
        source_path    TEXT,
        text           TEXT NOT NULL,
        provider       TEXT NOT NULL,
        prompt_version TEXT,
        extracted_json TEXT,
        edited_json    TEXT,
        created_at     TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS runs (
        id          TEXT PRIMARY KEY,
        created_at  TEXT NOT NULL,
        company     TEXT,
        role        TEXT,
        jd_hash     TEXT,
        engine      TEXT,
        config_hash TEXT,
        out_dir     TEXT,
        status      TEXT NOT NULL,
        pdf_sha256  TEXT,
        json        TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS runs_status ON runs(status);
    CREATE INDEX IF NOT EXISTS runs_company ON runs(company);

    CREATE TABLE IF NOT EXISTS run_notes (
        run_id     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        note       TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS run_notes_run ON run_notes(run_id);
    """,
]


def now() -> str:
    """The timestamp format every column in this file uses."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Where the database lives ─────────────────────────────────────────────────
def data_dir(index_dir: str | Path | None = None) -> Path:
    """``ATS_DATA_DIR`` → ``<index_dir>/../data/`` → ``./data/``.

    The index directory is the anchor rather than the working directory so a
    run started from anywhere still finds the store belonging to the index it
    is reading.
    """
    configured = str(getattr(config, "DATA_DIR", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    if index_dir:
        return Path(index_dir).resolve().parent / "data"
    return Path("data")


def store_path(index_dir: str | Path | None = None) -> Path:
    return data_dir(index_dir) / STORE_FILENAME


# ── Rows ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Entity:
    id: str
    kind: str
    content_hash: str
    data: dict[str, Any]
    updated_at: str


@dataclass(frozen=True)
class JDProfile:
    jd_hash: str
    source_path: str | None
    text: str
    provider: str
    prompt_version: str | None
    extracted: Any
    edited: Any
    created_at: str


@dataclass(frozen=True)
class Run:
    id: str
    created_at: str
    company: str
    role: str
    jd_hash: str | None
    engine: str
    config_hash: str | None
    out_dir: str | None
    status: str
    pdf_sha256: str | None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def final(self) -> bool:
        return self.status in FINAL_STATUSES

    @property
    def coverage(self) -> dict[str, Any]:
        check = self.data.get("check") or {}
        return check.get("coverage") or {}


@dataclass(frozen=True)
class Note:
    run_id: str
    created_at: str
    note: str


class StatusError(ValueError):
    """A status outside the tracker vocabulary."""


def validate_status(status: str) -> str:
    value = str(status or "").strip().lower().replace(" ", "_").replace("-", "_")
    if value not in STATUSES:
        raise StatusError(
            f"unknown status {status!r}; expected one of: {', '.join(STATUSES)}"
        )
    return value


# ── The store ────────────────────────────────────────────────────────────────
class Store:
    """A typed, deliberately small face on one SQLite file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    # ── lifecycle ──
    @classmethod
    def open(cls, index_dir: str | Path | None = None) -> "Store":
        """The store belonging to ``index_dir`` (see :func:`data_dir`)."""
        return cls(store_path(index_dir))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def migrate(self) -> int:
        """Apply every migration past ``user_version``; returns the new version."""
        with closing(self.conn.execute("PRAGMA user_version")) as cursor:
            version = int(cursor.fetchone()[0])
        for step, script in enumerate(MIGRATIONS[version:], start=version + 1):
            self.conn.executescript(script)
            self.conn.execute(f"PRAGMA user_version={step}")
            self.conn.commit()
        return len(MIGRATIONS)

    @property
    def version(self) -> int:
        with closing(self.conn.execute("PRAGMA user_version")) as cursor:
            return int(cursor.fetchone()[0])

    # ── entities ──
    def upsert_entity(
        self, entity_id: str, kind: str, content_hash: str, data: Mapping[str, Any]
    ) -> str:
        """``"added"`` / ``"updated"`` / ``"unchanged"`` — what ``sync`` counts."""
        row = self.conn.execute(
            "SELECT content_hash FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is not None and row["content_hash"] == content_hash:
            return "unchanged"
        self.conn.execute(
            "INSERT INTO entities (id, kind, content_hash, json, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, "
            "content_hash=excluded.content_hash, json=excluded.json, "
            "updated_at=excluded.updated_at",
            (entity_id, kind, content_hash, _dumps(data), now()),
        )
        self.conn.commit()
        return "updated" if row is not None else "added"

    def entities(self, kind: str | None = None) -> list[Entity]:
        if kind is None:
            rows = self.conn.execute("SELECT * FROM entities ORDER BY id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE kind = ? ORDER BY id", (kind,)
            ).fetchall()
        return [
            Entity(r["id"], r["kind"], r["content_hash"], json.loads(r["json"]),
                   r["updated_at"])
            for r in rows
        ]

    def get_entity(self, entity_id: str) -> Entity | None:
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return None
        return Entity(row["id"], row["kind"], row["content_hash"],
                      json.loads(row["json"]), row["updated_at"])

    def prune_entities(self, keep_ids: Iterable[str]) -> list[str]:
        """Delete entities whose ids are no longer in the index; returns them."""
        keep = {str(i) for i in keep_ids}
        gone = [
            row["id"]
            for row in self.conn.execute("SELECT id FROM entities").fetchall()
            if row["id"] not in keep
        ]
        if gone:
            self.conn.executemany(
                "DELETE FROM entities WHERE id = ?", [(i,) for i in gone]
            )
            self.conn.commit()
        return gone

    # ── embeddings ──
    def get_embedding(self, content_hash: str, model: str) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT dim, vec FROM embeddings WHERE content_hash = ? AND model = ?",
            (content_hash, model),
        ).fetchone()
        return None if row is None else _vector(row)

    def get_embeddings(
        self, content_hashes: Sequence[str], model: str
    ) -> dict[str, np.ndarray]:
        """Batched :meth:`get_embedding`, for the pipeline's per-run lookup."""
        found: dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(content_hashes))
        for start in range(0, len(unique), 500):
            chunk = unique[start:start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT content_hash, dim, vec FROM embeddings "
                f"WHERE model = ? AND content_hash IN ({placeholders})",
                (model, *chunk),
            ).fetchall()
            for row in rows:
                found[row["content_hash"]] = _vector(row)
        return found

    def put_embeddings(
        self, items: Iterable[tuple[str, str, str, "np.ndarray | Sequence[float]"]]
    ) -> int:
        """Store ``(content_hash, model, kind, vector)`` rows, row-normalised.

        Vectors go in as float32 bytes: the models produce float32, and a
        resume's worth of vectors is small enough that the compactness matters
        less than not having to guess a dtype on the way out.
        """
        payload = []
        for content_hash, model, kind, vector in items:
            vec = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = (vec / norm).astype(np.float32)
            payload.append(
                (content_hash, model, kind, int(vec.size), vec.tobytes(), now())
            )
        if not payload:
            return 0
        self.conn.executemany(
            "INSERT INTO embeddings (content_hash, model, kind, dim, vec, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(content_hash, model) DO UPDATE SET kind=excluded.kind, "
            "dim=excluded.dim, vec=excluded.vec, created_at=excluded.created_at",
            payload,
        )
        self.conn.commit()
        return len(payload)

    def count_embeddings(self, model: str | None = None) -> int:
        if model is None:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM embeddings WHERE model = ?", (model,)
            ).fetchone()
        return int(row["n"])

    # ── job descriptions ──
    def get_jd_profile(self, jd_hash: str) -> JDProfile | None:
        row = self.conn.execute(
            "SELECT * FROM jd_profiles WHERE jd_hash = ?", (jd_hash,)
        ).fetchone()
        if row is None:
            return None
        return JDProfile(
            jd_hash=row["jd_hash"], source_path=row["source_path"], text=row["text"],
            provider=row["provider"], prompt_version=row["prompt_version"],
            extracted=_loads(row["extracted_json"]), edited=_loads(row["edited_json"]),
            created_at=row["created_at"],
        )

    def put_jd_profile(
        self,
        jd_hash: str,
        text: str,
        provider: str = "fallback",
        source_path: str | None = None,
        prompt_version: str | None = None,
        extracted: Any = None,
        edited: Any = None,
    ) -> None:
        """Record the posting and what the extractor made of it.

        An existing ``edited_json`` is kept when the caller passes none: the
        human edit outlives any number of re-extractions.
        """
        existing = self.get_jd_profile(jd_hash)
        if edited is None and existing is not None:
            edited = existing.edited
        self.conn.execute(
            "INSERT INTO jd_profiles (jd_hash, source_path, text, provider, "
            "prompt_version, extracted_json, edited_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(jd_hash) DO UPDATE SET source_path=excluded.source_path, "
            "text=excluded.text, provider=excluded.provider, "
            "prompt_version=excluded.prompt_version, "
            "extracted_json=excluded.extracted_json, edited_json=excluded.edited_json",
            (jd_hash, source_path, text, provider, prompt_version,
             _dumps(extracted) if extracted is not None else None,
             _dumps(edited) if edited is not None else None,
             existing.created_at if existing else now()),
        )
        self.conn.commit()

    # ── runs ──
    def record_run(
        self,
        run_id: str,
        company: str,
        role: str,
        data: Mapping[str, Any],
        jd_hash: str | None = None,
        engine: str = "v2",
        config_hash: str | None = None,
        out_dir: str | Path | None = None,
        status: str = INITIAL_STATUS,
        pdf_sha256: str | None = None,
        created_at: str | None = None,
    ) -> Run:
        status = validate_status(status)
        created = created_at or now()
        self.conn.execute(
            "INSERT INTO runs (id, created_at, company, role, jd_hash, engine, "
            "config_hash, out_dir, status, pdf_sha256, json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET company=excluded.company, "
            "role=excluded.role, jd_hash=excluded.jd_hash, engine=excluded.engine, "
            "config_hash=excluded.config_hash, out_dir=excluded.out_dir, "
            "pdf_sha256=excluded.pdf_sha256, json=excluded.json",
            (run_id, created, company, role, jd_hash, engine, config_hash,
             str(out_dir) if out_dir is not None else None, status, pdf_sha256,
             _dumps(dict(data))),
        )
        self.conn.commit()
        run = self.get_run(run_id)
        assert run is not None
        return run

    def get_run(self, run_id: str) -> Run | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return None if row is None else _run(row)

    def list_runs(
        self, status: str | None = None, company: str | None = None
    ) -> list[Run]:
        """Newest first, which is the order the tracker table prints."""
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(validate_status(status))
        if company:
            clauses.append("LOWER(company) = ?")
            params.append(company.strip().lower())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM runs{where} ORDER BY created_at DESC, id DESC", params
        ).fetchall()
        return [_run(row) for row in rows]

    def set_status(self, run_id: str, status: str, note: str | None = None) -> Run:
        """Move a run's status, recording the transition as a dated note.

        Leaving ``drafted`` stamps ``applied_at``: that is the date every
        follow-up is counted from, and it is not recoverable from the run's
        creation time (a resume is often drafted days before it is sent).
        """
        status = validate_status(status)
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"no such run: {run_id}")
        data = dict(run.data)
        if run.status == INITIAL_STATUS and status != INITIAL_STATUS:
            data.setdefault("applied_at", now())
        self.conn.execute(
            "UPDATE runs SET status = ?, json = ? WHERE id = ?",
            (status, _dumps(data), run_id),
        )
        self.conn.commit()
        message = f"status: {run.status} → {status}"
        if note:
            message = f"{message} — {note}"
        self.add_note(run_id, message)
        updated = self.get_run(run_id)
        assert updated is not None
        return updated

    def add_note(self, run_id: str, note: str, created_at: str | None = None) -> Note:
        text = str(note).strip()
        if not text:
            raise ValueError("a note cannot be empty")
        if self.get_run(run_id) is None:
            raise KeyError(f"no such run: {run_id}")
        created = created_at or now()
        self.conn.execute(
            "INSERT INTO run_notes (run_id, created_at, note) VALUES (?, ?, ?)",
            (run_id, created, text),
        )
        self.conn.commit()
        return Note(run_id, created, text)

    def notes(self, run_id: str) -> list[Note]:
        """Oldest first — a note list is a history, and histories read forwards."""
        rows = self.conn.execute(
            "SELECT * FROM run_notes WHERE run_id = ? ORDER BY created_at, rowid",
            (run_id,),
        ).fetchall()
        return [Note(r["run_id"], r["created_at"], r["note"]) for r in rows]

    def last_note_at(self, run_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT created_at FROM run_notes WHERE run_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return None if row is None else row["created_at"]


# ── helpers ──────────────────────────────────────────────────────────────────
def _vector(row: sqlite3.Row) -> np.ndarray:
    """Bytes back to a unit-norm float64 row, the dtype the pipeline works in."""
    vec = np.frombuffer(row["vec"], dtype=np.float32).astype(float)
    if vec.size != int(row["dim"]):
        raise ValueError(
            f"stored vector has {vec.size} values, expected {int(row['dim'])}"
        )
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"], created_at=row["created_at"], company=row["company"],
        role=row["role"], jd_hash=row["jd_hash"], engine=row["engine"],
        config_hash=row["config_hash"], out_dir=row["out_dir"], status=row["status"],
        pdf_sha256=row["pdf_sha256"], data=json.loads(row["json"]),
    )


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=False, default=str)


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)
