# Phase 3b — Runtime store, overrides, run records

Reference: `docs/v2-design.md` §4 (runtime store), §7 (tracker vocabulary,
"write facts back"), §6 (labels derived from overrides). Builds on the 3a
v2 pipeline (`ats_tailor/pipeline.py`, `adapt.py`, `ats_match`).

Principle: the index (YAML) and labels stay the git-tracked source of
truth; everything in the store is derived or per-application state and can
be rebuilt or lost without losing content. Overrides are human-editable
files next to the run output *and* recorded in the store.

## 1. Store — `ats_tailor/store.py`

- `Store(path)` over `sqlite3` (stdlib), WAL mode, `PRAGMA user_version`
  migrations (`MIGRATIONS: list[str]` applied in order; v1 creates all
  tables). Path resolution: `ATS_DATA_DIR` env → `<index_dir>/../data/` →
  `./data/`; file `ats.db`; directory created on demand.
- Tables:
  - `entities(id TEXT PRIMARY KEY, kind TEXT, content_hash TEXT, json TEXT, updated_at TEXT)`
    — every role/project/cert/skill of the index, `json` = pydantic dump.
  - `embeddings(content_hash TEXT, model TEXT, kind TEXT, dim INTEGER, vec BLOB, created_at TEXT, PRIMARY KEY(content_hash, model))`
    — float32 row-normalised vectors; `kind` ∈ `claim|req|skill`.
  - `jd_profiles(jd_hash TEXT PRIMARY KEY, source_path TEXT, text TEXT, provider TEXT, prompt_version TEXT, extracted_json TEXT, edited_json TEXT, created_at TEXT)`
    — 3b stores the JD text and the deterministic parser's requirements
    (`provider = "fallback"`); 3c adds LLM/agent output and `edited_json`.
  - `runs(id TEXT PRIMARY KEY, created_at TEXT, company TEXT, role TEXT, jd_hash TEXT, engine TEXT, config_hash TEXT, out_dir TEXT, status TEXT, pdf_sha256 TEXT, json TEXT)`
    — `json` holds the `RunResult` dict plus `coverage`, `trace`,
    `check` (parse + coverage reports when `--check` ran), `overrides`
    (the overrides applied). `status` uses the tracker vocabulary:
    `drafted → applied → interview → offer → hired | rejected | no_response | offer_declined | withdrawn`
    (underscores; the last five are final).
  - `run_notes(run_id TEXT, created_at TEXT, note TEXT)` — dated notes
    (status changes are also recorded here as `status: x → y`).
- `Store` API (typed, small): `upsert_entity`, `entities(kind)`,
  `prune_entities(keep_ids)`, `get_embedding(hash, model)`,
  `put_embeddings(items)` (batched), `get_jd_profile`, `put_jd_profile`,
  `record_run`, `get_run`, `list_runs(status=None, company=None)`,
  `set_status(run_id, status, note=None)`, `add_note`, `notes(run_id)`.
- Embedding cache: the v2 pipeline's embedder wrapper reads/writes the
  store instead of `.npy` files (v1 keeps `.emb_cache/` untouched). Cache
  key = `(content_hash of the text, model)`. `sync --embed` pre-computes
  claim and skill embeddings.

## 2. `ats-tailor sync [--index index] [--embed]`
Load the index, upsert all entities with content hashes, prune entities
whose ids are gone, optionally embed. Idempotent; prints counts
(`added / updated / unchanged / pruned / embedded`).

## 3. Overrides — `ats_tailor/overrides.py`
`Overrides` (pydantic, `extra="forbid"`), stored as `<out_dir>/overrides.yaml`
and written into the run record:
```yaml
pin_items: [carehaven]            # role/project ids that must be shown
exclude_items: [wishbone]
pin_bullets: [carehaven.i1]       # bullet ids that must be shown (their item is implied pinned)
exclude_bullets: [razer-swe.b3]
bullet_text:                      # per-run phrasing; source facts must survive
  carehaven.r1: "Fronted 10+ services with a gRPC gateway enforcing auth and rate limits"
skills_pin: [kubernetes]
skills_exclude: [dart]
summary: null                     # replace the summary text for this run
requirements:                     # edits to the extracted requirement list
  remove: [req-3]
  reweight: {req-7: 1.5}
  add:
    - {text: "Terraform / IaC", kind: must, terms: [Terraform, IaC]}
budget_lines: null                # override ATS_MAX_PAGE_LINES − fixed
```
- The v2 pipeline loads `overrides.yaml` from `--output` if present (and
  `--overrides <file>` to point elsewhere). Mapping to `ats_match`:
  pins/excludes → `Constraints`; `bullet_text` replaces the claim text
  before matching *and* rendering; `skills_*` → skill-section
  constraints; `requirements` edits applied after extraction; `summary`
  at render.
- Metric guardrail (light version; the full LLM-facing one is Phase 5):
  when `bullet_text` drops a number/unit present in the source bullet's
  `metrics`, print a warning naming the bullet and the missing metric —
  never block, the user is editing their own resume.
- CLI helper: `ats-tailor override <out_dir> [--pin id]... [--exclude id]...
  [--pin-bullet id]... [--exclude-bullet id]... [--text bullet_id=TEXT]...
  [--summary TEXT] [--show]` edits/creates the file via ruamel round-trip;
  `--show` prints it. After editing, the user re-runs `./tailor … --engine v2`
  (same output dir) and the run record notes `overrides applied`.

## 4. Run records and the tracker
- Every `tailor --engine v2` (not `eval`) records a run: `id = "<company>-<role>-<YYYYMMDD-HHMMSS>"`, `status = drafted`, `pdf_sha256` when compiled. `selection.json` stays for `check`.
- `ats-tailor runs list [--status S] [--company C]` — table: id, created,
  company/role, status, must/nice coverage if checked, out_dir.
- `ats-tailor runs show <id>` — the record: selection, coverage summary,
  overrides, notes.
- `ats-tailor runs status <id> <status> [--note TEXT]` — validates the
  vocabulary, records the transition as a note; moving off `drafted`
  stamps `applied_at` in `json`.
- `ats-tailor runs note <id> TEXT`.
- `ats-tailor runs quiet [--days 10]` — open runs (not final, not
  drafted) with no note in the last N days: id, company/role, days quiet.
  (Drafting follow-ups is Phase 5.)

## 5. Labels from overrides
`ats-tailor labels from-run <out_dir|run_id> [--labels-dir labels]` merges
the run's `pin_items`/`exclude_items` into `labels/<slug>.yaml`
(`pin_projects`/`pin_roles` by item kind, `exclude`), preserving existing
entries and comments (ruamel), so eval labels accumulate from real
decisions. Prints what changed.

## 6. Tests
- Store: migrations idempotent; entity upsert/prune; embedding round trip
  preserves dtype/shape/normalisation; run record + status transitions
  (invalid status rejected; final statuses accepted); notes ordering.
- Overrides: schema (`extra="forbid"`); pins/excludes reach `Constraints`
  and are honoured by `ats_match.select` on a synthetic problem;
  `bullet_text` replaces text in claims and render context; metric
  guardrail warning fires on a dropped number; `requirements.add/remove/
  reweight` applied; `override` CLI round-trips a file with a comment.
- `sync` idempotent (second run reports all unchanged); `labels from-run`
  merge keeps existing pins and comments.
- Smoke (`ATS_SMOKE=1`): v2 run on `examples/` records a run, `runs list`
  shows it, an `overrides.yaml` pinning a project changes the selection
  on re-run and the trace shows the pin.

## 7. Docs
README: store/sync, overrides file + CLI, runs/tracker, labels from-run.
Parent repo: `data/` is gitignored (the coordinator adds the rule).

## Out of scope
LLM/agent extraction and gates, providers, alias-aligned rendering (3c);
follow-up drafting (Phase 5); TUI/MCP (Phase 4).
