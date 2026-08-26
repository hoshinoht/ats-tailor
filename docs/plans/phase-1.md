# Phase 1 — Schema, per-entity index, migration, packaging

Reference: `docs/v2-design.md` §3 (data model) and §8. Everything in this
phase must leave the existing v1 pipeline (`python -m ats_tailor.tailor …`)
working end-to-end on both the old and the new index layout.

## Deliverables

### 1. Packaging
- `pyproject.toml` (PEP 621): name `ats-tailor`, `requires-python >= 3.11`,
  deps `pyyaml`, `ruamel.yaml`, `pydantic>=2`, `sentence-transformers`,
  `numpy`, `jinja2`, `matplotlib`. Extras: `mlx = [mlx-lm]`,
  `rerank = []` (placeholder), `dev = [pytest]`.
  `[project.scripts] ats-tailor = "ats_tailor.cli:main"`.
- `uv sync --extra dev` creates `.venv` + `uv.lock` (commit the lock file).
  Add `.venv/` and `data/` to `.gitignore` if missing.
- Keep `requirements.txt` but make it a pointer (`-e .`) so the README's
  `pip install -r requirements.txt` still works.

### 2. Models — `ats_tailor/models.py` (pydantic v2)
- `Period{start: str, end: str}`; `start`/`end` match `^\d{4}(-\d{2})?$`,
  `end` may also be `present`. Methods: `format()` → `"Mon YYYY - Mon YYYY"`,
  `"Mon YYYY - Present"`, or `"2017 - 2018"` for year-only — **ASCII hyphen
  (U+002D) with spaces, never an en dash**: en-dash date ranges are a
  confirmed silent ATS import failure (Workday drops the range). v1's
  `parse_start_date`/`parse_end_date` are separator-agnostic so the legacy
  path is unaffected. `Period.parse` must accept `-`, `–`, `—`, `to`;
  `start_date()`, `end_date()` returning `datetime` (year-only → January /
  December; present → now). Class method `parse(str)` accepting every format
  v1's `parse_start_date` / `parse_end_date` accept (full and abbreviated
  month names, `Sept.`, `Present`, bare years).
- `Bullet{id, kind: Literal[responsibility, impact], text, skills: list[str]=[], metrics: list[str]=[], variants: dict[str,str]={}}`
- `Skill{id, name, proficiency: Literal[advanced,intermediate,familiar], aliases: list[str]=[], acronym: str|None, evidence: list[Evidence]=[]}`;
  `Evidence{ref, type: Literal[role,project,cert], detail: str|None}`.
- `SkillCategory{category: str, skills: list[Skill]}`
- `Project{id, name, type: str|None, repo: str|None, period, summary, tech_stack: dict[str, list[str]]={}, skills: list[str]=[], keywords: list[str]=[], bullets: list[Bullet]}`
- `Role{id, company, title, type: Literal[internship,research,fulltime], period, hidden: bool=False, skills, keywords, bullets}`
- `Certification{id, name, issuer, date: str|None, skills, keywords, relevance: list[str]=[]}`
- `Education{institution, degree, field, period}`
- `Summaries{default: str, variants: dict[str,str]={}, highlights: list[str]=[]}`
- `Index{skills: list[SkillCategory], projects, roles, certifications, education, summaries}` with a model-level validator:
  ids unique across roles/projects/certs; skill ids unique; every
  `Evidence.ref`, `Project.skills`, `Role.skills`, `Bullet.skills`,
  `Certification.skills` resolves; bullet ids unique; non-hidden roles have
  ≥1 bullet. Errors are `ValueError` with a path. Separate `warnings()`
  method returning non-fatal issues: bullet text > 180 chars, project with no
  impact bullet, skill with no evidence, item with zero skills and zero keywords.
- `Index.model_json_schema()` is used by `ats-tailor schema`.

### 3. IO — `ats_tailor/index_io.py`
- `detect_layout(index_dir) -> "v1" | "v2"`: v2 iff `index_dir/projects/`
  is a directory.
- `load_index_v2(index_dir) -> Index`. File layout per design §3:
  `education.yaml`, `summaries.yaml`, `certifications.yaml`,
  `skills/<category-slug>.yaml`, `roles/<id>.yaml`, `projects/<id>.yaml`.
  Filename stem must equal the entity `id` (error otherwise).
- `save_index_v2(index: Index, index_dir)` and per-entity `save_project`,
  `save_role`, … using `ruamel.yaml` round-trip: when the file exists, load
  it with ruamel, update values in place (preserve comments and key order),
  write back; when new, write keys in schema declaration order. Each new
  file starts with a `# yaml-language-server: $schema=../../schema/<kind>.schema.json`
  comment (relative path from the file).
- `to_legacy(index: Index) -> (skills_data, projects_data, exp_data, certs_data, summary_data)`
  producing exactly the v1 dict shapes the current pipeline consumes:
  - `period` → `Period.format()` string.
  - project `responsibilities` = texts of `kind=responsibility` bullets,
    `impact` = texts of `kind=impact` bullets (order preserved).
  - `ats_tags` = `keywords` + for every referenced skill id (item-level and
    bullet-level): the skill `name`, its `aliases`, and `acronym`; deduped,
    order preserved. Roles the same. Certs: `ats_keywords` likewise.
  - `skills_used` (roles) = names of `skills`.
  - roles with `hidden: true` are **omitted**.
  - `skills.yaml` shape: categories → skills with `name`, `proficiency`,
    `ats_keywords` (= `[name] + aliases + [acronym]`), `evidence`.
  - `summary_data = {"summary": summaries.default, "highlights": summaries.highlights}`.
- `loaders.load_index(index_dir)` becomes: if v2 → `to_legacy(load_index_v2(...))`
  then the existing `validate_index` as today; else the current v1 path.
  Nothing else in the v1 pipeline changes.

### 4. Migration — `ats_tailor/migrate.py`
`migrate_v1_to_v2(src_dir, dst_dir, profile_path=None) -> MigrationReport`:
- Skill ids: slugify `name` (lowercase, spaces/`/`→`-`, collapse dashes) with
  overrides `C# → csharp`, `C++ → cpp`, `.NET → dotnet`,
  `JavaScript / Node.js → javascript`; error on collision.
- Skill `aliases` = v1 `ats_keywords` minus the name (case-insensitive);
  `acronym` = the shortest all-caps alias ≤ 6 chars if one exists.
- Project/role `ats_tags` partition: a tag matches a skill iff it equals
  (case-insensitive) a skill name, alias, or acronym → goes to `skills`;
  otherwise → `keywords`. Report counts per item and the global set of
  unmapped tags.
- Bullets: responsibilities → `id=<item>.r<n>`, impact → `<item>.i<n>`,
  role bullets → `<item>.b<n>` (1-based). `skills` on a bullet = skill ids
  whose name/alias/acronym appears in the text (word-boundary,
  case-insensitive; reuse `scoring.keyword_in_text`). `metrics` = regex
  matches of numbers with optional units/%/x/+ (e.g. `15%`, `5,100+`,
  `2M+`, `47 ms`, `1.000`).
- Roles with empty `bullets` → `hidden: true` plus a YAML comment
  `# hidden: no bullets yet` on that key.
- Periods via `Period.parse`.
- `summary.yaml` → `summaries.yaml` (`default`, `variants: {}`, `highlights`).
- `education` → `education.yaml`.
- `MigrationReport` printed as a table: per item tags→skills / keywords
  counts, bullets created, metrics found, hidden roles, unmapped tags.
- Refuse to write into a non-empty `dst_dir` unless `--force`.

### 5. CLI — `ats_tailor/cli.py`
`ats-tailor <cmd>`:
- `tailor …` — passes through to the existing `tailor.main` argv unchanged.
- `validate <index_dir>` — loads either layout, prints errors (exit 1) and
  warnings (exit 0).
- `migrate <src> <dst> [--force]`.
- `schema [--out schema/]` — writes `index.schema.json` plus one file per
  top-level kind (`project`, `role`, `skills`, `certifications`,
  `education`, `summaries`).
Update the top-level `python -m ats_tailor.tailor` entry so it keeps working.

### 6. Examples and docs
- Convert `examples/index/` to the v2 layout using the migrator; keep the
  original at `examples/index_v1/` (used by tests).
- Commit generated `schema/*.json`.
- README: new "Index layout (v2)" section, the `ats-tailor` CLI table, and
  `uv sync` setup. Keep the v1 notes under a "Legacy layout" heading.

### 7. Tests (`tests/`, pytest)
- `Period.parse`/`format` round trips for every v1 date format in
  `examples/index_v1` and the parent repo's `../index/*.yaml`.
- Model validation: duplicate id, dangling ref, hidden role with no bullets
  passes, visible role with no bullets fails.
- Migration round trip on `examples/index_v1`: `to_legacy(load_index_v2(dst))`
  vs the original v1 dicts — assert equal `responsibilities`/`impact`/
  `bullets` lists, equal period strings, and every original `ats_tag`
  (lowercased) ∈ lowercased legacy `ats_tags`.
- ruamel round trip: load a v2 project file containing a comment, change one
  bullet's text via `save_project`, reload → comment and key order intact.
- `detect_layout` on both example dirs.
- Existing `tests/test_core.py` still passes.
- Smoke: `python -m ats_tailor.tailor --index examples/index --profile
  examples/profile.yaml --jd examples/jd/sample-backend.txt --company acme
  --role backend --output <tmp>` produces `resume.tex` (uses the cached
  embedding model; set `ATS_LLM_EXPAND=false ATS_RERANK=false`).

## Out of scope for this phase
SQLite store, `check`/`eval`, scoring changes, API, UI. Do not migrate the
user's real `../index` in place — only run the migrator against it into a
scratch directory and report.
