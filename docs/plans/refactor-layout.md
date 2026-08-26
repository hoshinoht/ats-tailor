# Refactor — package layout for `ats_tailor`

Approved 2026-08-26. A pure restructuring step between Phase 4a and 4b:
moves and import rewrites only, no behaviour change. Verified by both test
suites and an unchanged v2 eval (metrics identical; only the code hash
moves).

## Target layout

```
ats_tailor/
  __init__.py
  config.py
  cli.py                      # `ats-tailor` entry (pyproject script)
  service.py                  # Phase 4a
  index/
    __init__.py
    models.py                 # ← models.py
    io.py                     # ← index_io.py
    migrate.py                # ← migrate.py
    loaders.py                # ← loaders.py (v1 loader, validate_index, date helpers, .npy cache)
  jd/                         # unchanged: models, gates, extract, requirements, hashing, prompt_extract.md.j2
  llm/                        # unchanged: backends, provider, expand
  engine/
    __init__.py
    pipeline.py               # ← pipeline.py (v2)
    adapt.py                  # ← adapt.py
    embedding.py              # ← embedding.py
    rerank.py                 # ← rerank.py
    results.py                # ← results.py
    legacy/
      __init__.py
      scoring.py              # ← scoring.py
      selection.py            # ← selection.py
      tailor.py               # ← tailor.py (v1 run, `--engine` dispatcher, `main`)
  render/
    __init__.py
    latex.py                  # ← rendering.py
    surface.py                # unchanged
    report.py                 # ← reporting.py (v1 + v2 reports, visualisation)
    template.tex.j2           # ← template.tex.j2
    prompt_template.md.j2     # ← prompt_template.md.j2
  ats/
    __init__.py
    check.py                  # ← atscheck.py
  evaluation/
    __init__.py
    evaluate.py               # ← evaluate.py
    labels.py                 # ← labels.py
  runs/
    __init__.py
    store.py                  # ← store.py
    sync.py                   # ← sync.py
    tracker.py                # ← tracker.py
    overrides.py              # ← overrides.py
  mcp/                        # Phase 4a, unchanged
  tui/                        # Phase 4b (created later)
```

## Rules
- `git mv` for every move so history follows (the coordinator commits;
  the implementer may run `git mv` — it is the one git command allowed in
  this step, and nothing else that changes git state).
- Rewrite every import in `ats_tailor/`, `tests/`, and docs; `ats_match`
  is untouched.
- Compatibility shims (one-line re-export modules with a deprecation
  comment) only where a documented external entry point moves:
  `ats_tailor/tailor.py` (`python -m ats_tailor.tailor` and
  `ats_tailor.tailor:main`), and `ats_tailor/atscheck.py`,
  `ats_tailor/evaluate.py`, `ats_tailor/store.py` if any README example
  imports them. Nothing else gets a shim; tests import the new paths.
- Template paths: `render/latex.py` and `jd/extract.py` locate their
  `.j2` files relative to their own module; `pyproject` package data must
  include `render/*.j2` and `jd/*.j2` (check the wheel build:
  `uv build` then inspect the wheel listing).
- `evaluate.code_hash()` walks the package recursively — confirm it still
  covers every `.py` and `.j2` after the move.
- The `.env`/config behaviour, CLI commands, output files and store schema
  are unchanged.

## Verification
- `uv run pytest -q` and `ATS_SMOKE=1 uv run pytest -q`: same counts as
  before the move (or more, if a shim test is added), zero failures.
- `uv run ats-tailor --help`, `uv run python -m ats_tailor.tailor --help`,
  `uv run ats-tailor mcp --help` all work; `./tailor` from the parent runs.
- `eval --engine v2 --provider none` into the scratchpad and
  `--compare eval/30a5cd1d5a71.json <new>`: every metric delta `+0.000`;
  per-run selections identical. Do not write to `eval/`.
- `grep -rn "from ats_tailor\.\(models\|index_io\|migrate\|loaders\|pipeline\|adapt\|embedding\|rerank\|results\|scoring\|selection\|rendering\|reporting\|atscheck\|evaluate\|labels\|store\|sync\|tracker\|overrides\)\b"` over the repo returns only the shims.
- README: update the architecture/module listing and any import shown.

## Out of scope
Renaming functions or classes, splitting `cli.py`, touching `ats_match`.
