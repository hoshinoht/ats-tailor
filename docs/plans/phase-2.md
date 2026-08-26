# Phase 2 — ATS check, labels, eval baseline

Reference: `docs/v2-design.md` §2 (ATS requirements) and §6 (evaluation).
Runs on the v1 pipeline over the v2 index layout; produces the baseline
numbers that Phase 3 must beat.

## Deliverables

### 1. `ats_tailor/atscheck.py`
- `compile_pdf(tex_path, out_name) -> CompileResult{pdf_path, pages, log_path}`
  using `latexmk -pdf -interaction=nonstopmode -halt-on-error` in the tex
  file's directory; then `latexmk -c` to remove aux files. Page count from
  `pdfinfo` if available, else the `Output written on … (N pages)` log line.
  `out_name` = `<First>_<Last>_Resume_<Company>.pdf` (non-alnum → `_`).
- `extract_text(pdf_path) -> str` via `pdftotext -layout -enc UTF-8` (the
  encoding flag is mandatory: Xpdf-based builds default to Latin-1 and turn
  every non-ASCII character into U+FFFD).
- `parse_check(text, profile) -> ParseReport` (list of `{check, ok, detail}`):
  - text layer has ≥ 500 non-whitespace characters, contains no `(cid:NNN)`
    markers and no U+FFFD replacement characters (font without Unicode map);
  - every date range uses an ASCII hyphen with both a start and an end
    (`Mon YYYY - Mon YYYY` / `- Present`); an en dash or a bare single year
    fails (confirmed Workday import failure);
  - name appears on the first non-empty line;
  - email, phone, `linkedin.com/in/<handle>`, `github.com/<handle>` each
    found by regex;
  - section headings `SUMMARY`/`PROFESSIONAL SUMMARY`, `EDUCATION`,
    `EXPERIENCE`, `PROJECTS`, `SKILLS`, `CERTIFICATIONS` found, in template
    order (case-insensitive; the template uppercases);
  - for every rendered role/project: a line containing its title/name and a
    date range matching `(\w{3,9}\.? \d{4}|\d{4}) – (\w{3,9}\.? \d{4}|\d{4}|Present)`
    on the same `-layout` line (use the run's selection list as input,
    not guesses);
  - reading order: heading positions strictly increasing;
  - pages == 1.
- `keyword_coverage(text, terms: dict[str, list[str]], index_terms: set[str],
  regions: dict[str, str]) -> CoverageReport`:
  - `terms` maps a canonical term → its surface forms (name, aliases,
    acronym); a term is *present* if any surface form matches with
    `scoring.keyword_in_text` semantics.
  - `regions` = the text split into `summary`, `skills`, `bullets` (by
    heading positions) so each hit records *where* it appears.
  - Per term: present, count, regions hit, `in_skills_and_bullet` flag.
  - Missing terms partitioned: `not_in_index` (no item/skill in the index
    has that term) vs `dropped` (index has it, resume doesn't).
  - Stuffing warnings: any term with count > 3.
  - Scores: `must_coverage`, `nice_coverage`, `context_rate` (share of
    present must-terms that appear in both skills line and a bullet).
- `auto_terms(jd_text, index) -> (must, nice)`: baseline term source when
  no labels exist — taxonomy terms (skill names/aliases/acronyms and item
  keywords) found in the JD; must if the term occurs in a chunk whose
  `compute_jd_weights` weight is 1.5, else nice.
- `write_reports(out_dir, parse_report, coverage_report)` → `ats_check.md`
  and `ats_check.json`.

### 2. Labels — `labels/<jd-slug>.yaml` (in the user's repo, tracked)
```yaml
jd: jd/cpf.txt
company: cpf
role: sre-intern
pin_projects: [carehaven]        # must be selected
pin_roles: [razer-swe]
exclude: [wishbone]              # must not be selected
must_terms: [Kubernetes, Prometheus, Grafana, Linux]
nice_terms: [Terraform, AKS]
ranking:                          # optional full preference order
  projects: [carehaven, virtualkopis, digital-kopitiam]
```
- `ats_tailor/labels.py`: `Label` pydantic model, `load_labels(dir)`,
  `slug(jd_path)`.
- CLI `ats-tailor labels init <jd-file> --company --role [--labels-dir]`
  scaffolds a file with `must_terms`/`nice_terms` pre-filled from
  `auto_terms`, everything else empty, so the user only edits.

### 3. CLI additions
- `ats-tailor check <output_dir> --jd <file> --profile <file> [--labels <file>]`
  compiles if no PDF, runs parse + coverage, prints a summary, writes reports.
  Exit code 1 if any parse check fails.
- `ats-tailor tailor … --check`: after rendering, run compile + check and
  print the summary (v1 `tailor.main` gains this flag; keep default off).
- `ats-tailor eval --index <dir> --labels-dir <dir> --jd-dir <dir>
  --out eval/ [--llm] [--rerank]`:
  - for each label file: run the pipeline into `eval/runs/<slug>/`, then
    `check`; collect `must_coverage`, `nice_coverage`, `context_rate`,
    `parse_ok`, `pages`, `precision@k` for projects and roles vs pins,
    `exclusion_violations`, Kendall τ vs `ranking` when present;
  - print a per-JD table plus means; write `eval/<config_hash>.json` where
    `config_hash` = sha256 of (all `ATS_*` config values, embed model,
    index content hash, git HEAD of ats-tailor if available);
  - `ats-tailor eval --compare <a.json> <b.json>` prints a per-metric delta
    table.

### 4. Template/rendering adjustments (v1 template, minimal)
- Section heading strings exactly as in design §2 (`Summary` instead of
  `Professional Summary`).
- If the parse check shows `\hfill` dates landing on a separate
  `pdftotext -layout` line, switch role/project header lines to a
  `\begin{tabular*}`-free form that keeps date on the same line (e.g.
  `\textbf{Title} -- Org \hfill Date` already does; verify before changing).
- `render_resume` gains an optional `target_title` that, when given, is
  appended to the summary as `Targeting: <title>.` — no other text changes.

### 5. Tests
- `parse_check` on the compiled `examples/` resume: all checks pass.
- `keyword_coverage` on synthetic text/index: presence via alias, region
  attribution, partition of missing terms, stuffing warning.
- `auto_terms` on `examples/jd/sample-backend.txt` returns at least the
  obvious stack terms.
- `eval` on `examples/` with one label file produces the JSON and the table;
  `--compare` on two copies prints zero deltas.

### 6. Docs
- README: `check`, `labels`, `eval` sections; ATS requirements summary
  linking to `docs/v2-design.md` §2.

## Out of scope
Requirement extraction, scoring changes, SQLite, API, UI. The user labels
their JDs after this phase lands (scaffolds via `labels init`).
