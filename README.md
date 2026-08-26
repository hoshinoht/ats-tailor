# ats-tailor

Semantic resume tailoring for ATS. Scores your YAML resume index against a job description using sentence embeddings, keyword matching, and recency weighting, then renders a one-page LaTeX resume. All processing is local.

## Setup

```bash
uv sync --extra dev     # creates .venv from pyproject.toml + uv.lock
uv run ats-tailor --help
```

`pip install -r requirements.txt` also works — `requirements.txt` is a pointer to
`-e .`, so it installs the same dependency set from `pyproject.toml`.

Optional extras: `--extra mlx` (local MLX inference), `--extra dev` (pytest).

## CLI

```bash
ats-tailor <command> [options]
```

| Command | Description |
| ------- | ----------- |
| `tailor …` | Generate a tailored resume. Arguments are identical to `python -m ats_tailor.tailor` (see below). |
| `validate [index_dir]` | Load an index in either layout; print schema errors (exit 1) and non-fatal warnings (exit 0). |
| `migrate <src> <dst> [--force]` | Convert a v1 index directory into the v2 layout and print a migration report. |
| `schema [--out DIR \| --index DIR]` | Write JSON Schema files for the v2 index. `--out` (default `schema/`) writes a standalone copy; `--index DIR` writes them into `DIR/.schema/`, where the files' `$schema` hints point. |
| `check <output_dir> …` | Compile the rendered resume and check that an ATS can read it: parse checks plus keyword coverage. Exit 1 if any parse check fails. |
| `labels init <jd> --company --role` | Scaffold the per-JD ground truth file `eval` scores against. `labels list` prints what is on disk. |
| `eval …` | Run the pipeline over every label file, score it, print the table, write `<config-hash>.json`. `--compare a.json b.json` diffs two results. |

`python -m ats_tailor.tailor …` remains supported and is unchanged.

## Usage

```bash
python -m ats_tailor.tailor \
  --index examples/index \
  --profile examples/profile.yaml \
  --jd examples/jd/sample-backend.txt \
  --company acme \
  --role backend \
  --output /tmp/ats-test
```

| Arg             | Default                      | Description                                      |
| --------------- | ---------------------------- | ------------------------------------------------ |
| `--jd`          | stdin                        | Job description text file                        |
| `--company`     | required                     | Company name                                     |
| `--role`        | required                     | Role name                                        |
| `--index`       | `index/`                     | YAML index directory                             |
| `--profile`     | `profile.yaml` next to index | Profile YAML (name, email, links)                |
| `--output`      | `output/{company}-{role}`    | Output directory                                 |
| `--model`       | `all-MiniLM-L12-v2`          | SentenceTransformer model name                   |
| `--llm`         | off                          | Expand JD keywords via local LLM (see below)     |
| `--no-llm`      | off                          | Disable env-enabled LLM expansion                |
| `--llm-backend` | `auto`                       | LLM backend: `auto`, `mlx`, `lmstudio`, `ollama` |
| `--rerank`      | off                          | Re-score top candidates with a cross-encoder     |
| `--no-rerank`   | off                          | Disable env-enabled cross-encoder reranking      |
| `--check`       | off                          | Compile the PDF and run the ATS check afterwards |

## Configuration

All CLI flags can be set via a `.env` file in the ats-tailor repo root. Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

| Variable                    | Default                       | CLI override    |
| --------------------------- | ----------------------------- | --------------- |
| `ATS_EMBED_MODEL`           | `all-MiniLM-L12-v2`           | `--model`       |
| `ATS_LLM_MODEL`             | `qwen3.5:4b`                  | `--llm`         |
| `ATS_LLM_BACKEND`           | `auto`                        | `--llm-backend` |
| `ATS_LLM_NUM_CTX`           | `4096`                        | —               |
| `ATS_LLM_TEMPERATURE`       | `0.3`                         | —               |
| `ATS_LLM_TWO_PASS`          | `false`                       | —               |
| `ATS_LLM_TEMPERATURE_P1`    | `0.2`                         | —               |
| `ATS_LLM_TEMPERATURE_P2`    | `0.5`                         | —               |
| `ATS_LMSTUDIO_URL`          | `http://localhost:1234`       | —               |
| `ATS_MLX_MODEL`             | `mlx-community/Qwen3-8B-4bit` | —               |
| `ATS_RERANK`                | `false`                       | `--rerank`      |
| `ATS_MAX_EXPERIENCE`        | `3`                           | —               |
| `ATS_MIN_EXPERIENCE`        | `2`                           | —               |
| `ATS_MAX_PROJECTS`          | `4`                           | —               |
| `ATS_MIN_PROJECTS`          | `2`                           | —               |
| `ATS_MAX_SKILL_LINES`       | `4`                           | —               |
| `ATS_MAX_PROJECT_BULLETS`   | `3`                           | —               |
| `ATS_MIN_PROJECT_BULLETS`   | `2`                           | —               |
| `ATS_MAX_EXP_BULLETS`       | `3`                           | —               |
| `ATS_MIN_EXP_BULLETS`       | `2`                           | —               |
| `ATS_MAX_CERTIFICATIONS`    | `3`                           | —               |
| `ATS_MIN_CERTIFICATIONS`    | `3`                           | —               |
| `ATS_CERT_THRESHOLD`        | `0.20`                        | —               |
| `ATS_CHARS_PER_BULLET_LINE` | `80`                          | —               |
| `ATS_MAX_PAGE_LINES`        | `72`                          | —               |
| `ATS_SECTION_PRIORITY`      | `experience`                  | —               |

CLI flags override env vars; env vars override `.env` file values.

## LLM keyword expansion

Tag overlap scoring uses exact string matching — if the JD says "AI Engineer" it won't match an `ats_tag` like "Deep Learning" because the literal string isn't there. The `--llm` flag sends the JD to a local [ollama](https://ollama.com) model to infer implied technical keywords, which are appended to the matching pool before scoring.

```bash
# use default model (qwen3.5:4b)
python -m ats_tailor.tailor --llm ...

# use a specific model
python -m ats_tailor.tailor --llm qwen2.5-coder:1.5b ...

# omit flag entirely to skip expansion (default)
python -m ats_tailor.tailor ...
```

**Default model:** `qwen3.5:4b`. Context window defaults to 4096 (`ATS_LLM_NUM_CTX`).

**Category-aware prompting.** The candidate's skill category names are injected into the prompt so the LLM prioritises terms the candidate can actually match, while still including important terms outside those areas.

**Two-pass mode** (`ATS_LLM_TWO_PASS=true`). Uses two LLM calls: pass 1 infers the 3-5 most relevant domains from the candidate's skill categories, pass 2 expands keywords scoped to those domains. ~2x latency but more targeted results on generic JDs. Falls back to single-pass automatically on failure. Each pass has its own temperature setting (`ATS_LLM_TEMPERATURE_P1` for domain classification, `ATS_LLM_TEMPERATURE_P2` for keyword expansion).

### Backends

Three backends are supported (`--llm-backend` / `ATS_LLM_BACKEND`):

| Backend    | How it works                                                   | Setup                                     |
| ---------- | -------------------------------------------------------------- | ----------------------------------------- |
| `mlx`      | In-process via [mlx-lm](https://github.com/ml-explore/mlx-lm)  | `pip install mlx-lm`                      |
| `lmstudio` | HTTP to [LM Studio](https://lmstudio.ai) OpenAI-compatible API | Start server: `lms server start`          |
| `ollama`   | HTTP to [Ollama](https://ollama.com) REST API                  | `ollama serve` + `ollama pull qwen3.5:4b` |
| `auto`     | Try MLX → LM Studio → Ollama, skip on failure                  | Any of the above                          |

**LM Studio notes:**
- Default URL: `http://localhost:1234` (override with `ATS_LMSTUDIO_URL`)
- Uses whatever model is loaded in the app; the `--llm` model name is sent as a hint
- For thinking models (e.g. Qwen3.5), disable reasoning in LM Studio's server settings for faster, cleaner output

If the backend is unreachable the pipeline prints a warning and continues without expansion. Token usage is logged to stderr after each LLM call.

Expanded terms appear in the console output and in the `match_report.md` under **LLM-Expanded Keywords**.

If `ATS_LLM_EXPAND=true` is set in the environment, pass `--no-llm` to force a non-LLM run.

## Output

| File              | Description                                     |
| ----------------- | ----------------------------------------------- |
| `resume.tex`      | One-page LaTeX resume (compile with `pdflatex`) |
| `prompt.md`       | LLM prompt for optional bullet rewriting        |
| `match_report.md` | Similarity scores and rankings                  |
| `match_viz.png`   | Visual score breakdown                          |
| `selection.json`  | What the run selected, in score order (input to `check` and `eval`) |
| `ats_check.md/.json` | Parse + keyword report, written by `check` or `tailor --check` |

## ATS requirements

What "ATS-capable" means here is pinned down in `docs/v2-design.md` §2; the
short version, all of it enforced by `ats-tailor check` against the compiled
PDF rather than the LaTeX source:

- **Parseable.** Single column, no tables or text boxes, a text layer of real
  characters (no `(cid:NNN)`, no U+FFFD), and reading order equal to visual
  order.
- **Standard headings, exact strings** — `Summary`, `Education`, `Experience`,
  `Projects`, `Skills`, `Certifications` — present and in that order.
- **Contact block as printed text**: name on its own line; email, phone,
  `linkedin.com/in/<handle>` and `github.com/<handle>` each recoverable by
  regex, never carried only by an icon or a link target.
- **Dates are ASCII hyphens with both endpoints** (`Mon YYYY - Mon YYYY`, or
  `- Present`), on the same `pdftotext -layout` line as the entry title. En
  dashes and bare single years are a confirmed silent Workday import failure.
- **One page.**
- **Keywords with context, never stuffing.** Must-have terms should appear in
  the Skills line *and* in a bullet; nothing appears more than three times.
  A term the index has no claim for is reported as a gap — the resume is
  assembled from index claims by id, so it is never invented to fill one.

### `check`

```bash
ats-tailor check output/cpf-sre-intern --jd jd/cpf.txt --index index
```

Compiles `resume.tex` if there is no PDF yet (add `--recompile` to force it),
copies the result to `<First>_<Last>_Resume_<Company>.pdf`, then prints a
summary and writes `ats_check.md` / `ats_check.json` into the run directory.
Exit code is 1 when a parse check fails, so it drops straight into a script.

Every JD term lands in one of four states:

| State | Meaning | What to do |
| --- | --- | --- |
| `covered` | the JD's own surface form is on the page | nothing |
| `synonym-only` | present through an alias, not the JD's word | align the rendered surface form |
| `missing (have it)` | the index supports it, selection or the page budget dropped it | re-rank or make room |
| `missing (gap)` | no index claim carries it | report it, never stuff it |

Terms come from `--labels <file>` when you have one, otherwise they are derived
from the JD by `auto_terms`. Either way a term may be written the way the JD
writes it, as an alias, or as the index's canonical name — it resolves to the
same alias set, and exactness is judged against the spelling you gave.
`--jd` may be omitted to run the parse checks alone.

### `labels`

Ground truth lives in the resume repo, one tracked file per JD:

```bash
ats-tailor labels init jd/cpf.txt --company cpf --role sre-intern \
  --labels-dir labels --index index
```

The scaffold pre-fills the term lists and leaves every judgement empty for you
to fill in. Terms are written in the JD's own wording (the posting says `K8s`,
so the label says `K8s`), tiered by the section they appear under — a
requirements heading makes a term must-have, a "nice to have" heading or plain
prose makes it nice — and JD technologies with no index claim behind them are
included as nice terms marked `# not in index`, so a scaffold can surface real
gaps and not just taxonomy hits:

```yaml
must_terms:
  - Kubernetes
nice_terms:
  - Terraform  # not in index
```

The full shape:

```yaml
jd: jd/cpf.txt
company: cpf
role: sre-intern
pin_projects: [carehaven]   # ids the run must select
pin_roles: [razer-swe]
exclude: [wishbone]         # ids it must never select
must_terms: [Kubernetes, Prometheus, Grafana, Linux]
nice_terms: [Terraform, AKS]
ranking:                    # optional full preference order
  projects: [carehaven, virtualkopis, digital-kopitiam]
```

### `eval`

```bash
ats-tailor eval --index index --labels-dir labels --jd-dir jd --out eval
```

Runs the pipeline for every label into `eval/runs/<slug>/`, compiles, checks,
and prints a per-JD table with means:

| Metric | Meaning |
| --- | --- |
| `must` / `nice` | share of the tier's terms present on the page |
| `ctx` | share of present must-terms in the Skills line *and* a bullet |
| `parse` / `pg` | parse checks passed, and the page count |
| `P@k proj` / `P@k role` | precision of the top-*k* selected items against the pins (*k* = number of pins) |
| `excl` | excluded ids the run selected anyway — should be 0 |
| `tau` | Kendall τ against the label's full preference order |

A metric with nothing to measure prints `—` and is left out of the mean. A
label whose JD file is not on this machine is skipped with a note and listed
under the table — labels are tracked, postings are local, so a missing one is
normal rather than a failed sweep.
Results are written to `eval/<config-hash>.json`, where the hash covers every
`ATS_*` setting, the embedding model, the index content and the engine's own
sources — so two runs are only comparable when the hash says they are, and
editing the scorer changes the hash even before it is committed. The git HEAD
is recorded alongside for provenance but not hashed, so committing unchanged
code does not rename a baseline:

```bash
ats-tailor eval --compare eval/8672be16bb0d.json eval/<new>.json
```

## Index layout (v2)

One YAML file per entity, so edits produce small readable diffs. See
`examples/index/` for a working example and `docs/v2-design.md` §3 for the
rationale.

```
index/
  .schema/<kind>.schema.json    # generated; what the $schema hints resolve to
  education.yaml                # list of education entries
  summaries.yaml                # default summary + archetype variants + highlights
  certifications.yaml           # list of certifications
  skills/<category-slug>.yaml   # one category of the skill taxonomy
  roles/<id>.yaml               # one work role; filename stem must equal `id`
  projects/<id>.yaml            # one project; filename stem must equal `id`
```

```yaml
# projects/carehaven.yaml
id: carehaven
name: CareHaven
type: academic                          # optional, free string
repo: hoshinoht/carehaven               # optional
period: {start: 2025-10, end: 2026-04}  # YYYY-MM or YYYY; end may be `present`
summary: Full-stack caregiver platform with an event-driven backend
tech_stack:
  languages: [C#, Dart, Go]
  frameworks: [.NET 10, Flutter]
skills: [csharp, dotnet, grpc]          # skill ids from skills/*.yaml
keywords: [Wolverine, event-driven]     # item-specific terms outside the taxonomy
bullets:
  - id: carehaven.r1                    # stable; never renumber
    kind: responsibility                # responsibility | impact
    text: Architected a gRPC API gateway with Firebase auth and rate limiting
    skills: [grpc, firebase]
    metrics: []                         # quantities a rewrite must preserve
    variants: {}                        # archetype -> alternate phrasing
```

Key points:

- `ats_tags` are gone. An item's matching terms are its `keywords` plus the
  name/aliases/acronym of every skill it and its bullets reference.
- Ids are unique across roles, projects and certifications; every skill id and
  evidence `ref` must resolve. A role with no bullets must be `hidden: true`.
- Periods are structured; the renderer formats them as `Mon YYYY - Mon YYYY`
  with an ASCII hyphen. En-dash date ranges are a silent ATS import failure.
- Files are written with `ruamel.yaml` round-trip, so comments and key order
  survive edits made by the CLI.
- Only fields that differ from their default are written, so a bullet with no
  skills, metrics or variants is just `id`/`kind`/`text`.
- An index is self-contained: `migrate` writes `.schema/` into the destination,
  and each file opens with a `# yaml-language-server:` header pointing at it
  (`../.schema/...` from `projects/`, `roles/`, `skills/`; `.schema/...` from the
  top level). Regenerate with `ats-tailor schema --index index/`. The copy at the
  repo root, written by `ats-tailor schema`, is the canonical one.

Run `ats-tailor validate index/` after editing.

### Legacy layout (v1)

The original four-file layout still loads unchanged — the pipeline detects it by
the absence of an `index/projects/` directory:

- **skills.yaml** -- categories with `ats_keywords` and `proficiency`
- **projects.yaml** -- projects with `tech_stack`, `responsibilities`, `ats_tags`
- **experience.yaml** -- roles with `bullets`, `skills_used`, `ats_tags`; education entries
- **certifications.yaml** -- certs with `issuer` and `ats_keywords`

`examples/index_v1/` is a v1 index kept for reference and tests. To move an
existing index to v2:

```bash
ats-tailor migrate index/ index_v2/
ats-tailor validate index_v2/
```

The migrator partitions each `ats_tags` list into skill references and
keywords, turns `responsibilities`/`impact`/`bullets` into structured bullets
with pre-filled metrics, parses period strings, and marks bulletless roles
`hidden`. It refuses to write into a non-empty destination without `--force`,
and never modifies the source.

To start with your own data, copy `examples/`, fill in your details, and run against real job descriptions.

## Scoring

```
score(i) = (semantic(i) + keyword_bonus(i) + tag_overlap(i)) * recency(i)
           + evidence_bonus(i) + rerank_bonus(i)
```

- **Semantic** -- max-pooled cosine similarity between JD chunks and item embedding
- **Keyword bonus** -- `0.06 * log2(1 + matches)` (logarithmic, no hard cap)
- **Tag overlap** -- blend of tag-hit ratio and absolute coverage, scaled to max +0.25
- **Recency** -- x1.10 current, x1.05 <12mo, x1.02 <24mo, x1.00 older
- **Evidence bonus** -- boosts projects/roles that substantiate high-scoring skills
- **Rerank bonus** -- optional cross-encoder adjustment for top project candidates

`match_report.md` shows the total score plus semantic, keyword, tag, recency,
evidence, and rerank components so selections are easier to audit.

See [docs/algorithm.pdf](docs/algorithm.pdf) for the full writeup.

## License

GNU GPLv3 (see [LICENSE](LICENSE))
