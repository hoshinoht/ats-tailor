# ats-tailor

Semantic resume tailoring for ATS. Scores your YAML resume index against a job description using sentence embeddings, keyword matching, and recency weighting, then renders a one-page LaTeX resume. All processing is local.

## Setup

```bash
uv sync --extra dev     # creates .venv from pyproject.toml + uv.lock
uv run ats-tailor --help
```

`uv` is canonical; `pip install -e .` installs the same dependency set from
`pyproject.toml` if you would rather use a virtualenv of your own.

Optional extras: `--extra tui` (the Textual front-end), `--extra mlx` (local
MLX inference), `--extra dev` (pytest, pytest-asyncio).

```bash
uv sync --extra dev --extra tui
uv run ats-tailor tui --repo ..
```

The repo is a `uv` workspace with two distributions: this application, and
`packages/ats-match` — the pure matching core. `uv sync` builds both.

## Package layout

```
ats_tailor/
  cli.py           argparse front end; `ats-tailor` is `cli:main`
  service.py       the one in-process typed API every front end goes through
  config.py        settings, read from the environment and `.env`
  index/           models, the v2 directory layout, the v1 loader, migration
  jd/              reading a posting: gates, extraction, requirements
  llm/             backends (Ollama, LM Studio, MLX) and the v2 provider
  engine/          selection: pipeline, adapt, embedding, rerank, results
    legacy/        the v1 whole-item ranking, behind `--engine v1`
  render/          LaTeX, the match report, alias-aligned surface forms
  ats/             what an ATS makes of the PDF: parse checks and coverage
  evaluation/      labels and the scoring sweep
  runs/            SQLite store, index sync, tracker, per-run overrides
  mcp/             the stdio MCP server
  tui/             the Textual front-end (`tui`, `serve`)
  tailor.py        compatibility shim for `python -m ats_tailor.tailor`
```

`docs/v2-design.md` §7 has the reasoning behind the split.

## CLI

```bash
ats-tailor <command> [options]
```

| Command | Description |
| ------- | ----------- |
| `tailor …` | Generate a tailored resume. Arguments are identical to `python -m ats_tailor.tailor` (see below). |
| `validate [index_dir]` | Load an index in either layout; print schema errors (exit 1) and non-fatal warnings (exit 0). |
| `migrate <src> <dst> [--force]` | Convert a v1 index directory into the v2 layout and print a migration report. |
| `schema (--index DIR \| --out DIR)` | Write JSON Schema files for the v2 index. `--index DIR` writes them into `DIR/.schema/`, where the files' `$schema` hints point; `--out DIR` writes a standalone copy somewhere else. One of the two is required. |
| `check <output_dir> …` | Compile the rendered resume and check that an ATS can read it: parse checks plus keyword coverage. Exit 1 if any parse check fails. |
| `labels init <jd> --company --role` | Scaffold the per-JD ground truth file `eval` scores against. `labels list` prints what is on disk. |
| `eval …` | Run the pipeline over every label file, score it, print the table, write `<config-hash>.json`. `--compare a.json b.json` diffs two results. |
| `labels from-run <dir\|id>` | Merge a run's pins and exclusions into its label file. |
| `sync [--index DIR] [--embed]` | Bring the runtime store in line with the index; `--embed` pre-computes the embeddings a run would otherwise wait for. |
| `override <out_dir> …` | Edit the per-run `overrides.yaml` next to a run's output; `--show` prints it. |
| `runs list\|show\|status\|note\|quiet` | The application tracker. A `⚠` in the first column means the posting's gates said something. |
| `jd show <jd> [--provider …] [--edit]` | Read a posting into gates and weighted requirements and print them, before any run. `--edit` opens the profile in `$EDITOR` and stores it as the edited one, which outranks every later extraction. |
| `jd add [file] --company --role [--source-url]` | Store a pasted posting as `jd/<slug>.txt`, sanitised: HTML comments, `<script>`/`<style>` blocks, tags, invisible characters and blank-line runs are stripped and counted. Reads stdin without a file. The URL is recorded as text and never fetched. |
| `fit <slug> [--json]` | Cheap triage before a run: the hard gates, the best requirement coverage the index could reach, the claims carrying it, and the terms nothing covers. No selection, no LaTeX, no PDF. |
| `review-pack <run> [--json]` | One run's posting, gates, requirements, selected claims and coverage table in one payload — what a reviewing agent reads. |
| `propose <bullet_id> --archetype … [--text …]` | Propose alternate wording for one bullet and print the guardrail's verdict with a diff. Exit 1 when the guardrail refuses it. Nothing is written. |
| `propose-summary <slug> [--text …]` | The same for the Summary line of one posting's archetype. |
| `proposals list\|show\|accept\|reject` | The proposed wording waiting on a decision. `accept` is the one command here that writes tracked content. |
| `mcp --repo DIR [--provider …] [--allow-eval]` | Serve the repository to an agent runtime over stdio (Model Context Protocol). Silent while idle. `--allow-eval` additionally exposes the scoring sweep. |
| `tui [--repo DIR] [--provider …]` | The Textual front-end: applications, one run in three panes, the index, the eval baselines. Needs the `tui` extra. |
| `serve [--repo DIR] [--host] [--port]` | The same app in a browser over HTTP (`textual-serve`). Defaults to `localhost:8000`; the container image serves `0.0.0.0:8080`. |

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
| `--engine`      | `v2` (env `ATS_ENGINE`)      | `v2` covers parsed requirements; `v1` is the legacy whole-item ranking |
| `--provider`    | env `ATS_LLM_PROVIDER`       | who reads the posting in v2: `auto`, `ollama`, `lmstudio`, `mlx`, `none`, `external` |
| `--reextract`   | off                          | v2: ignore any cached extraction and read the posting again |

## Configuration

All CLI flags can be set via a `.env` file in the ats-tailor repo root. Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

| Variable                    | Default                       | CLI override    |
| --------------------------- | ----------------------------- | --------------- |
| `ATS_EMBED_MODEL`           | `all-MiniLM-L12-v2`           | `--model`       |
| `ATS_LLM_MODEL`             | `qwen3.5:4b`                  | `--llm`         |
| `ATS_LLM_PROVIDER`          | `auto`                        | `--provider`    |
| `ATS_PROMPT_VERSION`        | `extract-1`                   | —               |
| `ATS_LLM_TIMEOUT`           | `600`                         | —               |
| `ATS_LLM_PROBE_TIMEOUT`     | `1.5`                         | —               |
| `ATS_OLLAMA_URL`            | `http://localhost:11434`      | —               |
| `ATS_LLM_BACKEND`           | `auto`                        | `--llm-backend` |
| `ATS_LLM_NUM_CTX`           | `4096`                        | —               |
| `ATS_LLM_TEMPERATURE`       | `0.3`                         | —               |
| `ATS_LLM_TWO_PASS`          | `false`                       | —               |
| `ATS_LLM_TEMPERATURE_P1`    | `0.2`                         | —               |
| `ATS_LLM_TEMPERATURE_P2`    | `0.5`                         | —               |
| `ATS_LMSTUDIO_URL`          | `http://localhost:1234`       | —               |
| `ATS_MLX_MODEL`             | `mlx-community/Qwen3-8B-4bit` | —               |
| `ATS_RERANK`                | `false`                       | `--rerank`      |
| `ATS_RERANK_MODEL`          | `cross-encoder/ms-marco-MiniLM-L-6-v2` | —      |
| `ATS_RERANK_TOP_K`          | `5`                           | —               |
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
| `ATS_ENGINE`                | `v2`                          | `--engine`      |
| `ATS_TARGET_TITLE`          | `true`                        | — (v2)          |
| `ATS_MAX_REQUIREMENTS`      | `40`                          | — (v2)          |
| `ATS_FALLBACK_REQ_WEIGHT`   | `0.5`                         | — (v2)          |
| `ATS_TAU`                   | `0.1`                         | — (v2)          |
| `ATS_MU`                    | `0.3`                         | — (v2)          |
| `ATS_BETA`                  | `3.0`                         | — (v2)          |
| `ATS_BREADTH_FIRST`         | `true`                        | — (v2)          |
| `ATS_REDUNDANCY_IN_SKILLS`  | `false`                       | — (v2)          |
| `ATS_SOFT_REQ_WEIGHT`       | `0.25`                        | — (v2)          |
| `ATS_SKILL_MIN_GAIN`        | `0.03`                        | — (v2)          |
| `ATS_PIN_SKILL_GROUPS`      | `Programming Languages`       | — (v2)          |
| `ATS_HEADER_CLAIMS`         | `false`                       | — (v2)          |
| `ATS_LATEX_ENGINE`          | `auto`                        | —               |
| `ATS_DATA_DIR`              | `<index>/../data/`            | `--data-dir`    |

CLI flags override env vars; env vars override `.env` file values.

`ATS_SKILL_MIN_GAIN` is the only gain floor wired to a setting, because skills
are the only claims that cost no lines. `SectionRule.min_gain` supports one per
section, so an `ATS_ROLE_MIN_GAIN` would be a one-line change in
`adapt.section_rules()` — deliberately not wired: under v2 the last entry slot
is filled with the best remaining role whatever it scores, which is what v1 did
too. To leave a weak entry off the page, lower `ATS_MAX_EXPERIENCE` (or
`ATS_MAX_PROJECTS`), or exclude the item for that posting in its label file.

## Providers (v2)

Nothing in the v2 pipeline calls a model. It calls a **provider**, and a
provider is allowed to answer "nothing" — in which case the deterministic paths
run and the resume is the same one a machine with no model would produce.

| Provider | What it is |
| -------- | ---------- |
| `none` | Every method returns `None`. The deterministic parser reads the posting. |
| `ollama` / `lmstudio` / `mlx` | A local model, through the backends v1 already used. |
| `external` | Results supplied from outside — an agent through the MCP server (Phase 4), or a profile already in the store. Never calls a model. |
| `auto` | Probe `mlx` → `lmstudio` → `ollama` (short timeouts, and a server with no model loaded counts as absent), then fall back to `none` with one warning. |

```bash
ats-tailor tailor --jd jd/cpf.txt --company cpf --role sre --llm          # the configured provider
ats-tailor tailor --jd jd/cpf.txt --company cpf --role sre --provider ollama
ats-tailor tailor --jd jd/cpf.txt --company cpf --role sre --no-llm       # force `none`
```

With **neither** flag the provider is `none` unless `ATS_LLM_PROVIDER` names one
outright: `auto` is an invitation to go looking for a model, and an invitation
is not an instruction, so a plain `ats-tailor tailor` stays fast and
reproducible. `--llm` on v1 keeps its old meaning (keyword expansion); on v2 it
means "let the configured provider read the posting".

### Requirement extraction and gates

`jd show` prints what the next run of the same posting will use:

```bash
ats-tailor jd show jd/cpf.txt                 # deterministic reading
ats-tailor jd show jd/cpf.txt --provider ollama --reextract
ats-tailor jd show jd/cpf.txt --edit          # correct it; the edit wins from now on
```

Three rules govern what comes out:

1. **The deterministic parser always runs**, model or no model. Requirements it
   produced whose terms the extraction never mentions are appended at
   `ATS_FALLBACK_REQ_WEIGHT` (0.5) — a model that tidies a posting into eight
   lines has thrown away the ninth thing it asked for.
2. **A human edit outranks everything**, including a fresh `--reextract`.
3. **The posting is data, never instruction.** The prompt wraps it in an
   explicit trust boundary, and whatever comes back is validated by pydantic
   before anything reads it.

Extractions are cached on `(jd hash, provider, prompt version)`, so a model
sweep is paid for once and is reproducible afterwards. The deterministic profile
is stored but never served from the cache: it costs milliseconds to recompute,
and a cached copy would outlive the parser that made it.

**Gates** are hard filters — citizenship and clearance, a language the profile
does not declare, relocation, a closing date, a seniority bar. They are printed
*before* any scoring, quoted from the posting, and written into
`match_report.md` and the run record. A gate never aborts a run: the user
decides. Where a model says `pass` and the posting's own words say otherwise,
the regexes downgrade it to `flag` and attach the quote.

### Alias-aligned skills

An ATS matches strings. Where the posting writes `K8s` and the index says
`Kubernetes`, the skills line renders `Kubernetes (K8s)`; where the posting
writes `Amazon Web Services` and the index says `AWS`, it renders
`Amazon Web Services (AWS)`; where it writes `Containerization` for what the
index calls `Docker`, it renders `Docker (Containerization)`. Both forms once,
never a third time, and no term is printed twice across the section — the
canonical keyword never leaves the page to make room for the posting's synonym.
**Bullets are never rewritten here.** A bullet reaches the page in the index's
words, in an accepted variant's words, or in a per-run override's — all three
are wording a human wrote or approved (see [Variants and the
guardrail](#variants-and-the-guardrail)); the renderer invents nothing.

The Summary takes `summaries.variants[<archetype>]` when the index has one for
the archetype the posting leans on, and the posting's job title is appended once
as an honest target statement (`… Targeting: Site Reliability Engineer Intern.`,
`ATS_TARGET_TITLE`). `ats-tailor propose-summary <slug>` proposes a tightened
one under the guardrail.

### Reranking

`ATS_RERANK=true` re-scores the top `ATS_RERANK_TOP_K` claims per requirement
with a cross-encoder; `ATS_RERANK=llm` asks the provider instead (one listwise
prompt per requirement). Both are off by default on evidence, not caution — see
`docs/v2-design.md` §8 for what the cross-encoder measured.

## LLM keyword expansion (v1)

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

This whole section is about `--engine v1`. The v2 engine does not expand
keywords: putting invented terms in front of the requirement parser would tier
and weight text the posting never contained. See **Providers** above for what
`--llm` does there.

## Service and MCP

Every front-end goes through one in-process API, `ats_tailor.service.Service`.
The CLI, the MCP server and (next) the TUI are adapters: they parse arguments,
format output, and in between call one method. Nothing above that line knows
about SQLite, ruamel, latexmk or the solver.

```python
from ats_tailor.service import Service, OverridesPatch

with Service("~/projects/personal/resume") as service:
    report = service.fit("cpf")                       # gates + coverage ceiling
    run = service.tailor("cpf", "CPF Board", "SRE Intern")
    service.override(run.run_id, OverridesPatch(pin_items=["carehaven"]))
    service.set_status(run.run_id, "applied", "sent by email")
```

`Service(repo_root, *, index_dir=…, jd_dir=…, labels_dir=…, output_dir=…,
eval_dir=…, data_dir=…, profile_path=…, provider=…)` resolves the conventional
layout under `repo_root`, each directory overridable. The index and the store
load on first use, so constructing one opens nothing and imports no model.
Every method returns a pydantic model and raises `ServiceError(code, message,
details)` with `code` in `invalid_input`, `not_found`, `validation_failed`,
`compile_failed`, `provider_unavailable`, `conflict`, `not_implemented`. No
method prints: the pipeline runs quiet and everything it would have said comes
back in `RunSummary.warnings`.

### MCP server

```bash
ats-tailor mcp --repo ~/projects/personal/resume
```

Stdio, twenty-three tools (twenty-four with `--allow-eval`), the conventions of
`gofetch-mcp` / `researcher-mcp`:
`snake_case` names, `{"ok": true, …}` or `{"ok": false, "error": {"code",
"message"}}` (nothing ever raises to the client), `max_chars`/`offset`
pagination answering with `total_chars`/`truncated`/`next_offset`, a
healthcheck, and silence at idle.

| Tool | What it does |
| ---- | ------------ |
| `ats_healthcheck` | Is this repo usable — index, store, postings, whether `latexmk`/`pdftotext` are installed. Call it first. |
| `ats_list_jds` | Every posting in `jd/` with its slug, company/role, gate summary, deadline and run count. |
| `ats_add_jd(text, company, role, source_url?)` | Store a pasted posting, sanitised, and report what was stripped. Never fetches a URL. |
| `ats_fit(slug)` | Triage: gates, `must_upper_bound`, strengths, gaps, uncovered must-haves. No run. |
| `ats_extract_requirements(slug, provider?, reextract?)` | The posting as gates + weighted requirements. With provider `none`/`external` it also returns the posting (trust-wrapped) and the profile JSON schema, so the driving agent can do the extraction. |
| `ats_set_requirements(slug, profile_json)` | Store the agent's reading; it outranks every later extraction. Pydantic-validated, provenance `agent`. |
| `ats_tailor(slug, company?, role?, check?, provider?, out_dir?, record?, engine?)` | Select, render, fit to one page, check, record. `check` defaults to true; `out_dir` defaults to `output/<company>-<role>` and must be inside the repository. |
| `ats_get_run(run_id, section?, max_chars?, offset?)` | One part of a run: `summary`, `selection`, `trace`, `coverage`, `check`, `report`, `tex`. The last three paginate. |
| `ats_override(run_id, pin_items?, exclude_items?, pin_bullets?, exclude_bullets?, bullet_text?, summary?, budget_lines?)` | Edit `overrides.yaml` and re-run into the same directory. |
| `ats_compile(run_id)` | Compile to PDF; records the file's sha256 on the run. |
| `ats_check(run_id, labels?)` | ATS parse check + the four-state keyword table. |
| `ats_review_pack(run_id, max_chars?, offset?)` | Posting, gates, requirements, selected claims and coverage in one payload, for a reviewer sub-agent. |
| `ats_index_get(kind, id)` | One entity: `project`, `role`, `skill`, `certification`, `summaries`, `education`. |
| `ats_index_put(kind, id, patch_json)` | Write a fact back. Bullets merge by id; the whole index is re-validated first; the YAML round-trips with its comments. |
| `ats_index_validate` | Fatal errors and author's-call warnings. |
| `ats_runs(status?, company?)` | The tracker table. |
| `ats_outcome(run_id, status, note?)` | Record what happened. Never inferred. |
| `ats_followups(days?)` | Open applications quiet for `days` days. |
| `ats_eval(engine?, provider?, only?)` | The full scoring sweep over the labels. Slow. **Only registered with `--allow-eval`** — see below. |
| `ats_propose_variant(bullet_id, archetype, text?, requirement_ids?, jd_slug?)` | Propose wording for one bullet. Guardrail-checked, stored `pending`, writes nothing. |
| `ats_propose_summary(slug, text?)` | Propose a tightened Summary for that posting's archetype. |
| `ats_proposals(status?, bullet_id?)` | Proposals with their verdicts, newest first. |
| `ats_accept_proposal(id)` | **Writes the index**: the text lands in the bullet's `variants[<archetype>]`. The user's decision, never the agent's. |
| `ats_reject_proposal(id, reason?)` | Refuse it, with the reason. Nothing is written. |

Conduct rules, enforced in code and tested:

- **A posting is untrusted data.** JD text is only ever returned inside
  `<posting untrusted="true">…</posting>` behind a preamble saying it is data,
  not instructions. `ats_add_jd` strips markup, `<script>`/`<style>` bodies and
  invisible characters — zero-width joiners, bidi marks, the BOM, and the
  Unicode tag block a prompt injection hides in — and reports the counts.
- **Every id is validated** against `^[a-z0-9._-]+$` and resolved inside the
  repo; paths in results are repo-relative.
- **The server never fetches a URL** and shells out to nothing but
  `latexmk`, `pdftotext` and `pdfinfo`.
- **Every write is schema-checked.** `ats_set_requirements` and
  `ats_index_put` go through pydantic; a failure comes back as
  `validation_failed` with the field-by-field errors and nothing is written.
- **Gaps are reported, never filled.** Nothing reaches a page that is not an
  index claim by id.

`--provider` decides who reads a posting when a tool does not say. The default
is `none`: the deterministic parser needs nothing running, and the premise of
the MCP is that the driving agent is the model, through
`ats_extract_requirements` → `ats_set_requirements`.

`--allow-eval` adds `ats_eval` to the server. Without it the tool is **absent**
from `list_tools` — not a stub that refuses. The sweep is minutes of CPU, it
writes into `eval/`, and nothing can cancel it once it has started; an agent
that can see it will eventually spend it answering a question one `ats_fit`
would have answered. Run it from the shell (`ats-tailor eval`) unless a session
is genuinely about measuring a scoring change.

#### OpenCode

```json
{
  "mcp": {
    "ats-tailor": {
      "type": "local",
      "command": ["sh", "-c", "uv run --project $HOME/projects/personal/resume/ats-tailor ats-tailor mcp --repo $HOME/projects/personal/resume"],
      "enabled": true
    }
  }
}
```

#### Claude Code

```bash
claude mcp add ats-tailor -- uv run --project $HOME/projects/personal/resume/ats-tailor ats-tailor mcp --repo $HOME/projects/personal/resume
```

An `AGENTS.md` at the resume repo root is the thin pointer both runtimes read:
what the repo is, where the MCP is, and the honesty rules.

### The same three things from the shell

```bash
pbpaste | ats-tailor jd add --company "CPF Board" --role "SRE Intern" \
                            --source-url https://careers.example/123
ats-tailor fit cpf
ats-tailor review-pack output/cpf-sre-intern
```

`jd add` writes `jd/<company>-<role>.txt` — never over an existing file; a
collision takes a `-2` suffix — and says what it stripped:

```
wrote jd/cpf-board-sre-intern.txt  (4211 chars, 8f31a0c4e2b7)
  sanitised: 3 html tags; 1 invisible chars (58 chars)
  next:      ats-tailor fit cpf-board-sre-intern
```

`fit` answers the only question worth asking before an afternoon of tailoring —
*can this index cover this posting at all?*

```
cpf
  title       Site Reliability Engineer Intern
  archetype   DevOps & Tooling

gates
  eligibility  PASS     no citizenship or clearance clause found
  language     PASS     no language beyond the posting's own
  location     PASS     no relocation or on-site clause found
  deadline     UNKNOWN  no closing date stated
  seniority    PASS     no seniority bar found

ceiling     must 74%  nice -  (4 must / 25 requirements)

strengths
  razer-swe: razer-swe.b3 (1.97)
  virtualkopis: virtualkopis.r2 (1.48)
  forest-link-protocol: forest-link-protocol.r3 (1.06)

terms not in the index (5)
  Reliability, CPF, Azure, AKS, Terraform
```

The ceiling credits every must-have requirement with its single best-matching
claim and ignores the page budget, the redundancy penalty, and everything else
that makes a real run worse than its best case. A low ceiling is a fact about
the index, not about the selection: no amount of tailoring fixes it, and the
listed gaps are the evidence that is missing. Requirements naming nothing at
all ("works well in a team") are excluded from the average — every claim
answers them about equally well, so their ceiling measures the encoder.

## TUI

```bash
uv sync --extra tui
ats-tailor tui --repo ~/projects/personal/resume
```

Four screens, reachable with `1`–`4`. Every one of them is a client of
`Service` and nothing else: no screen imports the pipeline, the solver or the
store, and every call it makes runs on a worker thread with a busy indicator,
so a two-minute re-run never freezes the terminal.

| Screen | What it is | Keys |
| ------ | ---------- | ---- |
| **1 Applications** | Every recorded run: id, when, company/role, status, pages, must/nice/context coverage, and how long it has been quiet. A `⚠` on the status means the posting's gates flagged something. | `enter` open in Tailor · `s` set status · `n` add a note · `q` quiet-only filter · `a` add a posting (paste → sanitise → `fit`) · `ctrl+r` reload |
| **2 Tailor** | One run in three panes: the gates and the requirements coloured by coverage (green ≥ 0.6, amber ≥ 0.3, red) with their weights; the `pdftotext -layout` text — what an ATS actually reads — under a `page N / must X% / ctx Y%` header; the selected items and bullets grouped by section with the gain each one bought. | `p`/`x` pin/exclude the item · `P`/`X` pin/exclude the bullet · `w` reweight the requirement (0.25 · 0.5 · 1 · 1.5) · `r` re-run · `c` compile + check · `o` open the PDF · `i` inline raster of page 1 · `l` promote pins into the label · `ctrl+r` reload |
| **3 Index** | The tree (projects · roles · skills by category · certifications), a form for the selected entity, and an **impact** panel: where that item ranked in every run on record. Bullets edit as `<bullet id> :: <text>`, one per line; metrics are shown read-only. | `ctrl+s` validate the whole index and save · `ctrl+r` reload from disk |
| **4 Eval** | The `eval/*.json` baselines with their means, and the rendered table for whichever is highlighted. | `e` run the sweep (engine and provider are asked for) · `d` diff two baselines · `ctrl+r` reload |

Global: `?` help, `ctrl+p` provider picker (`auto`/`ollama`/`lmstudio`/`mlx`/
`none`/`external`), `ctrl+q` quit. The status bar carries the repository, the
selection and LaTeX engines, the provider and the store path.

Pins and exclusions are written into the run's `overrides.yaml` the moment they
are pressed; `r` is what pays for them. That split is deliberate — choosing is
instant, re-running is a minute — and it means the file on disk always matches
what the screen shows, even when the re-run never happens.

The layout is built for 100×30. Below 90 columns the right-hand pane is hidden
(two panes at 45 columns read; three at 30 do not); the keys still work on
whatever has focus.

`i` needs a terminal that can draw images (Kitty graphics, iTerm2 or sixel) and
`pdftoppm` on the PATH. Where either is missing it says so and points at the
rendered PNG instead of failing.

### In a browser

```bash
ats-tailor serve --repo ~/projects/personal/resume --port 8000
```

`textual-serve` runs the same app, one subprocess per connected browser. There
is no authentication in front of it: anyone who can reach the port can edit the
index and start runs, so bind it to `localhost` unless something else is doing
the authenticating.

`--public-url` is the address the served page tells the browser to load its
assets from. It only matters when that differs from the bind address — behind a
proxy, or in the container, where binding `0.0.0.0` would otherwise put
`http://0.0.0.0:8080/static/…` into the HTML. The compose file passes
`--public-url ${ATS_PUBLIC_URL:-http://localhost:8080}` for exactly that
reason.

## Container

The image is the engine — Python, Tectonic and poppler, no TeX Live — and the
repository is a bind mount. Nothing about one person's index, postings or runs
is baked in.

```bash
# from the resume repository (the parent of ats-tailor/)
docker compose -f ats-tailor/docker-compose.yaml build
docker compose -f ats-tailor/docker-compose.yaml up                  # browser UI on :8080
docker compose -f ats-tailor/docker-compose.yaml run --rm -it app tui
docker compose -f ats-tailor/docker-compose.yaml run --rm app tailor jd/cpf.txt "CPF Board" "SRE Intern" --check
docker compose -f ats-tailor/docker-compose.yaml run --rm app runs list
```

`REPO` says which repository to mount and defaults to `..` — the resume
repository this engine is a submodule of. Point it elsewhere to drive another
one with the same image:

```bash
REPO=~/work/other-resume docker compose -f ats-tailor/docker-compose.yaml run --rm app fit acme
```

Mounted: `index/`, `jd/`, `labels/`, `output/`, `eval/`, `data/` and
`profile.yaml`. Two named volumes hold the caches worth keeping between runs —
the embedding model (`hf-cache`) and Tectonic's package bundle
(`tectonic-cache`).

The LLM stays on the host: there is no GPU inside Docker on macOS, so
`ATS_LLM_PROVIDER=ollama` and `ATS_OLLAMA_URL=http://host.docker.internal:11434`
point at an Ollama running natively. MLX is not available in the container at
all. `ATS_LATEX_ENGINE=tectonic` because there is no `latexmk` in the image.

**Air-gapped runs.** The first run downloads the embedding model into
`hf-cache` and the LaTeX packages into `tectonic-cache`. After that:

```bash
HF_HUB_OFFLINE=1 docker compose -f ats-tailor/docker-compose.yaml run --rm app tailor jd/cpf.txt "CPF Board" "SRE Intern" --check
```

With `HF_HUB_OFFLINE=1` `sentence-transformers` never reaches the network and
fails loudly if the model is not in the volume, rather than hanging on a
resolve. Tectonic reads its bundle from the cache the same way.

**Ownership.** On Docker Desktop a bind mount is writable whatever uid the
container runs as. On Linux it is not — set `ATS_UID`/`ATS_GID` to your own so
`output/` and `data/` do not fill with root-owned files:

```bash
ATS_UID=$(id -u) ATS_GID=$(id -g) docker compose -f ats-tailor/docker-compose.yaml run --rm app check output/cpf-sre-intern
```

### `ATS_LATEX_ENGINE`

`auto` (the default) uses `latexmk` when it is on the PATH and Tectonic
otherwise; `latexmk` and `tectonic` name one explicitly and fail if it is
missing rather than quietly typesetting with the other. Everything downstream
reads the PDF's text layer, so the parse and keyword checks are the same either
way — which is the point: the container measures what the laptop measures.

## Output

| File              | Description                                     |
| ----------------- | ----------------------------------------------- |
| `resume.tex`      | One-page LaTeX resume (compile with `pdflatex`) |
| `prompt.md`       | LLM prompt for optional bullet rewriting        |
| `match_report.md` | Similarity scores and rankings                  |
| `match_viz.png`   | Visual score breakdown                          |
| `selection.json`  | What the run selected, in score order (input to `check` and `eval`). Under `--engine v2` it also carries the parsed requirements, per-requirement coverage, and the full selection trace |
| `ats_check.md/.json` | Parse + keyword report, written by `check` or `tailor --check` |
| `overrides.yaml`  | Per-run pins, exclusions and phrasing — hand-written or written by `override`, read by the next `--engine v2` run of the same directory |

## Runtime store

`data/ats.db` (SQLite, gitignored) holds everything derived from the index or
produced by a run: entity snapshots, the v2 embedding cache, the postings that
were tailored to, and the run records the tracker reads. The index YAML and the
labels stay the source of truth — delete the database and `ats-tailor sync`
rebuilds all of it except the run history, which was never in the index.

```bash
ats-tailor sync --index index            # entities only, idempotent
ats-tailor sync --index index --embed    # + claim, skill and cert embeddings
```

The path is `ATS_DATA_DIR` if set, else `<index>/../data/`, else `./data/`.
`sync` prints `added / updated / unchanged / pruned / embedded`; a second run
over an unchanged index reports everything unchanged and writes nothing.

The v2 engine reads and writes its embeddings here, keyed on the hash of each
individual text, so editing one bullet re-encodes that bullet and nothing else.
The v1 engine still uses its `.emb_cache/*.npy` files and is unaffected.

## Overrides

The solver decides what covers the posting best; you know things it cannot.
An `overrides.yaml` in a run's output directory says so, and the next
`--engine v2` run of that directory honours it — hard, before anything is
scored:

```yaml
pin_items: [carehaven]      # role/project ids that must be shown
exclude_items: [wishbone]   # ids that must never be shown
pin_bullets: [carehaven.i1] # bullet ids (their item is pinned too)
exclude_bullets: [razer-swe.b3]
bullet_text:                # per-run phrasing; the source facts must survive
  carehaven.r1: "Fronted 10+ services with a gRPC gateway enforcing auth and rate limits"
skills_pin: [kubernetes]    # skill ids for the Skills section
skills_exclude: [dart]
summary: null               # replacement summary text ("" drops the section)
requirements:               # edits to the extracted requirement list
  remove: [req-3]
  reweight: {req-7: 1.5}
  add:
    - {text: "Terraform / IaC", kind: must, terms: [Terraform, IaC]}
budget_lines: null          # claim-line budget, replacing the computed one
```

`bullet_text` is substituted before the score matrix is built, so an edited
bullet is what gets embedded, costed and rendered — a rephrase changes the
decision, not just the page. When an edit drops a number the source bullet's
`metrics` claimed, the run prints a warning naming the bullet and the missing
figure. It never blocks: it is your resume, and dropping a figure is sometimes
right. Quietly turning a measured achievement into a claim about nothing is
not.

Write the file by hand, or:

```bash
ats-tailor override output/cpf-sre-intern --pin carehaven --exclude wishbone
ats-tailor override output/cpf-sre-intern --text 'p1.i1=Cut deploy time by 40%'
ats-tailor override output/cpf-sre-intern --show
```

The editor round-trips through ruamel, so comments and hand-written order
survive. Ids naming nothing in the index are reported and ignored rather than
failing the run. Then re-run `tailor --engine v2 --output <same dir>`; the new
run record carries the overrides that were applied.

## Variants and the guardrail

An override changes one application's wording. A **variant** changes it for a
whole class of them: `variants[<archetype>]` on a bullet is the phrasing used
whenever a posting resolves to that archetype, and the archetype is the skill
category the posting leans on hardest (`ats-tailor fit` prints it).

```yaml
bullets:
  - id: carehaven.r1
    kind: responsibility
    text: Architected gRPC API Gateway with Firebase auth and rate limiting
    variants:
      DevOps & Tooling: Architected gRPC API Gateway with Firebase auth, rate limiting, and Docker-based rollout  # proposed 2026-08-26 via external, accepted
```

The principle is one sentence: **a model or an agent may propose wording; only
a human accepts it, and nothing reaches a PDF that the guardrail has not
passed.** The index stays the only source of facts.

### The guardrail

`ats_tailor.guardrail.check_variant(source, item, proposed, index)` returns
`{ok, violations: [{rule, detail}], diff}`. Seven rules, each of them a way a
rephrase can quietly stop being true:

| Rule | What it refuses |
| ---- | --------------- |
| `metrics` | dropping a number the source bullet claims. `15%` and `15 %` are the same figure; `15%` and `20%` are not. Numbers the bullet's text carries count even when nobody filled in `metrics`. |
| `technology` | dropping a technology the source *says* — by any of its surface forms, so "Kubernetes" still answers a source that said "K8s". |
| `new_technology` | naming a technology that is in neither the source bullet nor the item's own `tech_stack`/`skills`/`keywords`. Checked against the index alias map *and* by the same gap detector `check` uses, so a word nothing in the index has ever heard of is refused too. |
| `numbers` | inventing a figure the source never carried. |
| `length` | more than 180 characters (`MAX_BULLET_CHARS`). |
| `style` | an empty line, or a trailing full stop — index bullets do not end in one. |
| `person` | the first person. A resume bullet has no subject. |

A refused proposal cannot be accepted at all, from any front-end. The report
carries a unified diff between the index's wording and the proposal, so what
changed is visible next to why it was refused.

The Summary has one extra rule: it may only restate what the résumé already
says, so every technology and every number in a proposed summary must appear in
the index's summaries, its highlights, or the claims the posting selected.

### The loop

```bash
ats-tailor propose carehaven.r1 --archetype 'DevOps & Tooling' \
  --text 'Architected gRPC API Gateway with Firebase auth, rate limiting, and Docker-based rollout'
ats-tailor proposals list
ats-tailor proposals accept carehaven.r1.devops-tooling-1   # writes index/projects/carehaven.yaml
ats-tailor proposals reject carehaven.r1.sre-1 --reason 'invented a throughput figure'
```

Without `--text` the configured provider writes it (`--provider ollama`), from
a prompt that carries the guardrail's rules verbatim, the source bullet, its
metrics, the item's stack, the archetype and the top three requirements the
bullet is being aligned to. A provider with nothing to say is
`provider_unavailable`, never an empty proposal.

Proposals live in the runtime store (`proposals`: id, bullet, archetype, text,
the guardrail verdict as JSON, `pending|accepted|rejected`, provenance, and
when it was decided) until they are accepted — never in the index. Accepting
writes the YAML through the round-trip writer with a provenance comment beside
the variant, so a pull request shows that a sentence was proposed by a model
and approved by a person. Rejected proposals are kept: what was refused, and
why, is the record that makes the guardrail worth having.

In the TUI, `v` on a bullet in the Tailor screen opens the proposal panel:
the proposal, the verdict with each violation spelled out, `a` accept
(which writes the index), `d` reject, `e` edit and `ctrl+s` to re-check the
edit. With no model reachable the panel opens empty, for wording of your own.

### What a variant does to a run

A variant is substituted before the score matrix is built, exactly where a
`bullet_text` override is: the variant's words are what gets embedded, costed
and rendered, so a rephrase changes what the solver picks and not only what the
page prints. A per-run `bullet_text` override still wins over a variant — the
human editing this one application outranks a variant they accepted for a class
of them. The run record and `selection.json` list every variant used, with the
guardrail verdict that let it in, and the match report prints them in a
"Phrasing variants used" table.

`ats-tailor eval --no-variants` scores the index with every bullet in its own
words; `--variants-report` additionally measures what each variant is worth by
running every posting of its archetype both ways.

## Tracker

Every `tailor --engine v2` records a run — id `<company>-<role>-<timestamp>`,
status `drafted`, the selection, the coverage the check measured, the hash of
the PDF that was compiled, and the overrides applied. (`eval` passes
`--no-record`: a sweep is measurement, not an application.)

```bash
ats-tailor runs list [--status applied] [--company cpf]
ats-tailor runs show <id>
ats-tailor runs status <id> applied --note "submitted via the portal"
ats-tailor runs note <id> "recruiter replied"
ats-tailor runs quiet [--days 10]
```

One vocabulary, underscores, no synonyms:
`drafted → applied → interview → offer → hired | rejected | no_response |
offer_declined | withdrawn`; the last five are final. Moving off `drafted`
stamps `applied_at`, which is the date follow-ups are counted from, and every
status change is kept as a dated note. Outcomes are recorded, never inferred —
`runs quiet` lists open applications nobody has said anything about in N days
and stops there.

Decisions made in an override are the same judgement a label records, so they
can be promoted:

```bash
ats-tailor labels from-run output/cpf-sre-intern --labels-dir labels
```

`pin_items` merge into `pin_projects`/`pin_roles` by item kind and
`exclude_items` into `exclude`, additively, keeping the label file's existing
entries and comments.

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
ats-tailor eval --engine v1 ...              # score the legacy engine instead
ats-tailor eval --provider ollama ...        # let a model read the postings
ats-tailor eval --provider ollama --reextract  # ... and read them again
ats-tailor eval --no-variants                # ignore accepted phrasing variants
ats-tailor eval --variants-report            # ... and measure what each is worth
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

When the index carries variants or the store has proposals, a `variants`
section follows the table: the guardrail pass rate over every proposal ever
made, and — with `--variants-report` — the mean must-coverage and context
delta each variant bought, measured by running every posting of its archetype a
second time with `--no-variants`. The attribution is per archetype (variants of
one archetype are on or off together), credited to the variants the run
actually selected; measuring one bullet at a time would cost a sweep per
bullet.

A metric with nothing to measure prints `—` and is left out of the mean. A
label whose JD file is not on this machine is skipped with a note and listed
under the table — labels are tracked, postings are local, so a missing one is
normal rather than a failed sweep.
Because extractions are cached on `(jd hash, provider, prompt version)`, a
`--provider ollama` sweep pays for the model once and is reproducible from the
store afterwards; a provider that turns out to be unreachable falls back per
posting and records `provider: fallback` for that run.

Results are written to `eval/<config-hash>.json`, where the hash covers every
`ATS_*` setting, the embedding model, the index content, the selected engine,
the provider and prompt version that read the postings, and both packages' sources — so two runs are only comparable when the hash says they are, and
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
  top level). Regenerate with `ats-tailor schema --index index/`; the copy inside
  the index is the canonical one.

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

Two engines. `--engine v2` (the default since Phase 3c) parses the posting into
requirements and covers them; `--engine v1` is the legacy path, which ranks
whole items against the posting. Both render
the same template and pass the same ATS check.

### `--engine v2` — requirement coverage

```
maximise  Σ_r w_r · softpool_τ(M[r, S])   requirement coverage (submodular)
        + β · ctx(S)                      must-terms in the Skills line *and* a bullet
        − μ · redundancy(S)               near-duplicate claims inside a section
        + Σ_{c∈S} prior_c                 small additive recency / impact / proficiency priors
subject to  lines(S) ≤ budget, and the ATS_MIN/MAX_* limits per section
```

0. **Gates.** Before anything is scored, the posting's hard filters are read
   and printed with the sentence each came from (see **Providers** above).
   Nothing is blocked; the point is that a citizenship clause is seen before the
   afternoon is spent, not after.
1. **Requirements.** A provider may read the posting into requirements, gates,
   a title and an archetype; with `none` (the default) it is read by the
   deterministic parser below, whose output is also merged into any model's as a
   safety net. The posting is chunked and each chunk tiered by the
   heading above it (`ats-tailor check` uses the same tiering, so the metric and
   the objective agree by construction): a requirements heading gives `must`,
   a "nice to have" heading gives `nice`, anything else is a `responsibility`
   at half weight. Lines with no index vocabulary *and* no requirement verb are
   dropped; a line kept only by its verb ("Works well independently and with
   others") is a *soft* requirement at `ATS_SOFT_REQ_WEIGHT` — still reported,
   but not competing for page space with "a solid understanding of Kubernetes",
   since every claim answers it about equally well. Deterministic — no model,
   same answer every time.
2. **Claims.** Every bullet of every visible entry, and every skill, becomes one
   atomic claim: `cost` in rendered lines, `terms` from its skills and keywords,
   a small `prior` from recency, impact and proficiency.
3. **Matrix.** Per requirement, dense cosine ranks and BM25-over-aliases ranks
   are fused with RRF, then multiplied by a *relevance gate* — the pair's cosine,
   or the share of the requirement's terms the claim carries, whichever is
   higher. Without the gate the top claim for a boilerplate line ("A hybrid work
   model…") scores as confidently as the one that answers "Kubernetes".
4. **Selection.** Lazy greedy over gain per rendered line, then a repair pass for
   the minimums (bullets per entry, one impact bullet per project), then groups
   are opened up to the configured maxima, paid for by dropping the claims that
   earned least. Skills cost no lines, so they answer to a gain floor
   (`ATS_SKILL_MIN_GAIN`) instead of the budget — without one a skills line
   fills with whatever has the highest proficiency prior. The categories in
   `ATS_PIN_SKILL_GROUPS` always get a line; what goes *on* it still has to
   clear the floor.
5. **Page fit.** Render, compile, count pages; if it spilled, cut the budget by
   4 lines and re-solve, up to three times.
6. **Surface forms.** Skills are labelled in the posting's own spelling
   (`Kubernetes (K8s)`), the Summary takes the archetype's variant when the
   index has one, and the posting's job title is appended once as a target
   statement. Bullets are never rewritten.

`match_report.md` becomes a coverage ledger: every requirement with its kind,
weight, best claim and coverage; the greedy trace with what each pick covered;
the section/group summary; and the requirements left under-covered.

The core of this (types, BM25, RRF, greedy, metrics) is `packages/ats-match`, a
standalone library whose only dependency is numpy. It has no notion of a resume:
models are injected through `Embedder` / `Reranker` protocols, and a test fails
the build if application vocabulary leaks into it.

### `--engine v1` — item ranking

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

Measured against the labelled JDs (`ats-tailor eval`), v2 matches v1 on
must-term coverage and parse rate, both stay on one page, and v2 puts
must-terms in context (Skills line *and* a bullet) 14 points more often.

See [docs/algorithm.pdf](docs/algorithm.pdf) for the full writeup.

## License

GNU GPLv3 (see [LICENSE](LICENSE))
