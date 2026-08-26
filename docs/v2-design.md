# ats-tailor v2 — Design

Status: approved direction, 2026-08-26. This document is the reference for the
v2 rebuild. Phase plans live in `docs/plans/`.

## 1. Goals

1. **ATS-capable output.** The rendered resume must (a) parse cleanly in
   common ATS resume parsers, (b) contain the JD's must-have keywords in the
   surface form the JD uses, in both the Skills section and in-context
   bullets, and (c) be measurable: every run reports an ATS keyword-coverage
   score and a parse check.
2. **Requirement coverage, not item ranking.** Select the subset of atomic
   claims that best covers the JD's weighted requirements inside a one-page
   budget, instead of ranking whole items independently.
3. **Measurable.** An eval harness over labelled JDs; every scoring change is
   judged by numbers, not by eye.
4. **Human in the loop.** Requirement extraction, selections, and phrasing
   are all overridable per application; overrides are persisted and double as
   eval labels.
5. **Git-diffable source of truth, queryable runtime.** Index content stays
   as small typed YAML files; SQLite holds derived data (embeddings, runs,
   overrides, labels).
6. **Containerised UI.** One image (FastAPI + built React bundle), Compose
   for volumes, host Ollama for LLM calls.

## 2. ATS requirements (what "ATS-capable" means concretely)

Parseability (checked by `ats-tailor check` on the compiled PDF):

- Single column; no tables, text boxes, headers/footers, icons, images.
- Standard section headings, exact strings: `Summary`, `Education`,
  `Experience`, `Projects`, `Skills`, `Certifications`.
- Contact block as plain text in the body: name on its own line; email,
  phone, LinkedIn, GitHub each recoverable by regex from `pdftotext` output.
- Every role/project renders `title — organisation` and a date range in
  `Mon YYYY - Mon YYYY` (or `- Present`) recoverable on the same or adjacent
  `pdftotext -layout` line. **Date separators are ASCII hyphens (U+002D)
  and every entry has both a start and an end** — en-dash ranges and bare
  single years are a confirmed silent Workday import failure (lesson from
  `MadsLorentzen/ai-job-search`). Prose keeps normal typography.
- Text-based PDF with `glyphtounicode`; extraction via
  `pdftotext -layout -enc UTF-8` shows no `(cid:NNN)` and no U+FFFD;
  reading order equals visual order (test: section headings appear in
  template order).
- Contact details are printed text, never carried only by an icon or a
  hyperlink target.
- Output filename `<First>_<Last>_Resume_<Company>.pdf`; the run directory
  name is `<company>_<role>` sanitised to `[a-z0-9_]` (single path
  component whatever the posting contains).

Keyword capability (driven by the optimiser and the renderer):

- Skills are rendered in the **surface form the JD uses** (JD says "K8s" →
  render `Kubernetes (K8s)`; JD says "Amazon Web Services" → render that,
  not "AWS"). Where the JD uses an acronym or expansion, emit both once.
- Must-have terms should appear in the Skills line **and** in at least one
  bullet (contextual use). The objective rewards both.
- The target job title appears once in the Summary (as an honest target
  statement, e.g. "…undergraduate targeting Site Reliability Engineering").
- No stuffing: a term appears at most 3 times across the page.
- `ats-tailor check` reports every JD term with one of four states —
  `covered` (JD's surface form present), `synonym-only` (present via an
  alias but not the JD's exact term → the renderer should align it),
  `missing (have it)` (index supports it, selection/budget dropped it),
  `missing (gap)` (not in the index; never stuff it) — plus must/nice
  coverage %, context rate, and the parse-check result.

**Honesty rules** (the resume is only ever assembled from index claims by
id, so fabrication is structurally impossible; these govern the edges):

- A gap is reported, never papered over: no keyword appears on the page
  unless an index claim or skill carries it.
- Facts the user states in conversation must be written back to the index
  in the same session (`ats_index_put`), or they do not exist for the next
  run.
- Proposed variants (LLM or agent) pass the metric guardrail — every number,
  unit, and named technology in the variant must appear in the source bullet
  or its item's `tech_stack`/`skills` — and land in the YAML as `variants`,
  never directly in a PDF.

## 3. Index data model (source of truth)

Layout (in the user's resume repo):

```
index/
  profile.yaml
  education.yaml
  summaries.yaml                # default + variants keyed by archetype
  certifications.yaml
  skills/<category-slug>.yaml   # list of skills for one category
  roles/<id>.yaml
  projects/<id>.yaml
```

Schema (pydantic v2, `ats_tailor/index/models.py`; JSON Schema written into
`<index>/.schema/` by `migrate` or `ats-tailor schema --index`, which the files'
`$schema` hints point at):

```yaml
# skills/cloud-infrastructure.yaml
category: Cloud & Infrastructure
skills:
  - id: kubernetes
    name: Kubernetes
    proficiency: intermediate            # advanced | intermediate | familiar
    aliases: [K8s, container orchestration]
    acronym: K8s                         # optional; rendered as "Name (Acronym)" when JD uses either
    evidence:                            # kept from v1; refs must resolve
      - ref: virtualkopis
        type: project
        detail: "Deployed backend services to a Kubernetes cluster"
```

```yaml
# projects/carehaven.yaml
id: carehaven
name: CareHaven
type: academic                           # free string, optional
repo: hoshinoht/carehaven                # optional
period: {start: 2025-10, end: 2026-04}   # YYYY-MM or YYYY; end may be "present"
summary: "Full-stack caregiver platform with event-driven backend"
tech_stack:                              # rendered as the Technologies line
  languages: [C#, Dart, Go]
  frameworks: [.NET 10, Flutter]
  infrastructure: [Docker, RabbitMQ]
skills: [csharp, dotnet, grpc, docker]   # skill ids (taxonomy refs)
keywords: [Wolverine, event-driven]      # item-specific terms not in the taxonomy
bullets:
  - id: carehaven.r1
    kind: responsibility                 # responsibility | impact
    text: "Architected gRPC API gateway with Firebase auth and rate limiting"
    skills: [grpc, firebase]
    metrics: []                          # numbers/quantities the guardrail must preserve
    variants: {}                         # archetype -> alternate phrasing
```

```yaml
# roles/razer-swe.yaml
id: razer-swe
company: Razer
title: Software Engineering (Cloud) Intern
type: internship                         # internship | research | fulltime
period: {start: 2022-04, end: 2022-09}
hidden: false                            # true = never selected (e.g. not started yet)
skills: [csharp, dotnet, react]
keywords: [Razer Synapse]
bullets: [...]                           # same Bullet schema
```

Rules:

- `ats_tags` no longer exist. Matching terms for an item =
  its `keywords` ∪ aliases of its `skills` ∪ aliases of skills its bullets cite.
- Bullet ids are stable and referenced by overrides/labels. Never renumber.
- Periods are structured; rendering formats them. Year-only precision allowed.
- Validation: ids unique across roles/projects/certs; skill ids unique; all
  refs resolve; bullets non-empty unless `hidden`; text ≤ 180 chars.
- IO uses `ruamel.yaml` round-trip so the UI/CLI can write files back without
  destroying key order or comments.

Migration from v1 (`ats-tailor migrate`): lossless by construction —
`ats_tags` are partitioned into `skills` (where a tag matches a skill name or
alias, case-insensitive) and `keywords` (the rest); `responsibilities` become
`kind: responsibility` bullets, `impact` become `kind: impact`; period strings
are parsed to structured periods; metrics are pre-filled by regex. A
compatibility loader turns the new layout back into the v1 dict shape so the
v1 pipeline keeps running until v3 scoring replaces it.

## 4. Runtime store (SQLite, derived, gitignored)

`data/ats.db`, built by `ats-tailor sync`; rebuildable from the index at any
time.

- `entities(id, kind, content_hash, json)`
- `embeddings(content_hash, model, vec BLOB)` — replaces `.emb_cache/*.npy`
- `jd_profiles(jd_hash, model, prompt_version, json, edited_json)`
- `runs(id, jd_hash, config_hash, created, status, json)`
- `run_notes(run_id, created_at, note)` — the tracker's history

Built in 3b (`ats_tailor/runs/store.py`), with two tables from the original sketch
deliberately absent:

- **`overrides`** is not its own table. An override is a file the user edits
  (`<out_dir>/overrides.yaml`) and a copy inside the run record's `json`. A
  third representation would be a third thing to keep in sync, and the file is
  the one a human reads.
- **`labels`** stays YAML in the repo. Ground truth is reviewed in pull
  requests and hand-edited; putting it in a gitignored database would make the
  eval's fixed point untrackable. `ats-tailor labels from-run` moves a run's
  decisions *into* the YAML instead.

`entities.id` is namespaced by kind (`project:carehaven`): items and skills are
validated in separate id namespaces in the index and share one primary key
here. Vectors are stored as float32 bytes with their `dim`, row-normalised on
write and re-normalised on read; a fresh encode is read back through the store
before use so a cold run and a warm one cannot differ in the last bits.

## 5. Pipeline v2

```
JD text ──► 1 extract ──► JDProfile{requirements[{id,text,kind,category,weight,terms[]}], title, archetype,
                                     gates{eligibility, language, location, deadline, seniority}}
                           (local LLM or agent via MCP, cached; deterministic fallback; user-editable)
                           gates are hard filters reported before any scoring: a citizenship/PR
                           requirement, an undeclared language, a relocation, or a past deadline
                           is surfaced with the quoted posting text, never silently scored
index ────► 2 claims ────► Claim{id, item_id, kind, text, skill_ids, terms}
                           one per bullet + one per skill-evidence detail
        ──► 3 matrix ────► M[req, claim] = RRF(dense, bm25_alias, rerank_topk)
                           dense: asymmetric model (bge/gte/e5 with query/passage prefixes)
                           bm25: over alias-expanded tokens; rerank: local LLM or bge-reranker on top-5 per req
        ──► 4 select ────► lazy-greedy submodular maximisation, with constraints + pins/excludes
        ──► 5 layout ────► render → compile → page count → shrink budget → re-solve (≤ 3 iters)
        ──► 6 artifacts ─► PDF, .tex, coverage report, ATS check, run record
```

Objective for stage 4:

```
maximise  Σ_r  w_r · cov_r(S)                     requirement coverage (submodular)
        + β · Σ_t∈must  [t in skills line] · [t in some selected bullet]   keyword-in-context bonus
        − μ · Σ_{c,c'∈S} sim(c, c')                redundancy
subject to  lines(S) ≤ budget
            2 ≤ bullets(item) ≤ 3 for shown items; ≥1 impact bullet per project when available
            MIN/MAX projects, roles; hidden items excluded; pins in, excludes out
cov_r(S) = softpool_{c∈S} M[r,c]   (log-sum-exp with temperature τ; τ→0 gives max)
w_r: must=1.0, nice=0.5, responsibility=0.7, scaled by extraction weight
```

Item priors (recency, proficiency) enter as small additive terms on claims,
never as multipliers on totals. The greedy trace (which requirement each
claim was picked for) is stored and shown in the UI.

Skills line: same coverage objective over the taxonomy; group by category;
≤ 8 per line; order by requirement weight covered; `familiar` skills only when
the JD names them; surface form chosen by alias alignment with the JD.

Summary: `summaries.yaml` variant chosen by archetype; the target title is
inserted from the JDProfile.

## 6. Evaluation

`labels/<jd-slug>.yaml`: `pin_projects`, `pin_roles`, `exclude`, `must_terms`,
`nice_terms`, optional full ranking. `ats-tailor eval` prints per-JD and mean:

- must-term coverage, nice-term coverage (from `check`)
- precision@k for projects/roles vs pins; Kendall τ vs full ranking when given
- parse-check pass rate; page-fit iterations
- before/after table when given two config hashes

## 7. Package boundary, front-ends, MCP, and container

Decided 2026-08-26 (supersedes the earlier FastAPI + React plan).

**Matching core is separate from resume features.** Two distributions in
one `uv` workspace (introduced in Phase 3 when the core is born; Phase 1
ships a single package):

```
packages/ats-match/      pure matching library — no YAML, LaTeX, or resume vocabulary
  ats_match/
    types.py       Requirement{id,text,kind,weight,terms}, Claim{id,text,terms,cost,group,attrs},
                   Constraints{group min/max, total cost budget, pins, excludes}
    lexical.py     BM25 over alias-expanded tokens
    fusion.py      RRF / weighted fusion
    select.py      lazy-greedy submodular coverage maximisation + trace
    eval.py        coverage@budget, P@k, Kendall τ on abstract labels
    providers.py   Protocols: Embedder, Reranker (injected; core has no model deps)

ats-tailor/              resume application — depends on ats-match (repo root)
  ats_tailor/
    cli.py         argparse front-end; the `ats-tailor` script is `cli:main`
    service.py     one in-process typed API used by every front-end
    config.py      settings, from the environment and `.env`
    index/         models, IO, migration, validation
    jd/            JD store, extraction (LLM/agent/fallback), gates
    llm/           backends (Ollama, LM Studio, MLX) and the v2 provider
    engine/        pipeline (v2), adapt, embedding, rerank, results
      legacy/      scoring, selection, tailor — the v1 whole-item ranking
    render/        LaTeX template, the match report, alias-aligned surface forms
    ats/           check.py: parse check + keyword table
    evaluation/    labels + the scoring sweep and its config fingerprint
    runs/          store (SQLite), sync, tracker, per-run overrides
    tui/  mcp/  http/
    tailor.py      shim: `python -m ats_tailor.tailor` still reaches the v1 entry
```

`adapt.py` (`Index` → claims: bullets/evidence become a `Claim` whose group is
the item id and whose cost is its estimated line count; a `Selection` becomes a
render context) sits in `engine/`, next to the pipeline that uses it. The
layout above was reached by a move-only refactor between Phases 4a and 4b —
`docs/plans/refactor-layout.md`.

Boundary rules: `ats_match` never imports `ats_tailor`; it has no notion of
"project", "role", or "page"; models are injected through the provider
protocols. `ats_tailor` is the only place that knows what a resume is.

```
ats_tailor/service.py   one in-process typed API: run, override, compile,
   │                    check, index CRUD, eval, requirement get/set, tracker
   ├─ cli/              argparse (existing commands + tui, mcp, serve)
   ├─ tui/              Textual app; `ats-tailor tui`; browser via `textual serve`
   ├─ mcp/              stdio MCP server (official `mcp` Python SDK)
   └─ http/             FastAPI adapter — optional, only for remote clients
```

`service.py` and `mcp/` landed in 4a; `tui/` is 4b and `http/` is still
optional. Three properties of the service are load-bearing for every adapter
after it:

- **It never prints.** The pipeline is driven with `quiet=True` and everything
  it would have said comes back in `RunSummary.warnings`. A front-end speaking
  a protocol over stdout cannot survive a library that logs to it, and the MCP
  additionally redirects `sys.stdout` to stderr for the duration of every call
  because `sentence-transformers` and `latexmk` print regardless.
- **One error type.** `ServiceError(code, message, details)` with a code from a
  closed set, so an adapter maps errors once instead of catching `ValueError`,
  `KeyError`, `FileNotFoundError` and `ValidationError` at every call site.
- **Nothing is eager.** The index and the store load on first use, so
  constructing a `Service` opens no file and imports no model — which is what
  makes `ats_healthcheck` able to answer "this index does not load" instead of
  failing to start.

**LLM provider interface** (`ats_tailor/llm/provider.py`): `extract(jd)`,
`rerank(requirement, claims)`, `propose_variant(bullet, archetype,
requirements, **context)`, `propose_summary(summary, archetype, title,
allowed)`.
Providers: `ollama`, `lmstudio`, `mlx`, `none` (deterministic fallbacks),
`external` (results are supplied through the service — used by the MCP so
the driving agent is the LLM). No step in the pipeline calls a model
directly.

**TUI (Textual).** *Delivered in 4b (`ats_tailor/tui/`).* Screens, reachable
with `1`–`4`: Applications (runs + status + coverage + days quiet, `⚠` when
the posting's gates flagged something), Tailor (three panes: requirements
coloured by coverage · `pdftotext -layout` preview, i.e. what an ATS sees ·
the selected items and bullets with the gain each bought; `p`/`x` and
`P`/`X` pin and exclude, `w` reweights, `r` re-runs, `c` compiles and
checks, `o` opens the PDF, `i` rasters page 1 via `textual-image` where the
terminal supports it, `l` promotes the pins into the label), Index (tree +
form, whole-index validation on `ctrl+s`, round-trip save, and an impact
panel giving the item's rank across every recorded run), Eval (baselines +
`e` run, `d` compare). `ats-tailor serve` runs the same app over HTTP with
`textual-serve` — no separate web front-end.

Two properties are load-bearing rather than cosmetic. Every service call
goes through one `run_task` on a Textual thread worker with a busy
indicator, so nothing blocks the event loop; and the app holds a single lock
around those calls, for the same reason the MCP does — this build of SQLite
is in multi-thread mode, so the store's connection may move between threads
but never be used by two at once.

**MCP (OpenCode/Claude Code compatible).** *Delivered in 4a
(`ats_tailor/mcp/`).* Same conventions as
`gofetch-mcp` / `researcher-mcp`: stdio, snake_case tools, structured
`error{code,message}` (`invalid_input`, `not_found`, `validation_failed`,
`compile_failed`, `provider_unavailable`, `conflict`, `not_implemented`),
`ats_healthcheck`, `max_chars`/`offset` pagination, silent at idle. Tools:
`ats_healthcheck`, `ats_list_jds`, `ats_add_jd`, `ats_extract_requirements`,
`ats_set_requirements`, `ats_tailor`, `ats_get_run`, `ats_override`,
`ats_compile`, `ats_index_get`, `ats_index_put`, `ats_propose_variant`,
`ats_propose_summary`, `ats_proposals`, `ats_accept_proposal`,
`ats_reject_proposal`,
`ats_eval`, plus from the `ai-job-search` lessons: `ats_fit(jd)` (cheap
triage: gates + upper-bound requirement coverage the index *could* reach,
strengths/gaps — before any tailoring; batchable over many JDs),
`ats_review_pack(run)` (JD + selected claims + coverage table in one
payload for a reviewer sub-agent, which returns edits by `bullet_id` with a
`reason` of `grounding|keyword|style`), `ats_outcome(run, status, note)`
and `ats_followups()` (tracker; see below).

MCP conduct rules (posting = untrusted data):
- JD text is returned inside an explicit trust-boundary wrapper; the server
  strips HTML comments, zero-width and invisible characters on `ats_add_jd`
  and flags that it did so.
- The server never fetches URLs (that is `gofetch`'s job) and never
  executes anything found in a posting.
- Every id in a tool argument is validated against `[a-z0-9._-]+` and
  resolved inside the repo; every index write goes through the pydantic
  schema and the ruamel round-trip writer.
- Requirements submitted by an agent (`external` provider) are schema-
  validated and stored with `provenance: agent`.
- An `AGENTS.md` in the resume repo is the single thin pointer for any
  agent runtime (OpenCode, Claude Code): where the MCP is, where the design
  doc is, and the honesty rules.

**Tracker.** `runs.status` uses one canonical vocabulary:
`drafted → applied → interview → offer → hired | rejected | no_response |
offer_declined | withdrawn` (underscores, never spaces; the last five are
final). Each run keeps the JD verbatim, the submitted `.tex`/PDF hash, and
dated notes. `ats_followups()` lists open applications quiet ≥ 10 days with
< 2 follow-ups; drafts may only restate claims present in the submitted
resume. Outcomes are recorded, not learned from (too sparse).

Config:
`{"type":"local","command":["sh","-c","uv run --project $HOME/projects/personal/resume/ats-tailor ats-tailor mcp --repo $HOME/projects/personal/resume"]}`.
A Go client MCP over the HTTP adapter can be added later if a static
binary is wanted; nothing else changes.

**Container.** *Delivered in 4b.* `python:3.12-slim` with the Tectonic
static binary and poppler-utils — no TeX Live, which is the difference
between a few hundred megabytes and a few thousand; `uv` from the official
image; `uv sync --no-dev --extra tui`; non-root; `ENTRYPOINT ["ats-tailor"]`
with `CMD serve --host 0.0.0.0 --port 8080`. `docker compose run --rm -it
app tui` for the TUI, `docker compose up` for the browser. Mounts `index/`,
`jd/`, `labels/`, `output/`, `eval/`, `data/` and `profile.yaml` from
`REPO` (default `..`), plus named volumes for the HF and Tectonic caches.
`ATS_LLM_PROVIDER=ollama`, `ATS_OLLAMA_URL=http://host.docker.internal:11434`:
the LLM stays on the host, because there is no GPU in Docker on macOS. MLX
is not available in the container at all.

`ats_tailor.ats.check.compile_pdf` gains `ATS_LATEX_ENGINE=auto|latexmk|
tectonic` (`auto` = latexmk if on PATH, else Tectonic). Everything after the
compile reads the PDF's text layer, so the parse and keyword checks are
engine-agnostic — which is the point: the container has to measure what the
laptop measures.

## 8. Phases

| Phase | Deliverable | Depends on |
|---|---|---|
| 1 | Schema, per-entity layout, migration, compat loader, `pyproject`/`uv`, CLI (`validate`, `migrate`), tests | — |
| 2 | `check` (parse + keyword coverage on PDF), labels format, `eval` baseline on v1 pipeline | 1 |
| 3a | **Delivered.** `ats-match` core (types, BM25, RRF, greedy selection, metrics) in a uv workspace; adapter index→claims; deterministic requirement parser; v2 pipeline with compile loop behind `--engine v2`; eval parity with v1 | 1, 2 |
| 3b | **Delivered.** SQLite runtime store (`entities`, `embeddings`, `jd_profiles`, `runs`, `run_notes`), `sync`, per-run `overrides.yaml` honoured by selection (pins/excludes/edited text/requirement edits), run records and the tracker, `labels from-run` | 3a |
| 3c | **Delivered.** LLM provider interface (`ollama|lmstudio|mlx|none|external`), requirement extraction + gates, alias-aligned surface forms in rendering, summary variants + target title, v2 becomes the default engine | 3b |
| 4a | **Delivered.** `service.py` (the one in-process API), stdio MCP server (19 tools by default, 20 with `--allow-eval`), `fit` / `jd add` / `review-pack` CLI commands, `AGENTS.md` | 3 |
| 4b | **Delivered.** Textual TUI (4 screens), `serve`, `ATS_LATEX_ENGINE`, Dockerfile/Compose | 4a |
| 5 | **Delivered.** Variant proposals + summary polish with metric guardrail (agent via MCP or local model), approval in TUI | 4 |

### 3a result (2026-08-26)

`ats-tailor eval --engine v2` over the 14 labelled JDs
(`eval/f647ef74f75b.json`), against the Phase 2 v1 baseline
`eval/d184bc9cec11.json`:

| metric | v1 | v2 | delta |
|---|---|---|---|
| must_coverage | 75.02% | 75.02% | +0.000 |
| nice_coverage | 34.33% | 33.44% | −0.009 |
| context_rate | 37.35% | 51.43% | **+0.141** |
| parse_ok | 100% | 100% | +0.000 |
| pages | 1.00 | 1.00 | +0.000 |
| exclusion_violations | 0 | 0 | +0.000 |

The exit criterion (must, context and parse ≥ v1, one page everywhere) holds, so
v2 is ready to become the default in 3c. Three things had to be measured rather
than assumed, and each is recorded where it is implemented:

- **Rank fusion alone is not a score.** Min-max over an RRF row makes every
  requirement's best claim a 1.0, so a posting's boilerplate looked as covered
  as its "Kubernetes" line and the greedy spent the page on soft skills. The
  fused rank is multiplied by a relevance gate — cosine, or the share of the
  requirement's terms the claim carries, whichever is higher
  (`ats_match.fusion.build_matrix`).
- **β = 0.2 is inert.** Coverage is a weighted sum over every requirement while
  `ctx` is one share in [0, 1]. At 3.0 the context bonus moves the metric ~9
  points; past ~5 it starts buying context at the expense of coverage.
- **A requirement that names nothing is noise with a weight on it.** A third of
  a posting is culture and benefits prose ("Works well independently and with
  others"). Every claim answers it about equally well, so the weight it carries
  is spent on nothing; at `ATS_SOFT_REQ_WEIGHT` = 0.25 it stays in the report
  and stops steering the page (context +4.2 points).
- **Free claims need a floor, not a budget.** A skill costs no lines, so the
  greedy keeps taking them for whatever their proficiency prior is worth.
  `SectionRule.min_gain` stops that. The floor is sharp: 0.0–0.01 removes
  nothing, 0.02–0.03 removes the tail, 0.05 deletes skills the posting asked
  for (must 75.0% → 74.1%, context 51.4% → 43.2%). Shipped at 0.03.
- **The objective cannot see a header.** Opening a group costs lines but its
  title and organisation are on the page and an ATS reads them. Groups are
  therefore opened up to `max_groups` *after* the greedy has chosen depth on
  merit, paid for by dropping the lowest-gain claims — a layout decision, not
  an objective term (`ats_match.select._open_to_max`).
- **Making the header a claim does not work, and the reason is instructive.**
  `ats_match` supports header claims (`attrs["header"]`: included free when a
  group opens, dropped with it) and `adapt.header_claims` builds them, behind
  `ATS_HEADER_CLAIMS`, off by default. Measured: context_rate 51.4% → 37.9%
  with the floor off, and must_coverage 75.0% → 74.2% with it on; the term the
  feature was designed to recover ("Software Engineering", which exists only in
  a job title) is still missing either way. Two mechanisms, both real: a
  header's coverage saturates the requirements a keyword-bearing bullet would
  otherwise have been picked for, and `atscheck` builds its bullets region by
  dropping entry and `Technologies:` lines, so a header can never satisfy the
  in-context test the bonus optimises. Headers are worth *rendering*, not
  worth *bidding with*.

The cross-encoder reranker (`ATS_RERANK`, `cross-encoder/ms-marco-MiniLM-L-6-v2`
over the top 5 claims per requirement) was measured once, not shipped: must
75.02% → 75.02%, nice 33.44% → 34.04%, context 51.43% → 48.30%, at 2.2× the wall
clock (45.8s → 101.2s for 14 JDs, 3.3 → 7.2 seconds per posting). It moves
nothing that matters and loses three points of context. 3c should spend that
latency budget on requirement extraction instead.

Deviations from `docs/plans/phase-3a.md`, all deliberate:

- The soft pool is `τ·log(1 + Σ(exp(x/τ) − 1))` rather than plain log-sum-exp,
  which would score the empty set −∞ and let zero-scoring claims accumulate
  `τ·log|S|` of coverage they never earned.
- The greedy ranks by gain *per rendered line*, not raw gain: with a budget
  rather than a cardinality limit, raw gain spends the page on long bullets.
- The keyword-in-context bonus counts the terms a claim's own words carry, not
  the ones it is tagged with — the keyword check reads the page, and a bullet
  tagged `docker` need not say "Docker" (`adapt.PAGE_TERMS_ATTR`).

### 3b result (2026-08-26)

The store, overrides and tracker are infrastructure, so the exit criterion was
that they change *nothing*: `ats-tailor eval --engine v2` over the same 14
labelled JDs, with every embedding now coming from `data/ats.db` instead of
`index/.emb_cache/*.npy`, reproduced `eval/f647ef74f75b.json` exactly —
must 75.02%, nice 33.44%, context 51.43%, parse 14/14, pages 1.00, and every
per-JD selection, budget and gap list identical (`eval/c353d77de549.json`, kept
as the baseline until 3c superseded it; the config hash moves only because
`code_hash` covers the new modules).

That parity is not automatic and cost two design decisions:

- **float32 storage must not leak into the run.** A vector encoded now and a
  vector read back from SQLite later have to be the same vector, or a cold run
  and a warm one drift apart in the last bits for no reason a user could ever
  diagnose. `encode_store` therefore writes and then *re-reads* what it just
  encoded, so every path through the cache returns the stored value.
- **Batch composition had to be checked, not assumed.** Where the `.npy` cache
  keyed on the whole list of texts, the store keys on each text, so a run after
  an edit encodes a different batch than the one that filled the cache.
  Measured on `all-MiniLM-L12-v2`: encoding a text alone, in a pair, or in a
  full batch gives bit-identical vectors, so per-text caching is safe. A model
  where that stopped being true would show up as eval drift with no code change
  behind it.

Deviations from `docs/plans/phase-3b.md`, all deliberate:

- No `overrides` or `labels` table (see §4): the override is the file plus a
  copy in the run record, and ground truth stays in tracked YAML.
- `entities.id` is namespaced by kind, since a skill and a project may share an
  id in the index but not in one primary key.
- `sync --embed` also pre-computes certification texts, which the v2 pipeline
  now reads from the store too — otherwise the "no run waits for a model"
  promise would hold for everything except the Certifications section.
- `eval` passes `--no-record` rather than the tracker inspecting its caller. A
  sweep is measurement; the flag says so where it can be read.

### 3c result (2026-08-26)

`ats-tailor eval --engine v2 --provider none` over the same fourteen labelled
JDs (`eval/30a5cd1d5a71.json`), against the Phase 3b baseline it replaces
(`c353d77de549`, deleted with this phase — its numbers are the "3b" column):

| metric | 3b | 3c | delta |
|---|---|---|---|
| must_coverage | 75.02% | 75.02% | +0.000 |
| nice_coverage | 33.44% | 34.46% | **+0.010** |
| context_rate | 51.43% | 51.43% | +0.000 |
| parse_ok | 100% | 100% | +0.000 |
| pages | 1.00 | 1.00 | +0.000 |
| exclusion_violations | 0 | 0 | +0.000 |

`--engine v1` reproduces the Phase 2 baseline exactly — every mean and every
per-JD field identical to `d184bc9cec11` — so the default flip costs the legacy
path nothing.

Every selection, budget, requirement count and fit-iteration count is
byte-identical to 3b: with `provider none` the extractor *is* the Phase 3a
parser, and the two things that did move are both rendering. Five postings lost
a `synonym-only` verdict — `Containerization` (cpf, infineon), `Dashboards`
(hitachi), `Agentic AI` (razer), `K8s` (tencent) are now `covered`, in the
posting's own spelling — and no term anywhere regressed. The single point of
nice-coverage is `Robotics` on dConstruct, which reached the page through the
target statement. Stuffing warnings are unchanged in number (40 across the
corpus, before and after); six postings shift one term by one occurrence.

No live model was reachable on this machine (`ollama list` is empty, `mlx-lm` is
not installed, LM Studio is not running), so the `--provider ollama` arm of the
exit criterion is **not** measured. The provider path is exercised by
`FakeProvider` in `tests/test_phase3c.py`, and `--provider external` runs it
end to end through the store.

Five things had to be decided rather than assumed:

- **A cached deterministic profile is a lie waiting to happen.** Extractions are
  cached on `(jd_hash, provider, prompt_version)`, but the *parser's* profile is
  stored and never read back: it costs milliseconds to recompute, and a cached
  copy would outlive the parser that produced it — an eval that reports no
  change after a scoring change is worse than one that reports the wrong change.
  Its row is also not keyed on a prompt version (it used no prompt), so a prompt
  bump cannot orphan a human's edit to it.
- **The plan's language regex would have failed five of the fourteen postings.**
  `proficien\w+ in (?P<lang>[A-Z]\w+)` reads "Proficiency in Python" as a
  language requirement. The gate matches against a whitelist of natural
  languages instead. A gate that cries wolf is a gate the user learns to skip,
  and then the one real citizenship clause goes past unread.
- **Alias alignment can print the same word twice.** A `SQL` skill aligned onto
  a posting's "PostgreSQL" renders `PostgreSQL (SQL)` next to a `postgresql`
  skill rendering `PostgreSQL`. Labels are therefore deduped by the *parts* they
  put on the page, across the whole Skills section rather than per line.
- **An aligned label keeps both names, never just the posting's.** The plan's
  "any other alias → `jd_form`" was shipped, measured and reverted within the
  phase: rendering CPF's "Containerization" *in place of* "Docker" satisfies the
  posting at the cost of the better-known keyword, and an ATS that matches one
  of them would have matched both. Every branch now emits a pair —
  `Kubernetes (K8s)`, `Amazon Web Services (AWS)`, `Docker (Containerization)` —
  and only the order differs, the acronym going in the brackets because the
  expansion is what a human reads. With one exception, found by looking at what
  the pair rule printed: when the index name already spells the posting's word
  as a whole word ("Git / Version Control" for "Git", "LSTM / GRU" for "LSTM"),
  the bracket is dropped — that is the same token twice on one line, which is
  the stuffing the alignment exists to avoid rather than a second surface form.
  Ten labels across six of the fourteen postings align, each adding a form the
  page did not already carry.
- **A title guess is a claim on the page, so it is allowed to be absent.** The
  deterministic reader finds a title on 5 of 14 postings (an explicit
  `Job Title:` label, a heading-shaped line *ending* in a role noun, or a
  "seeking a … Engineer" sentence) and answers `None` on the rest. An earlier
  version that merely *contained* a role noun offered "Internship Learning
  Outcomes" as Infineon's job title. A model provider is expected to do better;
  being silently wrong is what is not allowed.

Deviations from `docs/plans/phase-3c.md`, all deliberate:

- **`--llm` is not the v2 default.** With neither `--llm` nor `--no-llm` the
  provider is `none` unless `ATS_LLM_PROVIDER` names one outright; `auto` alone
  does not opt a plain `tailor` run into probing for a model and spending
  minutes on extraction. `auto` is what `--llm` resolves to.
- **`ProviderReranker` batches by requirement, not "≤ 8 requirements per
  call".** `ats_match.Reranker` is called once per requirement with that
  requirement's top-k pairs, so there is no cross-requirement batch to build;
  pairs are grouped by query, which for the only caller is the same thing.
- **Alias alignment reaches the skills line only.** The plan also names "the
  certification keyword hints"; the template prints certifications as
  `name -- issuer` and has no keyword hints to align.
- **`RequirementSpec.weight` is not clamped to [0.5, 1.5]** — that band is
  applied to *extracted* weights in `parse_profile`. The deterministic parser
  weights a requirement that names nothing at `ATS_SOFT_REQ_WEIGHT` (0.25), and
  clamping the field would have quadrupled the page budget spent on boilerplate.
- **Two carry-overs from 3b landed here:** `embedding.encode_cached` is deleted
  (nothing called it once the store existed), and `tracker.run_id` counts up
  `-2`, `-3` … when an id is already in the store rather than overwriting the
  record it collides with.

One interaction worth naming, because it is a gap in a report rather than a bug
in a run: a term that reaches the page *only* through the target statement
("Robotics" on dConstruct) counts as `covered` by `ats-tailor check` and drops
off the gap list, even though no index claim carries it. The sentence is honest
— it says what the application is *for*, not what was done — but the gap report
is quieter than it was. Phase 4 should either exclude the target statement's own
words from the coverage tally or keep listing them as gaps.

### 4a result (2026-08-26)

`ats_tailor/service.py` (one API, ~1900 lines) and `ats_tailor/mcp/`
(twenty tools over it, nineteen served by default, ~890 lines), plus
`ats-tailor fit | jd add | review-pack | mcp` and the parent repo's
`AGENTS.md`. 46 new tests; the suite is 322 passing with `ATS_SMOKE=1`.

Nothing in `pipeline.py`, `adapt.py`, `select.py`, `evaluate.py` or the
renderer was touched, so the v2 baseline (`eval/30a5cd1d5a71.json`) still
stands by construction. Outside the new files the diff is three things: one
line of `store.py` (`check_same_thread=False`, below), four new `cli.py`
commands, and `mcp` in `pyproject.toml`.

Six things had to be decided rather than assumed:

- **A stray `print` is a dead MCP session.** The check in `tailor.py` prints
  unconditionally, so the service grew a silent twin of it rather than a `say=`
  parameter threaded through the pipeline: the point was to leave the pipeline
  byte-identical, and "the library does not print" is a better invariant than
  "the library prints unless asked not to". The tool wrapper redirects stdout
  to stderr anyway, because the model loader and `latexmk` are not ours.
- **The SDK runs sync handlers on a thread pool.** This build of SQLite is in
  multi-thread mode (`sqlite3.threadsafety == 1`): a connection may move
  between threads but never be used by two at once. The store now opens with
  `check_same_thread=False` and the MCP holds one lock around every tool call —
  which also means two `ats_tailor` calls cannot race into one output
  directory. The alternative, a connection per thread, would have made
  "close the store" unanswerable.
- **An edit outranks extraction, whichever reader would have run.** The
  extraction cache is keyed on `(jd_hash, provider, prompt_version)`, so an
  edit stored under `external` is invisible to a `--provider none` run. A
  correction is a statement about the posting, not about which model would
  otherwise have read it, so `set_requirements` writes the deterministic row
  *and* the service's own provider row.
- **`fit`'s ceiling excludes requirements that name nothing.** "Works well
  independently" is answered about equally well by every claim in any index, so
  including it measures the encoder rather than the evidence. Those
  requirements are still counted and still reported; they are not averaged.
  On `jd/cpf.txt` the real index reaches `must_upper_bound` 0.74 over 4 hard
  must-haves out of 25 requirements, with `Reliability, CPF, Azure, AKS,
  Terraform` as terms no claim carries.
- **`index_put` creates as well as edits.** The honesty rule in §2 is that a
  fact stated in conversation is written back in the same session; a writer
  that could only ever edit an existing entity would fail on the first new job.
  Creation goes through the same whole-index validation, so a half-specified
  entity is a `validation_failed` with the missing fields named and nothing on
  disk.
- **Provenance is a YAML comment, not a field.** `# added 2026-08-26 via mcp`
  beside a new bullet's id is for the human reading the pull request. As a
  schema field it would reach the store, the report, the legacy projection and
  every JSON payload, for no reader's benefit.

Deviations from `docs/plans/phase-4a.md`, all deliberate:

- **`mcp.server.mcpserver.MCPServer`, not `mcp.server.fastmcp.FastMCP`.** The
  current SDK is 2.x, where `FastMCP` was renamed; importing the old path
  raises with a migration message. The alternative was pinning `mcp<2` to
  match the plan's spelling, which is a worse trade than one renamed class.
- **`Service` takes an override for every directory, not just two.** Tests
  need a whole repository in `tmp_path`, and a service that can only relocate
  `index/` and `data/` would have written runs into the real `output/`.
- **`add_jd` records company, role and source URL in `data/jd_meta.json`.**
  The posting file holds the posting; a URL written into it would change the
  text that gets hashed and extracted. A gitignored sidecar beside the store
  is derived data, which is what it is. No new SQL table: a schema migration
  for four strings is not worth an upgrade path.
- **`fit` also returns `uncovered` and `blocking_gates`.** The plan's
  `strengths` and `gaps` say what the index has and what it lacks; neither
  says which must-have nothing answers, which is the one a person acts on.
- **`ats_get_run` has JSON sections as well as text ones.** `summary`,
  `selection`, `coverage` and `check` are bounded and are returned whole;
  only `report`, `tex` and `trace` paginate.
- **`ats_tailor` takes named parameters, not the plan's `options` object.**
  A free-form object gives an agent no schema to read; `check`, `provider`,
  `out_dir`, `record` and `engine` are arguments with types and defaults, and
  `out_dir` is additionally checked to resolve inside the repository — it is
  the one argument that could otherwise write a rendered resume anywhere on
  the disk.
- **`ats_eval` is gated behind `ats-tailor mcp --allow-eval`.** The plan lists
  it in the tool set; a plain server does not register it. It is minutes of
  CPU, it writes into `eval/`, and nothing cancels it once it has started, so
  an agent that can see it will eventually spend it answering a question one
  `ats_fit` would have answered. Absent beats present-and-refusing: a tool
  that is not in `list_tools` is not a tool anybody has to reason about.

### 4b result (2026-08-26)

`ats_tailor/tui/` (four screens over `Service`, ~2,100 lines plus a
215-line stylesheet), `ats-tailor
tui | serve`, `ATS_LATEX_ENGINE`, a `Dockerfile` and a `docker-compose.yaml`.
45 new tests; the suite is 369 passing with `ATS_SMOKE=1`.

The pipeline is untouched. Re-running the sweep against the real index with
`--engine v2 --provider none` reproduces `eval/30a5cd1d5a71.json` metric for
metric — must 75%, nice 34%, context 51%, parse 100%, 1.00 pages, 0
exclusion violations, every delta `+0.000`. The config *hash* moves
(`3e00250f3069`), because `code_hash` covers every `.py` in the package and
`tui/` is new; that is the fingerprint working as designed, not drift.

Six things had to be decided rather than assumed:

- **Choosing and paying are separate keystrokes.** `p`/`x`/`P`/`X`/`w` write
  `overrides.yaml` immediately through a new `Service.set_overrides`, and `r`
  (`Service.rerun`) is what spends the minute. `Service.override` refuses an
  empty patch — correctly, for an agent — so a front-end that edits the file
  one pin at a time needed the two halves apart. The payoff is that the file
  on disk always matches the screen, even when the re-run never happens.
- **The TUI holds the same lock the MCP does.** Textual runs every worker on
  its own thread and this build of SQLite is in multi-thread mode: two
  concurrent service calls segfaulted the process within a minute of the first
  pilot test. One lock around every call fixes it, serialises the store, and
  incidentally stops two re-runs racing into one output directory. A long eval
  sweep therefore blocks the other screens, which is honest: it is one
  repository and one store.
- **A tab has to hand focus to its screen.** Every single-letter key is a
  binding on the screen's container, and Textual only reaches a binding from a
  widget inside it. Left alone, `TabbedContent` focuses the tab bar and `s`,
  `p` and `e` do nothing at all — silently. Each tab now focuses its primary
  widget on activation.
- **`ctrl+p` was already taken.** Textual binds it to the command palette. The
  palette is off (`ENABLE_COMMAND_PALETTE = False`): a provider picker is the
  thing this app has that a generic palette does not.
- **The impact panel needed a new question of the store.**
  `Service.item_usage(item_id)` reports, for every recorded run, whether the
  item was selected and where its first bullet came in the greedy trace. An
  item no run has ever picked reports `rank: None` rather than being omitted —
  "the solver has never reached for this" is the finding an editor wants, and
  nothing else in the repository says it.
- **Bullets edit as `id :: text`.** `index_put` merges bullets by id, so the id
  has to survive the round trip through a TextArea. Showing it also makes the
  refusal legible: a line without one is rejected with its line number, rather
  than reaching the schema as a bullet with no `kind`.

Container: **1.33 GB**, `python:3.12-slim` + Tectonic 0.17.0 + poppler.
Cold run (model and LaTeX bundle downloading into the volumes) 3m37s; warm
run with `HF_HUB_OFFLINE=1` **8.0s**, 1 page, `check` 14/14 parse checks,
must 100%. `docker compose up` serves the TUI on :8080 (verified over the
websocket: the app boots and paints inside the container).

The naive image was **10.7 GB**. `sentence-transformers` pulls PyPI's torch,
which on Linux is the CUDA build — on aarch64 too, since NVIDIA publishes
those wheels now — and uv's download cache kept every wheel in the layer.
Reinstalling the *same* torch version from PyTorch's CPU index, dropping the
`nvidia-*`/`cuda-*` distributions and running `uv cache clean` in the same
layer is 8× smaller for an image that will never see a GPU: on macOS there is
none to see, which is why the provider talks to an Ollama on the host.

Deviations from `docs/plans/phase-4b.md`, all deliberate:

- **`textual-serve`, not `textual serve`.** The `textual` CLI is `textual-dev`
  and its `serve` subcommand is the `textual-serve` package; `ats-tailor serve`
  uses that package's `Server` API directly, which is one dependency rather
  than two and needs no shell quoting of a nested command line.
- **`ATS_LATEX_ENGINE` reaches four call sites, not one.** The plan puts it in
  `compile_pdf`. A guard that still asks `which("latexmk")` would refuse to
  compile in an image that has Tectonic, so `Service.compile`, the silent
  check, and `pipeline._page_count` all resolve the engine instead. On a
  machine with latexmk every one of them resolves to latexmk, which is why the
  baseline reproduces exactly.
- **`healthcheck` reports `tectonic` and the resolved engine.** One more key in
  `tools` and a `latex_engine` field. `ats_healthcheck` is how an agent finds
  out it cannot compile, and "latexmk is missing" is the wrong answer in a
  container where nothing is missing.
- **The right pane hides below 90 columns; it does not reflow.** Two panes at
  45 columns read and three at 30 do not. The keys still work on whatever has
  focus, so the screen is narrower rather than less capable.
- **The Index tree covers the four kinds the plan lists.** Summaries and
  education are addressable through `index_get`/`index_put` but have no form:
  they are one-of-a-kind entities with no bullets and no impact to show.
- **`profile.yaml` is mounted too.** The plan lists six directories; the engine
  cannot render a resume without a name and contact details.
- **`i` needs `pdftoppm` as well as a capable terminal.** `textual-image` draws
  an image; something has to rasterise the PDF first. Both absences degrade to
  a message naming the PNG on disk.


### 5 result (2026-08-26)

`ats_tailor/guardrail.py` (484 lines of pure functions), a `proposals` table
(store migration 3), `Provider.propose_variant` / `propose_summary` with two
prompt templates, variant-aware claim building, five MCP tools, three CLI
commands, the TUI's `v` panel, and `eval --no-variants` / `--variants-report`.
58 new tests; the suite is 413 passing (427 with `ATS_SMOKE=1`).

The pipeline's numbers are unchanged, by construction and by measurement: no
bullet in the index carried a variant when the sweep ran, so
`ats-tailor eval --engine v2 --provider none` reproduced
`eval/30a5cd1d5a71.json` with **zero differing per-JD fields** and every mean
delta `+0.000` — must 75%, nice 34%, context 51%, parse 14/14, 1.00 pages, 0
exclusion violations. The result is filed as `eval/f3056afb53e7.json` (the
config hash moves because `code_hash` covers the new modules) and supersedes
`30a5cd1d5a71` as the current baseline. The only shape change in the JSON is
`options.variants: true`.

Seven things had to be decided rather than assumed:

- **A rule the model is told about is still a rule that is checked.** The
  prompt carries the guardrail's rules verbatim — the metrics that must
  survive, the technologies that must stay, the ones the item is allowed to
  add — because a local 8B model that is not told will not comply. It is
  checked afterwards anyway, because a model that *is* told still will not
  always. Neither half is redundant: the prompt is what makes proposals usable,
  the check is what makes them safe.
- **"No new technology" needs two detectors, not one.** The index alias map
  catches a real technology borrowed from somewhere else in the resume
  ("Kubernetes" in a project that has never used it); it cannot catch a word
  the index has never heard of. `gap_terms` — the same detector that tells the
  coverage report a posting asks for something nobody has — catches that when
  it is pointed at the proposal instead of the posting. Two rules, one
  violation type, and the second one is what stops a model inventing "Consul".
- **Every word of the source bullet is known, punctuation included.** The first
  version passed the source *text* into the gap detector, which splits on
  whitespace and slashes only: a source saying "Docker, TLS, and Flutter" left
  `tls,` in the known set and `TLS` outside it, so a faithful rephrase was
  refused for naming a technology it had not moved. Known vocabulary is
  tokenised now, keeping the punctuation technology names are spelled with
  (`C++`, `.NET`, `CI/CD`) and dropping the punctuation sentences are.
- **A sentence-initial capital is grammar, not a proper noun.** The gap
  detector treats any capitalised word as a candidate technology, which on a
  résumé bullet means the leading verb ("Designed", "Architected") is one. The
  proposal's first word is therefore known by construction. A technology in
  that position is still caught by the alias-map rule whenever the index knows
  it at all, which is the case that matters.
- **Metrics are read from the text, not only from the field.** `metrics:` is
  filled in by the migration and by `index_put`, and a hand-edited bullet may
  carry a figure nobody recorded. The guardrail unions the field with
  `migrate.find_metrics(bullet.text)` — the same reader — so "Cut p99 latency
  by 40%" is guarded whether or not anyone typed `40%` into the schema. Bare
  years stay excluded: a date is not a quantity.
- **A variant is substituted where an override is, so it can change the
  decision.** `claims_from_index(archetype=…)` emits the variant as the claim's
  text, so the variant's words are what gets embedded, costed and rewarded by
  the keyword-in-context bonus. Measured on a scratch copy of the real index:
  accepting a *second* DevOps variant changed which CareHaven bullet the solver
  picked for `jd/cpf.txt` — the longer variant cost a line and the greedy spent
  it elsewhere. A variant that only reached the renderer would have been a
  cosmetic feature with a scoring bug hiding behind it.
- **The variants report attributes per archetype and credits per selection.**
  Variants of one archetype are on or off together, and isolating one bullet
  would cost a sweep per bullet. Each posting of a variant's archetype is
  therefore run twice, and the delta is credited to the variants the run
  actually *selected* — a variant the solver never reached for did not move the
  number, whatever else changed. It found a real negative on the scratch index
  immediately: a `virtualkopis.r2` variant that spelled out "containerised …
  Docker images … Prometheus dashboards" cost CPF 25 points of context rate
  (100% → 75%) while must coverage stayed at 100%. That is the report working:
  the variant reads better and measures worse.

Deviations from `docs/plans/phase-5.md`, all deliberate:

- **The TUI's proposal panel is a modal, not a side pane.** The Tailor screen
  is three panes at 90 columns and already hides the third below that; a fourth
  would have nowhere to go. Every other "collect one answer" surface in the app
  is a modal, and the keys the plan asks for (`a`/`d`/`e`, plus `ctrl+s` to
  re-check an edit) are the same keys either way.
- **`propose_variant` takes `jd_slug` as well as `requirement_ids`.** A
  requirement id (`req-3`) means nothing without the posting it was extracted
  from. With a slug the ids resolve against that posting; without one the
  recorded runs are searched newest-first, which is what the TUI means, where
  the ids on screen came from the run that is open. An id nothing resolves is
  dropped rather than handed to a model as the literal string "req-3".
- **A rejection's reason is a column, not a side file.** `proposals.note`
  exists from the first migration. The alternative was writing it into the
  gitignored JD metadata sidecar, where nothing else about a proposal lives.
- **`--no-variants` turns off the summary variant too.** Since this phase a
  summary variant is an accepted proposal like any other, and a flag that
  measured the page without the bullets' variants but with the summary's would
  be measuring neither state.
- **The summary's corpus is summaries, highlights and selected claims — not
  the skills taxonomy.** A skill the index lists and no bullet demonstrates is
  a line on the Skills row, not a sentence about what the candidate has done.
  With no run of the posting on record, every visible bullet is the corpus,
  which is the superset a first application has to be judged against.
- **The eval record gained no field.** The variants report needs each run's
  archetype and the variants it used; both are read from the run's own
  `selection.json` rather than added to the eval JSON, so a sweep of an index
  with no variants stays comparable field-for-field with every baseline taken
  before this phase existed.
- **`ats_propose_variant` answers `ok: true` for a proposal the guardrail
  refused.** `ok` is whether the *call* worked; `guardrail_ok` is whether the
  proposal passed. An agent that conflated them would accept a refused
  rewrite, so the two are separate keys and the violations come with the
  answer.

### Live extraction result (2026-08-27)

The `--provider ollama` arm of the Phase 3c exit criterion, deferred in 3c
because no local model was reachable, is now measured. `gpt-oss:20b` on Ollama
(`ATS_LLM_NUM_CTX=16384`, `ATS_RERANK=false`), over the same fourteen labelled
JDs, against the current `none` baseline (`eval/f3056afb53e7.json`):

| metric | none | ollama | delta |
|---|---|---|---|
| must_coverage | 75.02% | 74.02% | −0.010 |
| nice_coverage | 34.46% | 40.16% | **+0.057** |
| context_rate | 51.43% | 46.90% | −0.045 |
| parse_ok | 100% | 100% | +0.000 |
| pages | 1.00 | 1.00 | +0.000 |
| exclusion_violations | 0 | 0 | +0.000 |

**A model is still not the better reader**, so `ATS_LLM_PROVIDER` keeps its
default and `--provider ollama` stays opt-in. What the run bought and what it
cost are both worth writing down, because the first is a fixed bug and the
second is the next one.

**The bug: a model told to copy the posting's spelling is a model whose answers
the index cannot match.** `prompt_extract.md.j2` asked for the posting's literal
words ("K8s", not "Kubernetes"), and `parse_profile` resolved them by exact
lookup in the alias map. A posting does not write alias-shaped strings: it
writes "Azure Kubernetes Services (AKS)", "Infrastructure as Code (IaC)",
"DevOps CI/CD workflows". Across the fourteen postings **81% of the 274 terms
the model produced resolved to nothing** — and a requirement whose terms resolve
to nothing is scored by the embedding channel alone, because `term_overlap` is
empty and the BM25 channel of `build_matrix` has no folded token to find. Half
the scoring went dark on the majority of what the model said, which is the worst
possible failure mode: silent, and invisible in the means.

Two changes, both measured:

- **Terms are scanned, not looked up.** `extract.normalise_term` finds every
  index surface form that occurs *inside* the phrase, under exactly
  `count_in_text`'s semantics (punctuated names as substrings, `C`/`Go`/`R`
  case-sensitively, everything else on a word boundary), longest first, with a
  longer match swallowing the shorter ones inside its span. "Azure Kubernetes
  Services (AKS)" answers `Kubernetes`; "Postgres/PostgreSQL tuning" answers
  `PostgreSQL` once. A phrase no form appears in is kept verbatim — that is a
  gap term, and the one thing this must never do is quietly delete a
  requirement the index cannot cover. On the *existing* cached extractions, with
  no re-extraction at all, this lifts the on-vocabulary rate from 19% to 32%.
- **The prompt shows the model the vocabulary.** `index_vocabulary` renders one
  line per skill (`Kubernetes (K8s, Container Orchestration, kubectl)`) — 59
  lines, ~3.4 kB, under a 7000-character cap so a larger index drops its tail
  instead of the posting — and asks for `terms` in those words, with the
  posting's own spelling moved to a new optional `surface` list that nothing
  matches against. Re-extracting cpf, hitachi and mediatek took the
  on-vocabulary rate from 33% to 48%, and on cpf every term that *could* resolve
  now does: what is left (`Azure`, `Terraform`, `Infrastructure as Code`) is not
  in the index, which is the honest answer.

Two decisions inside that:

- **Normalisation happens at cache *read* time as well as at parse time.** The
  rows in the store were written by whatever prompt was current when they were
  made. Bumping `ATS_PROMPT_VERSION` to `extract-2` re-extracts them, but the
  scanning step is post-processing, not a prompt, and a code change to it should
  not need fourteen model calls to take effect. So `extract_requirements`
  re-normalises what it reads back. The same call applies the archetype guard,
  so a stored `"Intern"` does not outlive the version that accepted it.
- **The archetype is checked against the category list.** Two of the fourteen
  extractions answered `Intern` and one `Student` where the prompt asked for one
  of the candidate's skill categories, and `_reconcile` only backfills an
  archetype that is *absent* — so an invented one won, and picked the summary
  variant. `valid_archetype` returns `None` for anything not on the list, which
  puts the deterministic count back in charge. All fourteen archetypes are now
  valid; cpf went from `Intern` to `Cloud & Infrastructure`.

**What remains, and it is not the vocabulary.** The fix worked and the means
barely moved (must 74.0% → 74.0%, context 46.4% → 46.9% against the previous
live run). The reason is visible in the requirement counts: 244 requirements
across the corpus with `none`, **322 with `ollama`**. The model splits a posting
into more requirements than the parser does, and — this is the defect —
**92 of its 193 requirements name no term at all, and every one of them carries
weight 1.0**: "Able to explain technical ideas in a simple way", "Currently
pursuing a Degree in …", "Works well independently and with others". That is
48% of the model's total requirement weight spent on lines every claim answers
about equally well. The deterministic parser has always weighted exactly those
at `ATS_SOFT_REQ_WEIGHT` (0.25) — `requirements_from_jd` does it, and
`merge_fallback` does it for the safety net, where 50 term-less requirements
carry 12.5 of 52.0 weight. The extraction path is the only one that does not,
and the selection it produces is diluted accordingly: `must_coverage` holds
because the terms still reach the page, but `context_rate` falls because the
bullets chosen are answering boilerplate. Applying the soft-weight rule to
extracted requirements whose normalised terms are empty is the obvious next
measurement — and it is only *now* a safe rule to apply, because before this
phase "empty terms" meant "the model spelled it the posting's way", not "the
requirement names nothing".
