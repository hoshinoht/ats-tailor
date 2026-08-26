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

Schema (pydantic v2, `ats_tailor/models.py`; JSON Schema emitted to
`schema/` for editor validation):

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
- `overrides(run_id, json)` — pins, exclusions, edited text, reweighted reqs
- `labels(jd_hash, json)` — eval ground truth, derived from overrides + manual

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

packages/ats-tailor/     resume application — depends on ats-match
  ats_tailor/
    index/         models, IO, migration, validation
    jd/            JD store, extraction (LLM/agent/fallback), gates
    adapt.py       Index → Claims (bullets/evidence → Claim, group = item id, cost = est. lines)
                   Selection → render context
    render/        LaTeX template, alias-aligned surface forms, compile loop
    atscheck.py    parse check + keyword table
    tracker.py     applications, statuses, follow-ups
    service.py     one in-process typed API used by every front-end
    cli/  tui/  mcp/  http/
```

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

**LLM provider interface** (`ats_tailor/llm/provider.py`): `extract(jd)`,
`rerank(requirement, claims)`, `propose_variant(bullet, archetype)`.
Providers: `ollama`, `lmstudio`, `mlx`, `none` (deterministic fallbacks),
`external` (results are supplied through the service — used by the MCP so
the driving agent is the LLM). No step in the pipeline calls a model
directly.

**TUI (Textual).** Screens: Applications (runs + status), Tailor (three
panes: requirements coloured by coverage · candidates/bullets with keyboard
pin/exclude/reorder · preview = `pdftotext -layout` text, i.e. what an ATS
sees; `o` opens the PDF; inline raster via `textual-image` where the
terminal supports it), Index (tree + form, live validation, round-trip
save), Eval (table + compare). `textual serve` exposes the same app in a
browser — no separate web front-end.

**MCP (OpenCode/Claude Code compatible).** Same conventions as
`gofetch-mcp` / `researcher-mcp`: stdio, snake_case tools, structured
`error{code,message}` (`invalid_input`, `not_found`, `validation_failed`,
`compile_failed`), `ats_healthcheck`, `max_chars`/`offset` pagination,
silent at idle. Tools:
`ats_healthcheck`, `ats_list_jds`, `ats_add_jd`, `ats_extract_requirements`,
`ats_set_requirements`, `ats_tailor`, `ats_get_run`, `ats_override`,
`ats_compile`, `ats_index_get`, `ats_index_put`, `ats_propose_variant`,
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

**Container.** Python image with Tectonic and poppler-utils; `docker
compose run --rm -it app ats-tailor tui` for the TUI, `ats-tailor serve`
(textual serve) as the exposed port; mounts `index/`, `jd/`, `output/`,
`labels/`, `data/`, HF and Tectonic caches; `ATS_LLM_BACKEND=ollama`,
`ATS_OLLAMA_URL=http://host.docker.internal:11434`. LLM stays on the host
(no GPU in Docker on macOS). MLX is not used in the container.

## 8. Phases

| Phase | Deliverable | Depends on |
|---|---|---|
| 1 | Schema, per-entity layout, migration, compat loader, `pyproject`/`uv`, CLI (`validate`, `migrate`), tests | — |
| 2 | `check` (parse + keyword coverage on PDF), labels format, `eval` baseline on v1 pipeline | 1 |
| 3a | **Delivered.** `ats-match` core (types, BM25, RRF, greedy selection, metrics) in a uv workspace; adapter index→claims; deterministic requirement parser; v2 pipeline with compile loop behind `--engine v2`; eval parity with v1 | 1, 2 |
| 3b | SQLite runtime store (`entities`, `embeddings`, `jd_profiles`, `runs`, `overrides`, `labels`), `sync`, per-run `overrides.yaml` honoured by selection (pins/excludes/edited text), run records | 3a |
| 3c | LLM provider interface (`ollama|lmstudio|mlx|none|external`), requirement extraction + gates, alias-aligned surface forms in rendering, summary variants + target title, v2 becomes the default engine | 3b |
| 4 | `service.py`, LLM provider interface, Textual TUI, MCP server, Dockerfile/Compose | 3 |
| 5 | Variant proposals + summary polish with metric guardrail (agent via MCP or local model), approval in TUI | 4 |

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
