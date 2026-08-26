# Phase 3a — Matching core (`ats-match`) and the v2 selection pipeline

Reference: `docs/v2-design.md` §5 (pipeline, objective), §7 (package
boundary). Phase 3 is split: **3a** (this) builds the pure matching core and
a v2 pipeline that runs end-to-end with a *deterministic* requirement parser;
**3b** adds the SQLite runtime store and overrides; **3c** adds LLM/agent
requirement extraction, gates, providers, and alias-aligned rendering.

Exit criterion for 3a: `ats-tailor eval --engine v2` runs on all labelled
JDs and its means for `must_coverage`, `context_rate`, and `parse_ok` are
≥ the v1 baseline (`eval/<hash>.json` from Phase 2) with `pages` = 1.0.
The v1 engine stays the default until that holds.

## 1. Workspace split

- Root `pyproject.toml` stays the `ats-tailor` app (so the parent wrapper's
  `uv run --project ats-tailor` keeps working) and becomes a uv workspace
  root: `[tool.uv.workspace] members = ["packages/*"]`,
  `[tool.uv.sources] ats-match = { workspace = true }`, and `ats-match` in
  `dependencies`.
- `packages/ats-match/pyproject.toml`: name `ats-match`, deps `numpy` only
  (BM25 implemented in-package; no sentence-transformers — models are
  injected). `ats_match/` package with `py.typed`.
- Boundary test in `tests/`: importing `ats_match` never imports
  `ats_tailor` (check `sys.modules` after a fresh subprocess import), and a
  grep test that no file under `packages/ats-match` mentions `resume`,
  `project`, `role`, `latex`, or `yaml`.

## 2. `ats_match` — pure library

`types.py`
```python
@dataclass(frozen=True) class Requirement:
    id: str; text: str; kind: Literal["must", "nice", "responsibility"]
    weight: float = 1.0; terms: tuple[str, ...] = (); category: str | None = None
@dataclass(frozen=True) class Claim:
    id: str; text: str; group: str; section: str
    cost: float = 1.0; terms: tuple[str, ...] = (); prior: float = 0.0
    attrs: Mapping[str, Any] = field(default_factory=dict)
@dataclass class SectionRule:   # per section, e.g. "role", "project", "skill"
    min_groups: int; max_groups: int; min_claims: int; max_claims: int
    group_cost: float = 0.0      # header lines when a group is opened
    required_attr: str | None = None   # e.g. "impact": ≥1 claim with attrs[required_attr] truthy when available
@dataclass class Constraints:
    budget: float; sections: dict[str, SectionRule]
    pin_claims: set[str] = ...; pin_groups: set[str] = ...
    exclude_claims: set[str] = ...; exclude_groups: set[str] = ...
@dataclass class Params:
    tau: float = 0.1         # softpool temperature (log-sum-exp)
    mu: float = 0.3          # redundancy penalty weight
    beta: float = 0.2        # keyword-in-context bonus weight
    kind_weights = {"must": 1.0, "responsibility": 0.7, "nice": 0.5}
```

`lexical.py` — `BM25(docs: list[list[str]])` with `score(query_tokens) ->
np.ndarray`; a `tokenize(text, aliases: Mapping[str, str]) -> list[str]`
that lowercases, splits on non-alphanumerics *except* inside known multi-word
or symbol terms (`C++`, `C#`, `.NET`, `CI/CD`, `Node.js`), and maps every
alias to its canonical token before scoring (so "K8s" and "Kubernetes" are
the same token).

`providers.py` — `class Embedder(Protocol): def encode(self, texts:
Sequence[str]) -> np.ndarray` (row-normalised; may carry `query_prefix` /
`passage_prefix` attributes for asymmetric models), `class
Reranker(Protocol): def score(self, pairs: Sequence[tuple[str, str]]) ->
Sequence[float]`.

`fusion.py` — `rrf(rank_lists, k=60)`; `build_matrix(requirements, claims,
embedder, alias_map, reranker=None, top_k=5, weights=(1.0, 1.0, 1.5)) ->
np.ndarray` of shape `[R, C]` in `[0, 1]`: per requirement, dense cosine
ranks (query = requirement text with `query_prefix`, passages = claim text
with `passage_prefix`) and BM25 ranks are fused with RRF; if a reranker is
given, the top-`k` claims per requirement are rescored and their fused
score replaced by a rank-fused triple. Also returns the dense claim×claim
similarity for redundancy. Both matrices are cached by the caller.

`select.py` — `select(matrix, requirements, claims, constraints, params,
claim_sim=None) -> Selection`:
- objective per design §5: `Σ_r w_r · softpool_τ(M[r, S]) + β · ctx(S) − μ ·
  redundancy(S)`, where `w_r = weight × kind_weights[kind]`,
  `ctx(S)` = share of must-requirement terms that appear in the terms of a
  selected `skill`-section claim **and** of a selected non-skill claim, and
  `redundancy` = Σ over selected pairs within the same section of
  `max(0, sim − 0.8)`.
- lazy greedy over marginal gain with a priority queue; a claim's effective
  cost includes its section's `group_cost` when it would open a new group;
  respects budget, `max_groups`/`max_claims`, excludes; pins are seeded
  first (and their groups opened) before greedy starts.
- repair pass: every opened group reaches `min_claims` (best remaining
  claims of that group by marginal gain), every section reaches
  `min_groups` (open the best-gain group), `required_attr` satisfied when
  the group has such a claim; if the repair overshoots the budget, drop
  the lowest-gain non-pinned claim/group until it fits, then re-repair.
- `Selection{claim_ids, group_ids_by_section, coverage: dict[req_id,
  float], objective, trace: list[Step{claim_id, gain, opened_group,
  covered: list[(req_id, delta)]}], budget_used}`.
- Deterministic: ties broken by `(−prior, claim.id)`.

`evaluate.py` — pure metrics: `coverage_at_budget`, `precision_at_k`,
`kendall_tau` (move the two from `ats_tailor.evaluate`; it imports them).

Tests (`packages/ats-match/tests/`): BM25 alias folding; RRF ordering;
greedy respects every constraint on a synthetic 3-section problem; pins and
excludes; repair pass; determinism; submodular sanity (adding a claim never
lowers coverage); redundancy penalty prefers a diverse pair.

## 3. `ats_tailor` — adapter and v2 pipeline

`ats_tailor/adapt.py`
- `claims_from_index(index: Index, chars_per_line) -> list[Claim]`: one
  claim per bullet of every non-hidden role and project; `group` = item id,
  `section` = `role` | `project`, `cost` = `ceil(len(text) /
  chars_per_line)`, `terms` = surface forms of `bullet.skills ∪ item.skills`
  ∪ `item.keywords`, `prior` = `0.05 × recency_norm` (`recency_multiplier`
  rescaled to `[0, 1]`) + `0.02` if `kind == impact`, `attrs` = `{kind,
  item_kind, impact: kind == "impact"}`.
- `skill_claims(index) -> list[Claim]`: one per skill, `group` = category
  name, `section` = `skill`, `cost` = 0 (the line cost is the group cost),
  `terms` = surface forms, `prior` by proficiency (advanced 0.03,
  intermediate 0.01, familiar 0 — and familiar claims are excluded unless a
  requirement names one of their forms).
- `alias_map(index) -> dict[str, str]` every surface form → canonical
  skill id, for `lexical.tokenize`.
- `section_rules(config) -> dict[str, SectionRule]` from the existing
  `ATS_MIN/MAX_*` settings; `role`/`project` `group_cost` = 2 lines
  (header + spacing; project = 3 with the Technologies line); `skill`
  `max_groups` = `ATS_MAX_SKILL_LINES`, `max_claims` = 8 per group.
- `render_context(selection, index) -> (exp_ctx, proj_ctx, skill_lines)` in
  the shapes `render_resume` already consumes: bullets are the selected
  claims of each group **in the item's original bullet order**; items in
  chronological order as today; skill lines ordered by group gain, skills
  within a line by marginal gain.

`ats_tailor/jd/requirements.py` (deterministic parser — the fallback 3c
keeps)
- `requirements_from_jd(jd_text, term_forms) -> list[Requirement]`: chunk
  with `scoring.chunk_text`; classify chunks with
  `atscheck.classify_jd_sections` → `kind` (must/nice/responsibility; chunks
  outside any recognised section are `responsibility` with weight 0.5);
  drop chunks with < 4 words or no letters; `terms` = every index surface
  form found in the chunk (via `count_in_text`) plus `detect_coverage_gaps`
  terms found in it; drop chunks with zero terms **and** no requirement verb
  (`experience|knowledge|familiar|proficien|understand|ability|able to|
  skills?|degree|build|design|develop|deploy|maintain|support|work`).
  Merge chunks whose lowercase text is identical. Ids `req-<n>`.

`ats_tailor/pipeline.py`
- `tailor_v2(jd_text, index, profile, company, role, out_dir, options) ->
  RunResult` (same `RunResult` as v1, plus `selection`, `requirements`,
  `coverage`):
  1. requirements ← `requirements_from_jd`; claims ← adapter; embedder ←
     `SentenceTransformer` wrapped to satisfy `Embedder` (model from
     config; prefixes for bge/e5 when the model name contains them);
     matrices cached under `index/.emb_cache/` keyed by (model, claim
     hash) and (model, requirement hash) as today.
  2. select with budget `ATS_MAX_PAGE_LINES − fixed_lines` where
     `fixed_lines` = header + summary + education + certs estimate (reuse
     `estimate_line_count` pieces).
  3. render → `compile_pdf` → if `pages > 1`: budget −= 4, re-select,
     re-render (≤ 3 iterations; keep the last one-page result; if none
     fits, fall back to the smallest and report). Certifications: keep v1
     `select_certifications`.
  4. write `resume.tex`, PDF (`<First>_<Last>_Resume_<Company>.pdf`),
     `selection.json` (as v1 plus `requirements`, `trace`, `coverage`),
     `match_report.md` v2: per-requirement table (kind, weight, best
     claim, score, covered?), selection trace, section/group summary,
     gaps (requirements with coverage < 0.3), skills line rationale; keep
     `match_viz.png` (coverage bar per requirement, coloured by kind).
- `tailor.run` gains `--engine {v1,v2}` (env `ATS_ENGINE`, default `v1`
  for now); `--check` works for both. `eval` gains `--engine` and records
  it in the config fingerprint.

## 4. Tests
- Adapter: claim counts equal visible bullets; hidden roles absent; terms
  include alias forms; costs computed; render_context preserves bullet
  order and chronological items.
- Requirement parser on `examples/jd/sample-backend.txt` and a synthetic
  CPF-style JD (What we are looking for / Nice to have) → kinds and terms.
- Pipeline smoke (`ATS_SMOKE=1`): v2 on `examples/` compiles to one page,
  passes `parse_check`, and `selection.json` has a non-empty trace.
- `eval --engine v2 --only mediatek cpf` (smoke) produces a table.

## 5. Docs
- README: engine selection, v2 report contents, the `ats-match` package.
- `docs/v2-design.md` §8: mark 3a delivered; record the v1→v2 eval delta.

## Out of scope
LLM/agent extraction, gates, providers (3c); SQLite, overrides, run store
(3b); alias-aligned surface forms in rendering and summary variants (3c);
TUI/MCP (Phase 4). Do not change the v1 scoring path.
