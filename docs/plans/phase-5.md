# Phase 5 — Guarded phrasing variants and summary polish

Reference: `docs/v2-design.md` §2 (honesty rules), §3 (`Bullet.variants`,
`metrics`), §7 (`propose_variant`, MCP), §5 (summary variants). Builds on
4a (`Service`, MCP) and 4b (TUI approval screen).

Principle: a model or agent may *propose* wording; only a human accepts it,
and nothing reaches a PDF that the metric guardrail has not passed. The
index remains the only source of facts.

## 1. Guardrail — `ats_tailor/guardrail.py`
`check_variant(source: Bullet, item, proposed: str) -> GuardrailReport`:
- every entry in `source.metrics` must appear in `proposed` (exact or the
  same number with the same unit, whitespace-insensitive; "15%" ≠ "15 %"
  is tolerated, "15%" ≠ "20%" is not);
- every technology/skill surface form present in the source text must be
  present in the proposal, and the proposal may not introduce a technology
  term that is neither in the source text nor in the item's
  `tech_stack`/`skills`/`keywords` (checked against the index alias map +
  `gap_terms` detection on the proposal);
- no new numbers: any number in the proposal not in the source (or the
  item's metrics) fails;
- length ≤ 180 chars, ends without a period (house style of the index),
  no first person;
- returns `{ok, violations: [{rule, detail}], diff}`.
Unit tests for each rule, both directions.

## 2. Proposals
- `Provider.propose_variant(bullet, archetype, requirements) -> str | None`
  implemented for ollama/lmstudio/mlx with a prompt that carries the
  guardrail rules verbatim, the source bullet, its metrics, the item's
  tech stack, the target archetype, and the top-3 requirements the bullet
  is being aligned to; `external` returns whatever the agent submits.
- `Service.propose_variant(bullet_id, archetype, text=None, requirement_ids=None) -> Proposal{bullet_id, archetype, text, guardrail, source}`:
  with `text` given (agent/MCP) it only runs the guardrail; without, it
  asks the provider. Proposals are stored in the store (`proposals`
  table: id, bullet_id, archetype, text, guardrail_json, status
  `pending|accepted|rejected`, provenance, created_at) — never in the
  index until accepted.
- `Service.accept_proposal(id)` writes `variants[archetype] = text` into
  the bullet's YAML via round-trip IO with a provenance comment, syncs the
  store, marks accepted; `reject_proposal(id, reason=None)`.
- Summary polish: `Service.propose_summary(slug)` builds a summary from
  the archetype variant/default + target title and asks the provider to
  tighten it under the same guardrail (no new claims: every noun phrase
  must appear in the index summaries, highlights, or selected claims);
  accepted text is written to `summaries.variants[archetype]`.

## 3. Selection uses variants
`adapt.claims_from_index` emits, for a bullet with a variant matching the
run's archetype, the variant text as the claim text (the bullet id is
unchanged; `attrs.variant = archetype`); the render context prints the
same text. A run's trace shows which variants were used. `overrides`
`bullet_text` still wins over a variant.

## 4. Front-ends
- MCP: `ats_propose_variant(bullet_id, archetype, text?)`,
  `ats_accept_proposal(id)`, `ats_reject_proposal(id, reason?)`,
  `ats_proposals(status?)`, `ats_propose_summary(slug, text?)`.
- TUI (Tailor screen): `v` on a bullet opens a side panel with the
  proposal, the guardrail verdict with violations highlighted in the
  text, `a` accept / `d` reject / `e` edit-then-recheck.
- CLI: `ats-tailor propose <bullet_id> --archetype sre [--text …]`,
  `ats-tailor proposals list|accept|reject`.

## 5. Eval
`eval` runs with variants on (default) and `--no-variants`; a variant that
lowers must/context on its archetype's JDs is reported per bullet in a
`variants` section of the eval JSON. Guardrail pass-rate over all stored
proposals is printed.

## 6. Tests
Guardrail rules; proposal lifecycle in the store; accept writes YAML with
the comment and syncs; claims pick up the variant for the matching
archetype only; MCP tool shapes; TUI pilot for accept/reject. Smoke:
`external` proposal through MCP → accept → run → the variant text appears
in `resume.tex` and the guardrail report is in the run record.

## Out of scope
Cover letters; rewriting bullets without a stored, accepted variant;
learning weights from outcomes.
