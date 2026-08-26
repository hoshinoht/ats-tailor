# Phase 4b — Textual TUI and container

Reference: `docs/v2-design.md` §7. Builds on the 4a `Service`; the TUI is
a client of it and never imports pipeline internals directly.

## 1. TUI — `ats_tailor/tui/` (Textual; optional extra `tui = [textual, textual-image]`)
`ats-tailor tui [--repo …]`; `ats-tailor serve [--host] [--port]` runs
`textual serve` on the same app.

Screens (tabs, `1`–`4`):
1. **Applications** — DataTable of runs (id, created, company/role, status
   with ⚠ for gate flags, must/nice/ctx, days quiet). Keys: `enter` open in
   Tailor, `s` set status (select from vocabulary), `n` add note, `q`
   quiet-only filter, `a` add posting (paste text + company/role → `add_jd`
   → `fit`).
2. **Tailor** — three panes for one run: left = gates block + requirements
   list coloured by coverage (green ≥ 0.6, amber ≥ 0.3, red) with weight;
   centre = preview of `pdftotext -layout` text (what an ATS sees) with a
   header line "page N / must X% / ctx Y%"; right = selected items and
   bullets grouped by section with the trace gain. Keys: `p` pin item,
   `x` exclude item, `P`/`X` for the bullet under the cursor, `w` reweight
   requirement (0.25/0.5/1.0/1.5), `r` re-run with overrides, `c` compile +
   check, `o` open the PDF in the system viewer, `i` inline raster of page 1
   via `textual-image` when the terminal supports it, `l` write pins to the
   label (`labels_from_run`). Every action calls the service; a busy
   indicator while it runs; errors shown in a toast with the `ServiceError`
   code.
3. **Index** — Tree (projects / roles / skills by category / certs) → form
   for the selected entity (bullets editable in a TextArea, skills as a
   comma list, metrics shown read-only); `ctrl+s` validates the whole index
   and saves via `index_put`; validation errors inline. An "impact" panel:
   for the selected project/role, its rank across all saved runs' traces
   (from the store) so the user sees what an edit affects.
4. **Eval** — table from the latest `eval/*.json`, `e` run eval (engine /
   provider picker), `d` diff two baselines (`--compare`).

Global: `?` help, `ctrl+p` provider picker (none/auto/ollama/external),
status bar with repo, engine, provider, store path. The app must render
correctly at 100×30 and degrade (hide the right pane) below 90 columns.

Tests: Textual's `App.run_test()` pilot — each screen mounts on the
examples repo, keys `p`/`x` produce the expected `overrides.yaml`, `s`
changes status through the service, the index form save round-trips a
bullet edit. No LaTeX in the default suite (mock `compile`); one
`ATS_SMOKE=1` end-to-end.

## 2. Container
- `Dockerfile`: `python:3.12-slim`; install `poppler-utils`, `curl`, and
  Tectonic (static binary release) — no TeX Live; `uv` copied from the
  official image; `uv sync --no-dev --extra tui` for the workspace; a
  non-root user; `ENTRYPOINT ["ats-tailor"]`, `CMD ["serve","--host","0.0.0.0","--port","8080"]`.
- `compile_pdf` gains `ATS_LATEX_ENGINE=auto|latexmk|tectonic` (auto = latexmk
  if present else tectonic); the parse checks are engine-agnostic.
- `docker-compose.yaml` (new, replaces the deleted dead one): service
  `app` with volumes `./index`, `./jd`, `./labels`, `./output`, `./eval`,
  `./data`, plus named volumes for the HF cache and Tectonic cache; env
  `ATS_LLM_PROVIDER=ollama`, `ATS_OLLAMA_URL=http://host.docker.internal:11434`,
  `ATS_LATEX_ENGINE=tectonic`; port 8080. Compose file lives in the parent
  repo? No — it lives in `ats-tailor/` with paths relative to a
  `REPO=..` env default, and the README shows `docker compose -f
  ats-tailor/docker-compose.yaml …` from the parent.
- Make targets or README commands: build, `docker compose run --rm -it app tui`,
  `docker compose up` for the browser UI, `docker compose run --rm app tailor …`.
- The image must run the examples end-to-end (`tailor --check`) with
  provider `none` and no network beyond the initial model download into the
  HF volume; document `HF_HUB_OFFLINE=1` for air-gapped runs after warm-up.

## 3. Tests / verification
- Unit: `ATS_LATEX_ENGINE` selection; Tectonic invocation builds the same
  command shape (mock subprocess).
- Manual/verified by the implementer: `docker build`, `docker compose run
  --rm app tailor …` on the examples, page count 1, `check` passes; `serve`
  reachable on 8080 from the host; TUI runs interactively in the container.
  Report image size and cold/warm run times.

## 4. Docs
README: TUI keys and screens, `serve`, container usage; design §7/§8;
parent `CLAUDE.md` lines for the coordinator.

## Out of scope
HTTP adapter (only if something remote needs it), Go MCP client, Phase 5.
