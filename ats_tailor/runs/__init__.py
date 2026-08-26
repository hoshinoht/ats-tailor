"""Runtime state that outlives a single run.

- :mod:`.store` — the SQLite store (index entities, embeddings, runs, JD profiles).
- :mod:`.sync` — bring the store in line with the index; optionally pre-embed.
- :mod:`.tracker` — applications, statuses, follow-ups.
- :mod:`.overrides` — the per-run ``overrides.yaml`` next to an output directory.
"""
