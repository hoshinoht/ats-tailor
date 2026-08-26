"""The content index: what a resume is made of, on disk and in memory.

- :mod:`.models` — the pydantic v2 schema every layout is loaded into.
- :mod:`.io` — read and write the v2 directory layout, and JSON Schema for it.
- :mod:`.loaders` — the v1 loader, ``validate_index``, date helpers, embedding cache.
- :mod:`.migrate` — v1 → v2 conversion.
"""
