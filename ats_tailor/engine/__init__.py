"""Selecting content for one posting.

- :mod:`.pipeline` — the v2 run: requirements in, a rendered resume out.
- :mod:`.adapt` — ``Index`` → ``ats_match`` claims, and back into a render context.
- :mod:`.embedding` — the sentence-transformer embedder and its store cache.
- :mod:`.rerank` — the optional cross-encoder pass.
- :mod:`.results` — ``RunResult``, what either engine hands back.
- :mod:`.legacy` — the v1 whole-item ranking, kept behind ``--engine v1``.
"""
