"""Compatibility shim — the code moved to :mod:`ats_tailor.engine.legacy.tailor`.

``python -m ats_tailor.tailor …`` and ``ats_tailor.tailor:main`` are documented
entry points (the repo's ``./tailor`` wrapper falls back to the first one when
``uv`` is missing), so the old module path keeps working. Prefer
``ats-tailor tailor`` or :mod:`ats_tailor.service` for new callers.
"""

from .engine.legacy.tailor import (  # noqa: F401
    load_index_for_terms,
    main,
    provider_choice,
    run,
)

__all__ = ["load_index_for_terms", "main", "provider_choice", "run"]


if __name__ == "__main__":
    main()
