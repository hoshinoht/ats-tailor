"""The Textual front-end. ``ats-tailor tui``; the same app over HTTP with
``ats-tailor serve``.

Importing this package must not import Textual — the CLI's ``--help`` and the
MCP server both live in the same process tree, and an optional extra that is
not installed has to fail when it is asked for and not when the package is
scanned. :func:`build_app` is where the import happens.
"""

from __future__ import annotations

from pathlib import Path

#: What to say when ``pip install`` has not been told about the extra.
MISSING = (
    "the TUI needs the optional 'tui' extra:\n"
    "  uv sync --extra tui        (inside ats-tailor/)\n"
    "  pip install 'ats-tailor[tui]'"
)


def build_app(
    repo: str | Path = ".",
    *,
    index_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    provider: str | None = None,
    scratch_dir: str | Path | None = None,
):
    """An :class:`~ats_tailor.tui.app.AtsTailorApp` over one repository."""
    from ..service import Service

    try:
        from .app import AtsTailorApp
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SystemExit(f"{MISSING}\n({exc})") from exc

    service = Service(
        Path(repo).expanduser().resolve(),
        index_dir=index_dir, data_dir=data_dir, provider=provider,
    )
    return AtsTailorApp(service, scratch_dir=scratch_dir)


def run(
    repo: str | Path = ".",
    *,
    index_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    provider: str | None = None,
) -> int:
    """Run the TUI in this terminal until the user quits."""
    app = build_app(repo, index_dir=index_dir, data_dir=data_dir, provider=provider)
    try:
        app.run()
    finally:
        app.service.close()
    return 0


__all__ = ["MISSING", "build_app", "run"]
