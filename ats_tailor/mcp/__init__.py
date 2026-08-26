"""The MCP face of :mod:`ats_tailor.service`.

An agent runtime (OpenCode, Claude Code) speaks to this repo over stdio. The
tools here are thin: they validate arguments, call one :class:`~ats_tailor.service.Service`
method, and turn the answer into JSON. Every decision about what a resume is
lives in the service; every decision about what an agent is allowed to be told
lives here.
"""

from .server import (
    DEFAULT_TOOLS,
    GATED_TOOLS,
    TOOLS,
    ToolSet,
    build_server,
    run_stdio,
    wrap_posting,
)

__all__ = [
    "DEFAULT_TOOLS",
    "GATED_TOOLS",
    "TOOLS",
    "ToolSet",
    "build_server",
    "run_stdio",
    "wrap_posting",
]
