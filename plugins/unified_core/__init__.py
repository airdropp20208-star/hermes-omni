"""Unified Core compatibility plugin for Hermes.

The unified layer is now tightly wired into Hermes' core dispatcher and built-in
tool registry. This plugin remains as an opt-in extension point for extra
middleware/tracing and backward-compatible helper imports.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.unified.integration import get_bus, on_session_start as _core_session_start, recall_context
from agent.unified.tracing import span


def on_session_start(**kwargs: Any) -> None:
    _core_session_start(
        session_id=kwargs.get("session_id", "") or "",
        profile=kwargs.get("profile_name", "") or "",
    )


def tool_execution(
    next_call: Callable[[dict[str, Any]], Any],
    tool_name: str,
    args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Optional AgentScope-style execution wrapper with OpenTelemetry spans."""
    with span(
        "hermes.tool",
        tool_name=tool_name,
        session_id=kwargs.get("session_id", ""),
        turn_id=kwargs.get("turn_id", ""),
    ):
        return next_call(args or {})


def transform_llm_output(text: str | None = None, **kwargs: Any) -> None:
    # Kept as an extension seam for downstream unified plugins. The core
    # integration records/recalls reflexions through tools rather than mutating
    # final user-visible output.
    return None


def unified_bus():
    """Return the process-local unified event bus."""
    return get_bus()


def unified_reflexion_context(query: str, *, limit: int = 5) -> str:
    """Public helper for context engines/subagents to retrieve lessons."""
    return recall_context(query, limit=limit)


def register(ctx: Any) -> None:
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("transform_llm_output", transform_llm_output)
    ctx.register_middleware("tool_execution", tool_execution)
