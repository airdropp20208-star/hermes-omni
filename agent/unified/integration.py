"""Tight Hermes integration for unified agent primitives.

This module is called from Hermes' core tool dispatcher and exposes vendored
OmniAgent/AgentScope packages through a stable Hermes-native integration layer.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .config import load_unified_config
from .events import EventBus
from .policy import Decision, PolicyEngine
from .reflexion import ReflexionRecord, ReflexionStore, record_from_tool_failure

_bus = EventBus()
_policy: PolicyEngine | None = None
_store: ReflexionStore | None = None


def enabled() -> bool:
    return load_unified_config().enabled


def get_bus() -> EventBus:
    return _bus


def get_policy() -> PolicyEngine:
    global _policy
    cfg = load_unified_config()
    if _policy is None:
        _policy = PolicyEngine.from_patterns(cfg.block_tools)
    return _policy


def get_store() -> ReflexionStore:
    global _store
    cfg = load_unified_config()
    if _store is None:
        _store = ReflexionStore(cfg.store_path, max_records=cfg.max_records)
    return _store


def current_scope() -> str:
    cfg = load_unified_config()
    if not cfg.scope_by_cwd:
        return "global"
    try:
        return str(Path.cwd().resolve())
    except Exception:
        return "global"


def on_session_start(*, session_id: str = "", profile: str = "", **_: Any) -> None:
    if not enabled():
        return
    _bus.emit("session.start", {"profile": profile}, session_id=session_id or "")


def before_tool_call(
    *,
    tool_name: str,
    args: dict[str, Any] | None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
) -> str | None:
    """Run unified guardian checks before Hermes executes a tool."""

    cfg = load_unified_config()
    if not cfg.enabled or not cfg.guardian_enabled:
        return None
    try:
        decision = get_policy().evaluate(tool_name, args or {})
        _bus.emit(
            "tool.policy",
            {
                "tool_name": tool_name,
                "decision": decision.decision.value,
                "rule": decision.rule,
                "task_id": task_id,
                "tool_call_id": tool_call_id,
                "api_request_id": api_request_id,
            },
            session_id=session_id or "",
            turn_id=turn_id or "",
        )
        if decision.decision == Decision.BLOCK:
            return decision.message
    except Exception:
        return None
    return None


def after_tool_call(
    *,
    tool_name: str,
    args: dict[str, Any] | None,
    result: Any,
    duration_ms: int = 0,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
) -> None:
    """Record tool events and extract reflexion lessons after execution."""

    cfg = load_unified_config()
    if not cfg.enabled:
        return
    try:
        _bus.emit(
            "tool.finish",
            {
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "result_preview": str(result)[:300],
                "task_id": task_id,
                "tool_call_id": tool_call_id,
                "api_request_id": api_request_id,
            },
            session_id=session_id or "",
            turn_id=turn_id or "",
        )
        if not cfg.reflexion_enabled:
            return
        record = record_from_tool_failure(
            tool_name=tool_name,
            args=args,
            result=result,
            session_id=session_id or "",
            turn_id=turn_id or "",
            scope=current_scope(),
        )
        if record is not None and get_store().add(record):
            _bus.emit(
                "reflexion.add",
                {"tool_name": tool_name, "lesson": record.lesson[:300], "record_id": record.record_id},
                session_id=record.session_id,
                turn_id=record.turn_id,
            )
    except Exception:
        return


def recall_context(query: str, *, limit: int = 5, scope: str | None = None) -> str:
    cfg = load_unified_config()
    if not cfg.enabled or not cfg.reflexion_enabled:
        return ""
    return get_store().format_context(query, limit=limit, scope=scope or current_scope())


def recall_records(query: str, *, limit: int = 5, scope: str | None = None) -> list[ReflexionRecord]:
    cfg = load_unified_config()
    if not cfg.enabled or not cfg.reflexion_enabled:
        return []
    return get_store().recall(query, limit=limit, scope=scope or current_scope())


def recall_tool(args: dict[str, Any], **_: Any) -> str:
    """Tool handler exposed through Hermes' built-in registry."""

    query = str(args.get("query", "")).strip()
    limit = int(args.get("limit", 5) or 5)
    scope = args.get("scope") or current_scope()
    context = recall_context(query, limit=max(1, min(limit, 10)), scope=str(scope))
    return json.dumps({"context": context, "empty": not bool(context), "scope": scope}, ensure_ascii=False)


def list_tool(args: dict[str, Any], **_: Any) -> str:
    limit = int(args.get("limit", 20) or 20)
    scope = args.get("scope")
    records = get_store().list(scope=str(scope) if scope else None)[-max(1, min(limit, 100)) :]
    return json.dumps(
        {
            "count": len(records),
            "records": [
                {
                    "record_id": r.record_id,
                    "tool_name": r.tool_name,
                    "source": r.source,
                    "scope": r.scope,
                    "tags": r.tags,
                    "created_at": r.created_at,
                    "lesson": r.lesson,
                }
                for r in records
            ],
        },
        ensure_ascii=False,
    )


def clear_tool(args: dict[str, Any], **_: Any) -> str:
    scope = args.get("scope")
    confirm = bool(args.get("confirm", False))
    if not confirm:
        return json.dumps({"error": "Set confirm=true to clear unified reflexion memory."}, ensure_ascii=False)
    removed = get_store().clear(scope=str(scope) if scope else None)
    return json.dumps({"removed": removed, "scope": scope or "all"}, ensure_ascii=False)


def framework_status() -> dict[str, Any]:
    def _probe(name: str) -> dict[str, Any]:
        try:
            mod = importlib.import_module(name)
            return {
                "available": True,
                "module": getattr(mod, "__file__", ""),
                "version": getattr(mod, "__version__", "unknown"),
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    cfg = load_unified_config()
    return {
        "unified_enabled": cfg.enabled,
        "guardian_enabled": cfg.guardian_enabled,
        "reflexion_enabled": cfg.reflexion_enabled,
        "auto_prefetch_enabled": cfg.auto_prefetch_enabled,
        "scope": current_scope(),
        "store": str(cfg.store_path),
        "vendored": {
            "omniagent": _probe("omniagent"),
            "agentscope": _probe("agentscope"),
        },
    }


def status_tool(args: dict[str, Any], **_: Any) -> str:
    return json.dumps(framework_status(), ensure_ascii=False)
