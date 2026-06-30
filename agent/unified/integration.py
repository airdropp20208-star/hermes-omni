"""Tight Hermes integration for unified agent primitives.

This module is called from Hermes' core tool dispatcher and exposes vendored
OmniAgent/AgentScope packages through a stable Hermes-native integration layer.

v1 extension: this module now also orchestrates the Reasoning Protocol
(plan → critique → execute → reflect) and the Smart Guardian (LLM-as-judge).
Both are opt-in via config; when disabled, behavior is identical to the
legacy integration (pattern-based guardian + reflexion-on-failure only).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .config import load_unified_config
from .decision import Classification, DecisionClass, DecisionFramework
from .events import EventBus
from .longrun import (
    LongRunEngine,
    WorkItemPriority,
    configure_engine,
    get_engine,
    shutdown_engine,
)
from .policy import Decision, PolicyEngine
from .reasoning import ReasoningPlan, ReasoningProtocol, configure_protocol, get_protocol
from .reflexion import ReflexionRecord, ReflexionStore, record_from_tool_failure
from .smart_guardian import (
    GuardianVerdict,
    RiskAssessment,
    SmartGuardian,
    configure_guardian,
    get_guardian,
)
from .tool_router import ToolRouter, get_router, refresh_router

_bus = EventBus()
_policy: PolicyEngine | None = None
_store: ReflexionStore | None = None
_decider = DecisionFramework()
# Per-tool-call state, keyed by tool_call_id, so after_tool_call can find
# the plan made by before_tool_call. Bounded; cleared on session start.
_pending_plans: dict[str, ReasoningPlan] = {}
_pending_classifications: dict[str, Classification] = {}
_pending_verdicts: dict[str, RiskAssessment] = {}


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
    """Run unified guardian checks before Hermes executes a tool.

    Pipeline (in order, fail-open at every step):
        1. Pattern-based PolicyEngine (legacy). Blocks catastrophic patterns
           like rm -rf /, fork bomb, mkfs, etc. Zero LLM cost.
        2. DecisionFramework classifies the action (TRIVIAL/STANDARD/
           CONSEQUENTIAL/IRREVERSIBLE).
        3. If reasoning_enabled AND classification requires a plan,
           ReasoningProtocol.plan() generates one. Persisted for
           after_tool_call to use.
        4. If smart_guardian_enabled AND classification is CONSEQUENTIAL+,
           SmartGuardian.assess() runs the LLM judge. Verdict may BLOCK
           or REQUIRE_USER_CONFIRM.
        5. If hard_block_irreversible AND classification is IRREVERSIBLE,
           BLOCK unconditionally (hard floor).
    """

    cfg = load_unified_config()
    if not cfg.enabled:
        return None

    # ----- Step 1: Pattern-based PolicyEngine (legacy) -----
    if cfg.guardian_enabled:
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
            pass  # fail open

    # ----- Step 2: Classification -----
    try:
        classification = _decider.classify(tool_name, args)
    except Exception:
        classification = Classification(
            cls=DecisionClass.STANDARD, reason="classifier failed"
        )
    if tool_call_id:
        _pending_classifications[tool_call_id] = classification
        _prune_pending()

    # Emit classification event for observability.
    try:
        _bus.emit(
            "tool.classify",
            {
                "tool_name": tool_name,
                "classification": classification.cls.name,
                "reason": classification.reason,
                "tool_call_id": tool_call_id,
            },
            session_id=session_id or "",
            turn_id=turn_id or "",
        )
    except Exception:
        pass

    # ----- Step 3: Reasoning Protocol (plan + critique) -----
    plan: ReasoningPlan | None = None
    if cfg.reasoning_enabled and classification.requires_plan:
        try:
            plan = get_protocol().plan(
                tool_name=tool_name,
                args=args,
                classification=classification,
            )
            if plan is not None and tool_call_id:
                _pending_plans[tool_call_id] = plan
                _prune_pending()
            # Emit the plan so TUI/gateway can display it.
            if plan is not None:
                _bus.emit(
                    "reasoning.plan",
                    {
                        "tool_name": tool_name,
                        "decision": plan.decision,
                        "goal_preview": plan.goal[:200],
                        "tool_call_id": tool_call_id,
                    },
                    session_id=session_id or "",
                    turn_id=turn_id or "",
                )
            # If the plan itself says "abort", respect that.
            if plan is not None and plan.decision == "abort":
                return (
                    f"Reasoning protocol aborted the action: {plan.rationale or 'no rationale given'}. "
                    f"Plan goal: {plan.goal}"
                )
            # If the plan says "ask_user", surface that — the conversation
            # loop is responsible for actually pausing; we just return a
            # message that the LLM will see and can choose to surface.
            if plan is not None and plan.decision == "ask_user":
                return (
                    f"Reasoning protocol recommends asking the user before proceeding: "
                    f"{plan.rationale or plan.goal}"
                )
        except Exception:
            plan = None  # fail open

    # ----- Step 4: Smart Guardian (LLM judge) -----
    if cfg.smart_guardian_enabled and classification.requires_critique:
        try:
            verdict = get_guardian().assess(
                tool_name=tool_name,
                args=args,
                plan=plan,
                classification=classification,
            )
            if tool_call_id:
                _pending_verdicts[tool_call_id] = verdict
                _prune_pending()
            _bus.emit(
                "guardian.verdict",
                {
                    "tool_name": tool_name,
                    "verdict": verdict.verdict.value,
                    "risk_level": verdict.risk_level,
                    "reasoning_preview": verdict.reasoning[:200],
                    "cache_hit": verdict.cache_hit,
                    "tool_call_id": tool_call_id,
                },
                session_id=session_id or "",
                turn_id=turn_id or "",
            )
            if verdict.verdict is GuardianVerdict.BLOCK:
                return (
                    f"Smart Guardian blocked this action: {verdict.reasoning}. "
                    f"Risk level: {verdict.risk_level}."
                )
            if verdict.verdict is GuardianVerdict.REQUIRE_USER_CONFIRM:
                return (
                    f"Smart Guardian requires user confirmation: {verdict.reasoning}. "
                    f"Risk level: {verdict.risk_level}."
                )
        except Exception:
            pass  # fail open

    # ----- Step 5: Hard floor for IRREVERSIBLE actions -----
    if cfg.hard_block_irreversible and classification.requires_acknowledgement:
        # The hard floor: even if the LLM judge allowed it, IRREVERSIBLE
        # actions must go through reasoning_decide tool (which the LLM
        # can choose to call) or be explicitly acknowledged. We don't
        # BLOCK outright — we return a message that surfaces the
        # irreversibility and lets the LLM decide to either call
        # reasoning_decide or pick a safer alternative.
        return (
            f"Action classified as IRREVERSIBLE ({classification.reason}). "
            "Hard-block floor is active. To proceed, call the `reasoning_decide` "
            "tool with `acknowledged_irreversible=true` and a rationale, OR "
            "choose a safer alternative. This protection is intentional — "
            "set `unified.smart_guardian.hard_block_irreversible: false` to "
            "disable it (Codex-style autonomy)."
        )

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
    """Record tool events, extract reflexion lessons, and run reflection."""

    cfg = load_unified_config()
    if not cfg.enabled:
        return

    # Recover per-call state saved by before_tool_call.
    plan = _pending_plans.pop(tool_call_id, None) if tool_call_id else None
    classification = (
        _pending_classifications.pop(tool_call_id, None) if tool_call_id else None
    )
    verdict = _pending_verdicts.pop(tool_call_id, None) if tool_call_id else None

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
                "classification": classification.cls.name if classification else None,
                "verdict": verdict.verdict.value if verdict else None,
            },
            session_id=session_id or "",
            turn_id=turn_id or "",
        )
    except Exception:
        pass

    # Legacy: extract reflexion record from tool failure (always runs
    # when reflexion is enabled, regardless of reasoning protocol).
    if cfg.reflexion_enabled:
        try:
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
            pass

    # New: reasoning reflection. Only when reasoning_enabled AND a plan
    # was made for this call.
    if cfg.reasoning_enabled and cfg.persist_reflections and plan is not None:
        # If longrun is enabled, push reflection to the background worker
        # (debounced + batched). Otherwise run synchronously.
        engine = get_engine() if cfg.longrun_enabled else None
        if engine is not None:
            # Background path — fire-and-forget.
            try:
                engine.enqueue_reflection(
                    {
                        "plan": plan,
                        "tool_name": tool_name,
                        "args": args,
                        "result": result,
                        "duration_ms": duration_ms,
                        "session_id": session_id or "",
                        "turn_id": turn_id or "",
                        "scope": current_scope(),
                    }
                )
            except Exception:
                pass  # fail-open; reflection is best-effort
        else:
            # Synchronous path (v1 behavior).
            try:
                reflection = get_protocol().reflect(
                    plan=plan,
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    duration_ms=duration_ms,
                )
                if reflection is not None and reflection.lesson:
                    from .reflexion import ReflexionRecord

                    refl_record = ReflexionRecord(
                        lesson=reflection.lesson,
                        source="reasoning_reflection",
                        score=reflection.score,
                        tags=["reflection", "reasoning"]
                        + (["outcome_mismatch"] if not reflection.outcome_matched else []),
                        session_id=session_id or "",
                        turn_id=turn_id or "",
                        tool_name=tool_name,
                        scope=current_scope(),
                    )
                    if get_store().add(refl_record):
                        _bus.emit(
                            "reflexion.add",
                            {
                                "tool_name": tool_name,
                                "lesson": reflection.lesson[:300],
                                "record_id": refl_record.record_id,
                                "outcome_matched": reflection.outcome_matched,
                            },
                            session_id=session_id or "",
                            turn_id=turn_id or "",
                        )
            except Exception:
                pass

    # Tool router usage feedback (async, non-blocking).
    if cfg.tool_router_enabled and cfg.tool_router_learn:
        try:
            # Use the tool_name + brief args as the "query" for usage
            # tracking. This is a heuristic; the conversation loop can
            # also call record_usage explicitly with the user's actual
            # message for better signal.
            args_str = json.dumps(args or {}, ensure_ascii=False, default=str)[:200]
            get_router().record_usage(tool_name, f"{tool_name} {args_str}")
        except Exception:
            pass


def _prune_pending(max_size: int = 256) -> None:
    """Bound the per-call state dicts so they can't grow unbounded.

    Each entry is small (a few hundred bytes), so 256 entries ≈ 100KB
    worst case. We evict oldest by insertion order (dict preserves it).
    """
    for table in (_pending_plans, _pending_classifications, _pending_verdicts):
        while len(table) > max_size:
            # pop first inserted key
            try:
                first_key = next(iter(table))
                table.pop(first_key, None)
            except StopIteration:
                break


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
        # v1 extension status
        "reasoning_enabled": cfg.reasoning_enabled,
        "smart_guardian_enabled": cfg.smart_guardian_enabled,
        "hard_block_irreversible": cfg.hard_block_irreversible,
        "persist_reflections": cfg.persist_reflections,
        "reasoning_protocol_wired": get_protocol()._llm_call is not None,
        "smart_guardian_wired": get_guardian()._llm_call is not None,
        # v1.1 extensions
        "longrun_enabled": cfg.longrun_enabled,
        "longrun_status": longrun_status(),
        "tool_router_enabled": cfg.tool_router_enabled,
        "tool_router_top_n": cfg.tool_router_top_n,
        "tool_router_learn": cfg.tool_router_learn,
        "tool_router_stats": (
            get_router().stats() if cfg.tool_router_enabled else None
        ),
        "scope": current_scope(),
        "store": str(cfg.store_path),
        "vendored": {
            "omniagent": _probe("omniagent"),
            "agentscope": _probe("agentscope"),
        },
    }


def status_tool(args: dict[str, Any], **_: Any) -> str:
    return json.dumps(framework_status(), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Public configuration API
# --------------------------------------------------------------------------- #
# These functions are called by Hermes' setup path once the LLM client is
# available, so that the reasoning protocol and smart guardian can use
# real LLM calls instead of the no-LLM fallback.
#
# Typical wiring (in hermes_cli/setup.py or agent/conversation_loop.py):
#
#     from agent.unified.integration import configure_reasoning_stack
#
#     def llm_call(system_prompt, user_prompt):
#         return my_llm_client.chat(system=system_prompt, user=user_prompt)
#
#     configure_reasoning_stack(llm_call=llm_call)


def configure_reasoning_stack(
    *,
    llm_call=None,
    conversation_context_provider=None,
) -> None:
    """Wire the LLM client into the reasoning protocol and smart guardian.

    Idempotent. Safe to call multiple times (e.g., once at startup, again
    when the user changes models via `hermes model`).

    Args:
        llm_call: callable(system_prompt: str, user_prompt: str) -> str.
            Must return the raw model output as a string. Should NOT
            raise on transient errors — the protocol handles that.
        conversation_context_provider: callable() -> str. Returns a brief
            summary of recent conversation context for the planning prompt.
            Should be cheap (cached/summarized); the protocol calls it
            on every consequential action.
    """
    if llm_call is None:
        return  # nothing to wire

    cfg = load_unified_config()
    configure_protocol(
        llm_call=llm_call,
        conversation_context_provider=conversation_context_provider,
    )
    configure_guardian(
        llm_call=llm_call,
        reflexion_recall=recall_context,
        cache_size=cfg.guardian_cache_size,
        cache_ttl_seconds=cfg.guardian_cache_ttl_seconds,
    )

    # If longrun is enabled, wire the reflection batch function and start
    # the engine.
    if cfg.longrun_enabled:
        engine = configure_engine(
            heartbeat_seconds=cfg.longrun_heartbeat_seconds,
            reflection_debounce_seconds=cfg.longrun_reflection_debounce_seconds,
            reflection_batch_size=cfg.longrun_reflection_batch_size,
            autostart=True,
        )
        # Wire the batch reflection function.
        from .reasoning import get_protocol

        engine.set_reflect_batch_fn(get_protocol().reflect_batch)


# --------------------------------------------------------------------------- #
# Tool router public API
# --------------------------------------------------------------------------- #


def suggest_tools_for_query(query: str, *, top_n: int | None = None) -> str:
    """Return a markdown block of suggested tools for `query`.

    Returns "" if tool router is disabled or no tools match. The
    conversation loop should prepend this to the system prompt before
    calling the LLM.
    """
    cfg = load_unified_config()
    if not cfg.tool_router_enabled:
        return ""
    try:
        router = get_router()
        n = top_n if top_n is not None else cfg.tool_router_top_n
        return router.format_suggestions(query, top_n=n)
    except Exception:
        return ""


def record_tool_usage_for_message(tool_name: str, user_message: str) -> None:
    """Record that `tool_name` was used in the context of `user_message`.

    More accurate signal than the heuristic in after_tool_call (which
    uses args). The conversation loop should call this when it knows
    the user message that triggered a tool call.
    """
    cfg = load_unified_config()
    if not cfg.tool_router_enabled or not cfg.tool_router_learn:
        return
    try:
        get_router().record_usage(tool_name, user_message)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Long-run engine public API
# --------------------------------------------------------------------------- #


def longrun_status() -> dict[str, Any]:
    """Return long-run engine stats, or {"enabled": false} if not running."""
    cfg = load_unified_config()
    if not cfg.longrun_enabled:
        return {"enabled": False}
    engine = get_engine()
    if engine is None:
        return {"enabled": True, "running": False}
    return {"enabled": True, "running": True, **engine.stats()}


def enqueue_longrun_work(
    *,
    kind: str,
    payload: dict[str, Any],
    priority: WorkItemPriority = WorkItemPriority.NORMAL,
    item_id: str = "",
) -> str | None:
    """Enqueue a work item to the long-run engine.

    Returns the item_id, or None if the engine is not running. The
    caller must register a handler for `kind` via `engine.register_handler`
    before enqueuing (otherwise the item is dropped with a warning).
    """
    cfg = load_unified_config()
    if not cfg.longrun_enabled:
        return None
    engine = get_engine()
    if engine is None:
        return None
    return engine.enqueue(kind=kind, payload=payload, priority=priority, item_id=item_id)


def shutdown_longrun(*, timeout: float = 5.0) -> None:
    """Shut down the long-run engine. Call on agent exit."""
    shutdown_engine(timeout=timeout)
