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
# v2 cognitive extensions
from .cognitive_tree import CognitiveTree, configure_tree, get_tree
from .hypothesis import HypothesisEngine, configure_hypothesis_engine, get_hypothesis_engine
from .context_distiller import ContextDistiller, configure_distiller, get_distiller
from .metacognitive import MetacognitiveMonitor, SelfDoubtSignal, configure_monitor, get_monitor
from .causal_graph import (
    CausalGraph,
    clear_graph,
    get_graph as get_causal_graph,
    get_or_create_graph,
    list_graphs,
    set_active_task,
)
# v2.1 learning + memory + skill synthesis
from .learning import LearningEngine, configure_learning_engine, get_learning_engine
from .skill_synthesizer import (
    SkillSynthesizer,
    configure_synthesizer,
    get_synthesizer,
    record_tool_call_for_synthesis,
)
# v2.2 task planning
from .task_planner import TaskPlanner, configure_planner, get_planner
# v2.3 output formatting
from .output_formatter import (
    OutputFormatter,
    configure_formatter,
    format_output_for_platform,
    get_formatter,
)

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
        # v2: CognitiveTree for CONSEQUENTIAL+ actions (if enabled).
        # Generates N branches, prunes, picks best. Replaces single-plan path.
        if cfg.cognitive_tree_enabled and classification.requires_critique:
            try:
                tree = get_tree()
                if tree is not None:
                    ctx = ""
                    try:
                        from .integration import _conversation_context_provider
                        ctx = _conversation_context_provider() if _conversation_context_provider else ""
                    except Exception:
                        ctx = ""
                    result_tree = tree.evaluate(
                        tool_name=tool_name,
                        args=args,
                        classification=classification,
                        conversation_context=ctx,
                    )
                    if result_tree is not None:
                        plan = result_tree.selected_plan
                        if plan is not None and tool_call_id:
                            _pending_plans[tool_call_id] = plan
                            _prune_pending()
                        _bus.emit(
                            "cognitive_tree.result",
                            {
                                "tool_name": tool_name,
                                "branches_count": len(result_tree.branches),
                                "confidence": result_tree.confidence,
                                "decision": result_tree.decision,
                                "llm_calls": result_tree.llm_calls,
                                "tool_call_id": tool_call_id,
                            },
                            session_id=session_id or "",
                            turn_id=turn_id or "",
                        )
                        if result_tree.decision == "abort":
                            return (
                                f"CognitiveTree aborted: {result_tree.rationale}"
                            )
                        if result_tree.decision == "ask_user":
                            return (
                                f"CognitiveTree recommends asking user: {result_tree.rationale}"
                            )
            except Exception:
                pass  # fail open, fall through to v1 plan path

        # v1 fallback: single plan (also runs if cognitive_tree disabled or failed).
        if plan is None:
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
                # If the plan says "ask_user", surface that.
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
            args_str = json.dumps(args or {}, ensure_ascii=False, default=str)[:200]
            get_router().record_usage(tool_name, f"{tool_name} {args_str}")
        except Exception:
            pass

    # v2: Metacognitive monitoring — track outcome, trigger self-doubt.
    if cfg.metacognitive_enabled:
        try:
            monitor = get_monitor()
            if monitor is not None:
                # Determine stated confidence from the plan (if any).
                stated = 0.7  # default if no plan/cognitive tree
                if plan is not None:
                    # Heuristic: v1 plans have no confidence field, use 0.7.
                    stated = 0.7
                # Determine actual outcome from result.
                result_str = str(result) if result is not None else ""
                lowered = result_str.lower()
                is_failure = any(
                    marker in lowered
                    for marker in ("error", "blocked", "permission", "traceback", "failed", "exception")
                )
                actual: str
                if is_failure:
                    actual = "failure"
                elif result_str and "success" in lowered:
                    actual = "success"
                elif result_str:
                    actual = "success"
                else:
                    actual = "partial"
                signal = monitor.record_outcome(
                    stated_confidence=stated,
                    actual_outcome=actual,  # type: ignore[arg-type]
                    tool_name=tool_name,
                )
                if signal is not None:
                    _bus.emit(
                        "metacognitive.self_doubt",
                        {
                            "trigger": signal.trigger,
                            "stated_confidence": signal.stated_confidence,
                            "calibrated_confidence": signal.calibrated_confidence,
                            "recommendation": signal.recommendation,
                            "rationale": signal.rationale,
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                        },
                        session_id=session_id or "",
                        turn_id=turn_id or "",
                    )
        except Exception:
            pass

    # v2.1: Skill synthesis — record tool call for pattern detection.
    if cfg.skill_synthesis_enabled:
        try:
            result_str = str(result) if result is not None else ""
            lowered = result_str.lower()
            is_failure = any(
                marker in lowered
                for marker in ("error", "blocked", "permission", "traceback", "failed", "exception")
            )
            record_tool_call_for_synthesis(
                tool_name=tool_name,
                args=args,
                success=not is_failure,
                session_id=session_id or "",
            )
            # Trigger a scan periodically (the synthesizer self-throttles).
            synth = get_synthesizer()
            if synth is not None:
                new_skills = synth.maybe_scan()
                if new_skills:
                    for s in new_skills:
                        _bus.emit(
                            "skill.synthesized",
                            {
                                "skill_id": s.skill_id,
                                "name": s.name,
                                "description": s.description,
                                "source_pattern": s.source_pattern,
                                "file_path": s.file_path,
                            },
                            session_id=session_id or "",
                            turn_id=turn_id or "",
                        )
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

    # v2: CognitiveTree (branching reasoning).
    if cfg.cognitive_tree_enabled:
        configure_tree(
            llm_call=llm_call,
            reflexion_recall=recall_context,
            n_branches=cfg.cognitive_tree_n_branches,
            min_confidence=cfg.cognitive_tree_min_confidence,
            max_confidence=cfg.cognitive_tree_max_confidence,
        )

    # v2: HypothesisEngine (diagnostic reasoning).
    if cfg.hypothesis_enabled:
        configure_hypothesis_engine(
            llm_call=llm_call,
            n_hypotheses=cfg.hypothesis_n_hypotheses,
            max_iterations=cfg.hypothesis_max_iterations,
            confidence_threshold=cfg.hypothesis_confidence_threshold,
        )

    # v2: ContextDistiller (structured insight extraction).
    if cfg.context_distiller_enabled:
        configure_distiller(
            llm_call=llm_call,
            distill_every_n_turns=cfg.context_distill_every_n_turns,
            max_distilled_items=cfg.context_distiller_max_items,
            merge_threshold=cfg.context_distiller_merge_threshold,
        )

    # v2: MetacognitiveMonitor (always configured, even if no LLM — pure stats).
    if cfg.metacognitive_enabled:
        configure_monitor(
            llm_call=llm_call,
            self_doubt_threshold=cfg.metacognitive_self_doubt_threshold,
            repeated_failure_count=cfg.metacognitive_repeated_failure_count,
            min_samples_for_calibration=cfg.metacognitive_min_samples,
        )

    # v2.1: LearningEngine (extract learnings from every interaction).
    if cfg.learning_enabled:
        configure_learning_engine(
            llm_call=llm_call,
            max_records=cfg.learning_max_records,
            extract_every_n_turns=cfg.learning_extract_every_n_turns,
        )

    # v2.1: SkillSynthesizer (auto-create skills from repeated patterns).
    if cfg.skill_synthesis_enabled:
        configure_synthesizer(
            llm_call=llm_call,
            min_occurrences=cfg.skill_synthesis_min_occurrences,
            max_skills=cfg.skill_synthesis_max_skills,
        )

    # v2.2: TaskPlanner (decompose, track, replan).
    if cfg.task_planner_enabled:
        configure_planner(
            llm_call=llm_call,
            max_subtasks=cfg.task_planner_max_subtasks,
            max_replans=cfg.task_planner_max_replans,
        )

    # v2.3: OutputFormatter (always configure — pure transformation, no LLM).
    if cfg.output_formatter_enabled:
        configure_formatter(
            summarize_long_output=cfg.output_formatter_summarize_long,
            summarize_threshold=cfg.output_formatter_summarize_threshold,
        )


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


# --------------------------------------------------------------------------- #
# v2 cognitive extensions — public API
# --------------------------------------------------------------------------- #


def start_hypothesis_session(*, symptom: str, context: str = "") -> str | None:
    """Start a diagnostic session. Returns session_id, or None if disabled."""
    cfg = load_unified_config()
    if not cfg.hypothesis_enabled:
        return None
    engine = get_hypothesis_engine()
    if engine is None:
        return None
    return engine.start_session(symptom=symptom, context=context)


def record_hypothesis_test(
    *,
    session_id: str,
    hypothesis_id: str,
    test_result: str,
) -> dict[str, Any] | None:
    """Record the result of testing a hypothesis."""
    cfg = load_unified_config()
    if not cfg.hypothesis_enabled:
        return None
    engine = get_hypothesis_engine()
    if engine is None:
        return None
    return engine.record_test_result(
        session_id=session_id,
        hypothesis_id=hypothesis_id,
        test_result=test_result,
    )


def get_distilled_context_block() -> str:
    """Return the current distilled context as a prompt block. Empty if disabled."""
    cfg = load_unified_config()
    if not cfg.context_distiller_enabled:
        return ""
    distiller = get_distiller()
    if distiller is None:
        return ""
    return distiller.get_prompt_block()


def maybe_distill_context(
    *,
    turn_count: int,
    conversation_segment: str,
    turn_start: int,
    turn_end: int,
) -> str | None:
    """Trigger distillation if enough turns have passed. Returns the
    new distilled block (or None if not triggered)."""
    cfg = load_unified_config()
    if not cfg.context_distiller_enabled:
        return None
    distiller = get_distiller()
    if distiller is None:
        return None
    result = distiller.maybe_distill(
        turn_count=turn_count,
        conversation_segment=conversation_segment,
        turn_start=turn_start,
        turn_end=turn_end,
    )
    if result is None:
        return None
    return distiller.get_prompt_block()


def metacognitive_stats() -> dict[str, Any]:
    """Return calibration stats, or {"enabled": false} if disabled."""
    cfg = load_unified_config()
    if not cfg.metacognitive_enabled:
        return {"enabled": False}
    monitor = get_monitor()
    if monitor is None:
        return {"enabled": True, "configured": False}
    return {"enabled": True, "configured": True, **monitor.stats()}


# --- Causal graph public API ---


def causal_graph_add_node(
    *,
    node_id: str,
    label: str,
    node_type: str = "action",
    status: str = "pending",
    evidence: str = "",
    task_id: str | None = None,
) -> bool:
    """Add a node to the causal graph for the active task."""
    cfg = load_unified_config()
    if not cfg.causal_graph_enabled:
        return False
    _, graph = get_or_create_graph(task_id)
    graph.add_node(
        node_id=node_id,
        label=label,
        node_type=node_type,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        evidence=evidence,
    )
    return True


def causal_graph_add_edge(
    *,
    src: str,
    dst: str,
    edge_type: str = "causes",
    strength: float = 1.0,
    rationale: str = "",
    task_id: str | None = None,
) -> bool:
    """Add a causal edge between two nodes."""
    cfg = load_unified_config()
    if not cfg.causal_graph_enabled:
        return False
    _, graph = get_or_create_graph(task_id)
    return graph.add_edge(
        src=src,
        dst=dst,
        edge_type=edge_type,  # type: ignore[arg-type]
        strength=strength,
        rationale=rationale,
    )


def causal_graph_root_causes(failure_node_id: str, task_id: str | None = None) -> list[dict[str, Any]]:
    """Find root causes of a failure node."""
    cfg = load_unified_config()
    if not cfg.causal_graph_enabled:
        return []
    graph = get_causal_graph(task_id)
    if graph is None:
        return []
    return [
        {
            "node_id": n.node_id,
            "label": n.label,
            "node_type": n.node_type,
            "status": n.status,
            "evidence": n.evidence,
        }
        for n in graph.root_causes(failure_node_id)
    ]


def causal_graph_to_mermaid(task_id: str | None = None) -> str:
    """Render the causal graph as a Mermaid diagram."""
    cfg = load_unified_config()
    if not cfg.causal_graph_enabled:
        return ""
    graph = get_causal_graph(task_id)
    if graph is None:
        return ""
    return graph.to_mermaid()


def causal_graph_stats(task_id: str | None = None) -> dict[str, Any] | None:
    """Return stats for the causal graph."""
    cfg = load_unified_config()
    if not cfg.causal_graph_enabled:
        return None
    graph = get_causal_graph(task_id)
    if graph is None:
        return None
    return graph.stats()


# --- Conversation context provider hook ---


_conversation_context_provider = None


def set_conversation_context_provider(provider) -> None:
    """Set a callable that returns recent conversation context (string).
    Used by CognitiveTree for planning prompts."""
    global _conversation_context_provider
    _conversation_context_provider = provider


# --------------------------------------------------------------------------- #
# v2.1 learning + skill synthesis — public API
# --------------------------------------------------------------------------- #


def recall_learnings(query: str, *, limit: int = 5, scope: str | None = None) -> str:
    """Recall relevant learnings as a prompt block. Empty if disabled."""
    cfg = load_unified_config()
    if not cfg.learning_enabled:
        return ""
    engine = get_learning_engine()
    if engine is None:
        return ""
    return engine.format_context(query, limit=limit, scope=scope)


def record_learning(
    *,
    event_type: str,
    content: str,
    importance: float = 2.0,
    context: str = "",
    associated_tools: list[str] | None = None,
    associated_queries: list[str] | None = None,
    scope: str | None = None,
) -> bool:
    """Manually record a learning event."""
    cfg = load_unified_config()
    if not cfg.learning_enabled:
        return False
    engine = get_learning_engine()
    if engine is None:
        return False
    event = engine.record_manual(
        event_type=event_type,
        content=content,
        importance=importance,
        context=context,
        associated_tools=associated_tools,
        associated_queries=associated_queries,
        scope=scope or current_scope(),
    )
    return event is not None


def maybe_extract_learnings(
    *,
    turn_count: int,
    conversation_segment: str,
    scope: str | None = None,
) -> int:
    """Trigger learning extraction if enough turns have passed.
    Returns the number of new learnings extracted."""
    cfg = load_unified_config()
    if not cfg.learning_enabled:
        return 0
    engine = get_learning_engine()
    if engine is None:
        return 0
    events = engine.maybe_extract(
        turn_count=turn_count,
        conversation_segment=conversation_segment,
        scope=scope or current_scope(),
    )
    return len(events)


def learning_stats() -> dict[str, Any]:
    """Return learning store stats."""
    cfg = load_unified_config()
    if not cfg.learning_enabled:
        return {"enabled": False}
    engine = get_learning_engine()
    if engine is None:
        return {"enabled": True, "configured": False}
    return {"enabled": True, "configured": True, **engine.stats()}


def list_synthesized_skills() -> list[dict[str, Any]]:
    """List auto-synthesized skills."""
    cfg = load_unified_config()
    if not cfg.skill_synthesis_enabled:
        return []
    synth = get_synthesizer()
    if synth is None:
        return []
    return synth.list_skills()


def skill_synthesis_stats() -> dict[str, Any]:
    """Return skill synthesis stats."""
    cfg = load_unified_config()
    if not cfg.skill_synthesis_enabled:
        return {"enabled": False}
    synth = get_synthesizer()
    if synth is None:
        return {"enabled": True, "configured": False}
    return {"enabled": True, "configured": True, **synth.stats()}


def trigger_skill_scan(*, force: bool = False) -> list[dict[str, Any]]:
    """Force a skill synthesis scan. Returns newly synthesized skills."""
    cfg = load_unified_config()
    if not cfg.skill_synthesis_enabled:
        return []
    synth = get_synthesizer()
    if synth is None:
        return []
    new_skills = synth.maybe_scan(force=force)
    return [
        {
            "skill_id": s.skill_id,
            "name": s.name,
            "description": s.description,
            "source_pattern": s.source_pattern,
            "file_path": s.file_path,
        }
        for s in new_skills
    ]


# --------------------------------------------------------------------------- #
# v2.2 task planning — public API
# --------------------------------------------------------------------------- #


def create_task_plan(*, request: str, context: str = "") -> str | None:
    """Decompose a complex request into a plan. Returns plan_id, or None."""
    cfg = load_unified_config()
    if not cfg.task_planner_enabled:
        return None
    planner = get_planner()
    if planner is None:
        return None
    plan = planner.create_plan(request=request, context=context)
    if plan is None:
        return None
    _bus.emit(
        "task.plan_created",
        {"plan_id": plan.plan_id, "subtask_count": len(plan.subtasks)},
    )
    return plan.plan_id


def get_task_plan_progress() -> str:
    """Return the active plan progress as a prompt block. Empty if no plan."""
    cfg = load_unified_config()
    if not cfg.task_planner_enabled:
        return ""
    planner = get_planner()
    if planner is None:
        return ""
    return planner.get_progress_block()


def start_task_subtask(subtask_id: str) -> bool:
    cfg = load_unified_config()
    if not cfg.task_planner_enabled:
        return False
    planner = get_planner()
    if planner is None:
        return False
    return planner.start_subtask(subtask_id)


def complete_task_subtask(subtask_id: str, *, result_summary: str = "") -> bool:
    cfg = load_unified_config()
    if not cfg.task_planner_enabled:
        return False
    planner = get_planner()
    if planner is None:
        return False
    ok = planner.complete_subtask(subtask_id, result_summary=result_summary)
    if ok:
        _bus.emit("task.subtask_done", {"subtask_id": subtask_id})
    return ok


def fail_task_subtask(subtask_id: str, *, failure_reason: str = "") -> bool:
    cfg = load_unified_config()
    if not cfg.task_planner_enabled:
        return False
    planner = get_planner()
    if planner is None:
        return False
    ok = planner.fail_subtask(subtask_id, failure_reason=failure_reason)
    if ok:
        _bus.emit("task.subtask_failed", {"subtask_id": subtask_id, "reason": failure_reason})
    return ok


def skip_task_subtask(subtask_id: str, *, reason: str = "") -> bool:
    cfg = load_unified_config()
    if not cfg.task_planner_enabled:
        return False
    planner = get_planner()
    if planner is None:
        return False
    return planner.skip_subtask(subtask_id, reason=reason)


def abandon_task_plan(*, reason: str = "") -> bool:
    cfg = load_unified_config()
    if not cfg.task_planner_enabled:
        return False
    planner = get_planner()
    if planner is None:
        return False
    return planner.abandon_plan(reason=reason)


def list_task_plans() -> list[dict[str, Any]]:
    cfg = load_unified_config()
    if not cfg.task_planner_enabled:
        return []
    planner = get_planner()
    if planner is None:
        return []
    return planner.list_plans()


def task_planner_stats() -> dict[str, Any]:
    cfg = load_unified_config()
    if not cfg.task_planner_enabled:
        return {"enabled": False}
    planner = get_planner()
    if planner is None:
        return {"enabled": True, "configured": False}
    return {"enabled": True, "configured": True, **planner.stats()}


# --------------------------------------------------------------------------- #
# v2.3 output formatting — public API
# --------------------------------------------------------------------------- #


def format_for_delivery(
    text: str,
    *,
    platform: str = "default",
    max_length: int | None = None,
) -> list[dict[str, Any]]:
    """Format agent output for delivery to a messaging platform.

    This is the main entry point for gateway/delivery.py to call before
    sending a message. Returns a list of chunk dicts, each ready to be
    sent as one message.

    Args:
        text: the raw agent output
        platform: "telegram" | "discord" | "slack" | "whatsapp" | "signal" |
                  "matrix" | "teams" | "email" | "sms" | "cli" | "default"
        max_length: override platform default limit

    Returns:
        list of {"text": str, "part": int, "total_parts": int, "is_last": bool}

    Platforms handled:
        telegram: MarkdownV2 escaping, JSON→readable, table→plain text,
                  control char stripping, chunking at 4096 chars
        discord:  JSON→readable, chunking at 2000 chars
        slack:    **bold**→*bold* (mrkdwn), chunking at 40000 chars
        cli:      pass-through (no transformation)
        default:  JSON→readable, table→plain text, control char stripping
    """
    cfg = load_unified_config()
    if not cfg.output_formatter_enabled:
        # Disabled — return single chunk with raw text.
        return [{"text": text, "part": 1, "total_parts": 1, "is_last": True}]
    return format_output_for_platform(text, platform=platform, max_length=max_length)


def format_for_telegram(text: str) -> list[dict[str, Any]]:
    """Convenience wrapper for Telegram formatting."""
    return format_for_delivery(text, platform="telegram")


def format_for_slack(text: str) -> list[dict[str, Any]]:
    """Convenience wrapper for Slack formatting."""
    return format_for_delivery(text, platform="slack")


def format_for_discord(text: str) -> list[dict[str, Any]]:
    """Convenience wrapper for Discord formatting."""
    return format_for_delivery(text, platform="discord")
