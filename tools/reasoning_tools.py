"""Reasoning tools — explicit reasoning surface for the agent.

These tools let the agent *explicitly* express its reasoning to the
framework, even when the implicit protocol (ReasoningProtocol.plan())
is disabled or unavailable. They are also the user-visible surface:
users can see the agent's plans, critiques, and decisions in the TUI
or messaging gateway.

Why expose reasoning as tools (instead of only running it implicitly)?
--------------------------------------------------------------------
1. **Transparency.** A user reading the conversation can see *why* the
   agent chose to act. This builds trust.
2. **Override hook.** The agent can choose to call `reasoning_decide`
   with `decision=ask_user` when it's genuinely unsure, instead of
   silently failing or recklessly proceeding.
3. **Persistent record.** Plans/decisions invoked through tools are
   logged in the conversation history and can be replayed/audited.
4. **Works with any LLM.** Models that don't natively emit structured
   reasoning can be prompted to call these tools explicitly.

The tools are intentionally low-overhead: they just format and persist
the reasoning artifact. They do NOT call the LLM (the conversation loop
already does that implicitly via ReasoningProtocol).
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent.unified.integration import current_scope, get_bus
from tools.registry import registry


# --------------------------------------------------------------------------- #
# Tool handlers
# --------------------------------------------------------------------------- #


def reasoning_plan_tool(args: dict[str, Any], **_: Any) -> str:
    """Persist an explicit reasoning plan. The agent calls this when it
    wants to record its thinking before a consequential action."""
    tool = str(args.get("tool", "")).strip()
    goal = str(args.get("goal", "")).strip()
    approach = str(args.get("approach", "")).strip()
    risks = str(args.get("risks", "")).strip() or "(none noted)"
    reversibility = str(args.get("reversibility", "")).strip() or "(unknown)"
    decision = str(args.get("decision", "proceed")).strip().lower()
    if decision not in {"proceed", "abort", "ask_user"}:
        decision = "proceed"

    plan = {
        "tool": tool,
        "goal": goal,
        "approach": approach,
        "risks": risks,
        "reversibility": reversibility,
        "decision": decision,
        "scope": current_scope(),
        "recorded_at": time.time(),
    }

    try:
        get_bus().emit(
            "reasoning.plan",
            {"tool": tool, "decision": decision, "goal_preview": goal[:200]},
        )
    except Exception:
        pass

    return json.dumps(
        {
            "status": "recorded",
            "plan": plan,
            "note": (
                "Plan recorded. Proceed with the planned tool call."
                if decision == "proceed"
                else "Plan recorded. Awaiting user input before proceeding."
                if decision == "ask_user"
                else "Plan recorded. Action aborted by agent's own decision."
            ),
        },
        ensure_ascii=False,
    )


def reasoning_critique_tool(args: dict[str, Any], **_: Any) -> str:
    """Persist a self-critique of a plan. Used by the agent to record
    that it has identified blind spots before proceeding."""
    plan_summary = str(args.get("plan_summary", "")).strip()
    blind_spots = str(args.get("blind_spots", "")).strip() or "(none identified)"
    recommendation = str(args.get("recommendation", "proceed")).strip().lower()
    if recommendation not in {"proceed", "modify", "abort"}:
        recommendation = "proceed"
    modified_approach = str(args.get("modified_approach", "")).strip()

    critique = {
        "plan_summary": plan_summary,
        "blind_spots": blind_spots,
        "recommendation": recommendation,
        "modified_approach": modified_approach,
        "recorded_at": time.time(),
    }

    try:
        get_bus().emit(
            "reasoning.critique",
            {"recommendation": recommendation, "blind_spots_preview": blind_spots[:200]},
        )
    except Exception:
        pass

    return json.dumps(
        {"status": "recorded", "critique": critique},
        ensure_ascii=False,
    )


def reasoning_decide_tool(args: dict[str, Any], **_: Any) -> str:
    """Make a decision with explicit reasoning. This is the agent's
    "I have thought about this and I choose to X" surface. It exists
    so that consequential decisions are always auditable."""
    action = str(args.get("action", "")).strip()
    decision = str(args.get("decision", "proceed")).strip().lower()
    if decision not in {"proceed", "abort", "ask_user"}:
        decision = "proceed"
    rationale = str(args.get("rationale", "")).strip()
    acknowledged_irreversible = bool(args.get("acknowledged_irreversible", False))

    record = {
        "action": action,
        "decision": decision,
        "rationale": rationale,
        "acknowledged_irreversible": acknowledged_irreversible,
        "scope": current_scope(),
        "recorded_at": time.time(),
    }

    try:
        get_bus().emit(
            "reasoning.decide",
            {
                "action": action,
                "decision": decision,
                "acknowledged_irreversible": acknowledged_irreversible,
            },
        )
    except Exception:
        pass

    return json.dumps(
        {
            "status": "recorded",
            "decision": record,
            "note": (
                "Decision recorded. Proceeding."
                if decision == "proceed"
                else "Decision recorded. Action aborted by agent."
                if decision == "abort"
                else "Decision recorded. Awaiting user confirmation."
            ),
        },
        ensure_ascii=False,
    )


def reasoning_reflect_tool(args: dict[str, Any], **_: Any) -> str:
    """Record a post-action reflection. The agent calls this after a
    consequential action to record whether the outcome matched its
    expectations and what it learned."""
    tool = str(args.get("tool", "")).strip()
    outcome_matched = bool(args.get("outcome_matched", True))
    expected = str(args.get("expected", "")).strip()
    actual = str(args.get("actual", "")).strip()
    lesson = str(args.get("lesson", "")).strip()

    # Persist the lesson into the reflexion store so future similar
    # situations can recall it. This bridges explicit reasoning tools
    # and the implicit reflexion layer.
    if lesson:
        try:
            from agent.unified.integration import get_store
            from agent.unified.reflexion import ReflexionRecord

            record = ReflexionRecord(
                lesson=lesson,
                source="explicit_reflection",
                score=2.5,
                tags=["reflection", "explicit"],
                tool_name=tool,
                scope=current_scope(),
            )
            get_store().add(record)
        except Exception:
            pass

    reflection = {
        "tool": tool,
        "outcome_matched": outcome_matched,
        "expected": expected,
        "actual": actual,
        "lesson": lesson,
        "recorded_at": time.time(),
    }

    try:
        get_bus().emit(
            "reasoning.reflect",
            {"tool": tool, "outcome_matched": outcome_matched, "lesson_preview": lesson[:200]},
        )
    except Exception:
        pass

    return json.dumps(
        {"status": "recorded", "reflection": reflection},
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------- #
# Tool schemas and registration
# --------------------------------------------------------------------------- #

_REASONING_PLAN_SCHEMA = {
    "name": "reasoning_plan",
    "description": (
        "Record an explicit reasoning plan before a consequential action. "
        "Use this BEFORE calling a tool whose classification is CONSEQUENTIAL "
        "or IRREVERSIBLE (e.g., rm -rf, git push --force, DROP TABLE, send "
        "email to many recipients). The plan is persisted for audit and "
        "future recall. This tool does NOT execute the action — it only "
        "records your reasoning. After recording, proceed with the actual "
        "tool call."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "description": "Name of the tool you plan to call next."},
            "goal": {"type": "string", "description": "What this action is supposed to achieve."},
            "approach": {"type": "string", "description": "How this tool call achieves the goal."},
            "risks": {"type": "string", "description": "What could go wrong. Be specific."},
            "reversibility": {"type": "string", "description": "Can this be undone? How?"},
            "decision": {
                "type": "string",
                "enum": ["proceed", "abort", "ask_user"],
                "description": "Your decision. 'ask_user' if you need human input before proceeding.",
            },
        },
        "required": ["tool", "goal", "approach", "decision"],
    },
}

_REASONING_CRITIQUE_SCHEMA = {
    "name": "reasoning_critique",
    "description": (
        "Record a self-critique of your own plan. Use this after "
        "reasoning_plan for IRREVERSIBLE actions to verify you haven't "
        "missed anything. Be skeptical of your own assumptions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plan_summary": {"type": "string", "description": "Brief summary of the plan being critiqued."},
            "blind_spots": {"type": "string", "description": "What the plan missed."},
            "recommendation": {
                "type": "string",
                "enum": ["proceed", "modify", "abort"],
                "description": "Your critique recommendation.",
            },
            "modified_approach": {
                "type": "string",
                "description": "If recommendation=modify, describe the change. Empty otherwise.",
            },
        },
        "required": ["plan_summary", "blind_spots", "recommendation"],
    },
}

_REASONING_DECIDE_SCHEMA = {
    "name": "reasoning_decide",
    "description": (
        "Make an explicit, auditable decision. Use this when you are about "
        "to do something the user might question — even if you're confident "
        "it's correct. The decision is recorded permanently so the user "
        "can review it later. This is your 'I have thought about this and "
        "I choose to X' surface."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "A short description of what you're about to do."},
            "decision": {
                "type": "string",
                "enum": ["proceed", "abort", "ask_user"],
                "description": "Your decision.",
            },
            "rationale": {"type": "string", "description": "One or two sentences explaining why."},
            "acknowledged_irreversible": {
                "type": "boolean",
                "description": "Set to true if you acknowledge this action cannot be undone.",
            },
        },
        "required": ["action", "decision", "rationale"],
    },
}

_REASONING_REFLECT_SCHEMA = {
    "name": "reasoning_reflect",
    "description": (
        "Record a reflection after a consequential action. Compare the "
        "expected outcome to what actually happened, and extract a "
        "transferable lesson. The lesson is stored in the reflexion "
        "memory and will be recalled in similar future situations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "description": "Name of the tool that was just executed."},
            "outcome_matched": {"type": "boolean", "description": "Did the actual outcome match your expectation?"},
            "expected": {"type": "string", "description": "What you expected to happen."},
            "actual": {"type": "string", "description": "What actually happened."},
            "lesson": {"type": "string", "description": "The transferable lesson, or empty if nothing notable."},
        },
        "required": ["tool", "outcome_matched", "expected", "actual"],
    },
}


registry.register(
    name="reasoning_plan",
    toolset="unified",
    schema=_REASONING_PLAN_SCHEMA,
    handler=reasoning_plan_tool,
    description="Record an explicit reasoning plan",
    emoji="🧭",
)
registry.register(
    name="reasoning_critique",
    toolset="unified",
    schema=_REASONING_CRITIQUE_SCHEMA,
    handler=reasoning_critique_tool,
    description="Record a self-critique of a plan",
    emoji="🔍",
)
registry.register(
    name="reasoning_decide",
    toolset="unified",
    schema=_REASONING_DECIDE_SCHEMA,
    handler=reasoning_decide_tool,
    description="Make an explicit, auditable decision",
    emoji="⚖️",
)
registry.register(
    name="reasoning_reflect",
    toolset="unified",
    schema=_REASONING_REFLECT_SCHEMA,
    handler=reasoning_reflect_tool,
    description="Record a post-action reflection",
    emoji="🪞",
)
