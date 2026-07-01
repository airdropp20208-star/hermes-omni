"""Runtime wiring — central hooks to integrate cognitive modules into Hermes.

This module is the SINGLE ENTRY POINT for wiring all 21 cognitive modules
into the Hermes runtime. Instead of scattering modifications across
mega-files (conversation_loop.py 4,843 lines, run_agent.py 5,590 lines),
each mega-file calls ONE function from here. This:

1. Reduces risk — only 1-2 line additions to mega-files
2. Centralizes wiring logic — easy to audit/disable
3. Makes cognitive modules testable in isolation
4. Allows graceful degradation — if any module fails, others still work

Usage from mega-files:

    # In agent/system_prompt.py build_system_prompt_parts() — volatile tier:
    from agent.unified.runtime_wiring import augment_volatile_prompt
    volatile_parts.append(augment_volatile_prompt(agent, user_message))

    # In gateway/delivery.py deliver():
    from agent.unified.runtime_wiring import format_for_delivery
    chunks = format_for_delivery(content, platform=target.platform.value)

    # In hermes_cli/setup.py or run_agent.py — once LLM client is ready:
    from agent.unified.runtime_wiring import wire_llm_client
    wire_llm_client(agent)

    # In model_tools.py after tool call:
    from agent.unified.runtime_wiring import on_tool_call_complete
    on_tool_call_complete(tool_name, args, result, user_message)

    # On agent exit:
    from agent.unified.runtime_wiring import shutdown
    shutdown()
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Track whether we've wired the LLM client (idempotent).
_llm_wired = False
_last_user_message = ""  # for tool usage tracking


def _set_last_user_message(msg: str) -> None:
    """Set the current user message (called from conversation_loop)."""
    global _last_user_message
    _last_user_message = msg or ""
    # Also feed the user model for profile building.
    try:
        from agent.unified.integration import observe_user_message

        observe_user_message(msg or "")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 1. LLM client wiring (call once after agent init)
# --------------------------------------------------------------------------- #


def wire_llm_client(agent: Any) -> bool:
    """Wire the agent's LLM client into all cognitive modules.

    Call this ONCE after the agent has its LLM client configured (after
    `hermes setup` or after `hermes model` changes the model). Idempotent.

    Args:
        agent: the AIAgent instance (must have provider/model/client attrs)

    Returns:
        True if wiring succeeded, False otherwise.
    """
    global _llm_wired
    if _llm_wired:
        return True
    try:
        llm_call = _build_llm_call(agent)
        if llm_call is None:
            logger.debug("wire_llm_client: no LLM client available yet")
            return False

        from agent.unified.integration import configure_reasoning_stack

        configure_reasoning_stack(
            llm_call=llm_call,
            conversation_context_provider=lambda: _get_conversation_context(agent),
        )
        _llm_wired = True
        logger.info(
            "cognitive modules wired (provider=%s, model=%s)",
            getattr(agent, "provider", "?"),
            getattr(agent, "model", "?"),
        )
        return True
    except Exception as exc:
        logger.warning("wire_llm_client failed: %r", exc)
        return False


def _build_llm_call(agent: Any):
    """Build a simple (system, user) -> str callable from the agent.

    The agent's chat completion path is complex (transports, failover,
    caching). For cognitive modules we want a SIMPLE synchronous call
    that returns a string. We use the agent's existing client if
    available, else return None.
    """
    try:
        # Try to extract a callable from the agent's transport layer.
        # The simplest path: use the OpenAI client directly.
        client = getattr(agent, "client", None) or getattr(agent, "_client", None)
        model = getattr(agent, "model", None)
        if client is None or model is None:
            return None

        def llm_call(system_prompt: str, user_prompt: str) -> str:
            try:
                # OpenAI-compatible call. Works for most providers
                # (OpenAI, OpenRouter, Nous, GLM, DeepSeek, etc.)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,  # low temp for structured output
                    max_tokens=4096,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                logger.debug("cognitive llm_call failed: %r", exc)
                return ""

        return llm_call
    except Exception:
        return None


def _get_conversation_context(agent: Any) -> str:
    """Get a brief summary of recent conversation context.

    Used by CognitiveTree and other modules that need context for planning.
    Returns last ~2000 chars of recent messages.
    """
    try:
        history = getattr(agent, "conversation_history", None) or []
        if not history:
            return ""
        # Take last 5 messages, extract text content.
        recent = history[-5:]
        parts = []
        for msg in recent:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part content — extract text parts.
                content = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if isinstance(content, str) and content.strip():
                parts.append(f"[{role}] {content[:400]}")
        return "\n".join(parts)[:2000]
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# 2. System prompt augmentation (call in build_system_prompt_parts volatile tier)
# --------------------------------------------------------------------------- #


def augment_volatile_prompt(agent: Any, user_message: Optional[str] = None) -> str:
    """Inject cognitive module outputs into the system prompt's volatile tier.

    Call this from `agent/system_prompt.py build_system_prompt_parts()` in
    the volatile_parts section. Returns a string to append, or "" if
    nothing to inject.

    Injects (only if enabled and available):
        - Constitution principles
        - User profile (expertise, style, domains)
        - Tool suggestions for current user message
        - Task plan progress (active plan)
        - Distilled context (insights from conversation)
        - Recalled learnings (success/correction/pattern memory)

    Each block is wrapped in XML-like tags for clear separation. All
    blocks are fail-safe (any error → skip that block).
    """
    global _last_user_message
    if user_message:
        _last_user_message = user_message

    blocks: list[str] = []

    # Constitution (always inject if enabled — it's the agent's values).
    try:
        from agent.unified.integration import get_constitution_prompt_block

        block = get_constitution_prompt_block()
        if block:
            blocks.append(block)
    except Exception:
        pass

    # User profile (expertise, style, domains).
    try:
        from agent.unified.integration import get_user_profile_block

        block = get_user_profile_block()
        if block:
            blocks.append(block)
    except Exception:
        pass

    # Tool suggestions for current user message.
    if user_message:
        try:
            from agent.unified.integration import suggest_tools_for_query

            block = suggest_tools_for_query(user_message)
            if block:
                blocks.append(block)
        except Exception:
            pass

    # Task plan progress.
    try:
        from agent.unified.integration import get_task_plan_progress

        block = get_task_plan_progress()
        if block:
            blocks.append(block)
    except Exception:
        pass

    # Distilled context (insights from earlier conversation).
    try:
        from agent.unified.integration import get_distilled_context_block

        block = get_distilled_context_block()
        if block:
            blocks.append(block)
    except Exception:
        pass

    # Recalled learnings.
    if user_message:
        try:
            from agent.unified.integration import recall_learnings

            block = recall_learnings(user_message, limit=5)
            if block:
                blocks.append(block)
        except Exception:
            pass

    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 3. Output formatting (call in gateway/delivery.py before sending)
# --------------------------------------------------------------------------- #


def format_for_delivery(content: str, platform: str = "default"):
    """Format agent output for delivery to a messaging platform.

    Call this from `gateway/delivery.py` before sending to Telegram/Slack/etc.
    Returns a list of chunk dicts (each is one message).

    If OutputFormatter is disabled, returns a single chunk with raw content.
    """
    try:
        from agent.unified.integration import format_for_delivery as _fmt

        return _fmt(content, platform=platform)
    except Exception:
        return [{"text": content, "part": 1, "total_parts": 1, "is_last": True}]


# --------------------------------------------------------------------------- #
# 4. Post-tool-call hooks (call in model_tools.py after tool execution)
# --------------------------------------------------------------------------- #


def on_tool_call_complete(
    tool_name: str,
    args: dict | None,
    result: Any,
    user_message: str = "",
) -> None:
    """Hook to call after a tool completes. Wires tool router usage tracking.

    Call from model_tools.py after `after_tool_call()`. The user_message
    (if available) gives better signal for tool router learning than the
    args-only heuristic in after_tool_call.
    """
    if not user_message:
        user_message = _last_user_message
    if not user_message:
        return
    try:
        from agent.unified.integration import record_tool_usage_for_message

        record_tool_usage_for_message(tool_name, user_message)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 5. Pre-response cognitive pipeline (call before sending response to user)
# --------------------------------------------------------------------------- #


def maybe_run_cognitive_pipeline(
    agent: Any,
    user_message: str,
    response: str,
    *,
    force_thinking_level: str = "",
) -> str:
    """Optionally run the full cognitive pipeline on a response.

    Call this from the conversation loop AFTER the agent generates a
    response but BEFORE sending it to the user. If cognitive modules
    are disabled, returns the response unchanged.

    Pipeline (if enabled):
        1. Verify response (critique → revise)
        2. Check constitution
        3. (Slow thinking and ensemble are NOT run here — they're
           pre-generation, not post-generation. They run in the
           conversation loop before the model call if enabled.)

    Args:
        agent: the AIAgent instance
        user_message: the user's request
        response: the agent's generated response
        force_thinking_level: override config default

    Returns:
        Possibly revised response (or original if disabled/failed).
    """
    try:
        from agent.unified.integration import (
            check_constitution,
            verify_agent_response,
        )
        from agent.unified.config import load_unified_config

        cfg = load_unified_config()
        if not cfg.enabled:
            return response

        current = response

        # Constitution check (pre-verify — if violations, we want to fix
        # before verifying correctness).
        if cfg.constitution_enabled:
            const = check_constitution(
                user_request=user_message,
                response=current,
            )
            if not const.get("aligned", True) and const.get("violations"):
                # Re-generate would be ideal, but we don't have the model
                # here cleanly. Instead, prepend a warning.
                violations = "; ".join(const["violations"][:2])
                current = f"[Note: response flagged for constitution review — {violations}]\n\n{current}"

        # Verifier (critique → revise → re-critique).
        if cfg.verifier_enabled:
            ver = verify_agent_response(
                user_request=user_message,
                agent_response=current,
            )
            revised = ver.get("final_response", "")
            if revised and revised != response:
                current = revised
            if not ver.get("passed", True) and ver.get("warning"):
                # Append warning so user knows verification didn't fully pass.
                current = f"{current}\n\n⚠️ {ver['warning']}"

        return current
    except Exception as exc:
        logger.debug("maybe_run_cognitive_pipeline failed: %r", exc)
        return response


# --------------------------------------------------------------------------- #
# 6. Shutdown (call on agent exit / atexit)
# --------------------------------------------------------------------------- #


def shutdown() -> None:
    """Shut down all cognitive modules gracefully. Call on agent exit.

    Flushes:
        - LongRunEngine (stop worker thread, flush checkpoint)
        - ReflectionWorker (process pending reflections)
        - SkillSynthesizer (persist any in-progress skills)
    """
    try:
        from agent.unified.integration import shutdown_longrun

        shutdown_longrun(timeout=5.0)
        logger.info("cognitive modules shut down")
    except Exception as exc:
        logger.debug("shutdown error: %r", exc)


# Register atexit handler so shutdown happens even on ungraceful exit.
try:
    import atexit

    atexit.register(shutdown)
except Exception:
    pass


# --------------------------------------------------------------------------- #
# 7. Diagnostics
# --------------------------------------------------------------------------- #


def wiring_status() -> dict:
    """Return the current wiring status of all cognitive modules."""
    try:
        from agent.unified.integration import framework_status

        status = framework_status()
        status["llm_wired"] = _llm_wired
        return status
    except Exception as exc:
        return {"error": repr(exc), "llm_wired": _llm_wired}
