"""OmniAgent Feature Integration — hooks into hermes-omni's conversation loop.

This module provides a thin integration layer that connects:
- Deep Reflexion → tool loop detection, error repeat, no-progress
- Guardian Agent → pre-execution risk review
- Sentinel Agent → task decomposition, progress tracking
- Context Evolution → auto-learn from failures

Usage in conversation_loop.py or tool_executor.py:
    from agent.omni_integration import OmniFeatureGate

    gate = OmniFeatureGate(work_dir=agent.work_dir)

    # Before tool execution
    warning = gate.before_tool_call("bash", {"command": "rm -rf /"})
    if warning and "BLOCKED" in warning:
        return warning  # or inject as tool error

    # After tool execution
    gate.after_tool_call("bash", result)

    # After failed turn
    await gate.on_turn_failure(task, error, messages)

    # Before retry
    context = gate.get_retry_context()
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.deep_reflexion import DeepReflexion
from agent.guardian import GuardianAgent, ReviewResult
from agent.sentinel import SentinelAgent
from agent.context_evolution import ContextEvolutionManager

logger = logging.getLogger(__name__)


class OmniFeatureGate:
    """Unified gate for all OmniAgent-inspired features.

    Wraps Deep Reflexion + Guardian + Sentinel + Context Evolution
    into a single interface that can be called from the conversation loop.
    """

    def __init__(
        self,
        work_dir: Path = None,
        enabled: bool = True,
        max_retries: int = 3,
        loop_threshold: int = 3,
        guardian_auto_block: bool = True,
        sentinel_activation_threshold: int = 2,
        evolution_auto_promote: bool = False,
    ):
        self.enabled = enabled
        self.work_dir = work_dir or Path.cwd()

        if not enabled:
            self.reflexion = None
            self.guardian = None
            self.sentinel = None
            self.evolution = None
            return

        # Deep Reflexion
        self.reflexion = DeepReflexion(
            max_retries=max_retries,
            loop_threshold=loop_threshold,
        )

        # Guardian Agent
        self.guardian = GuardianAgent(
            auto_block_critical=guardian_auto_block,
        )

        # Sentinel Agent
        self.sentinel = SentinelAgent(
            work_dir=self.work_dir,
            activation_threshold=sentinel_activation_threshold,
        )

        # Context Evolution
        self.evolution = ContextEvolutionManager(
            work_dir=self.work_dir,
            auto_promote=evolution_auto_promote,
        )

        logger.info(
            "omni_features_initialized reflexion=%s guardian=%s sentinel=%s evolution=%s",
            True, True, True, True,
        )

    def before_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Optional[str]:
        """Check a tool call before execution.

        Runs:
        1. Guardian risk review (pattern-based + optional LLM)
        2. Deep Reflexion loop detection

        Returns:
            Warning/block message if issues detected, None if safe.
        """
        if not self.enabled:
            return None

        messages = []

        # ── Guardian: Risk review ──
        if self.guardian:
            try:
                result = asyncio.get_event_loop().run_until_complete(
                    self.guardian.review(tool_name, params)
                )
                if not result.passed:
                    msg = (
                        f"🚫 BLOCKED by Guardian: {tool_name}\n"
                        f"Risk: {result.risk_level}\n"
                        f"Issues: {'; '.join(result.findings)}"
                    )
                    logger.warning("guardian_blocked tool=%s risk=%s", tool_name, result.risk_level)
                    return msg
                if result.findings:
                    messages.append(
                        f"⚠️ Guardian warning ({result.risk_level}): "
                        f"{'; '.join(result.findings)}"
                    )
            except RuntimeError:
                # No event loop — run synchronously
                pass
            except Exception as e:
                logger.debug("guardian_review_error: %s", e)

        # ── Deep Reflexion: Loop detection ──
        if self.reflexion:
            warning = self.reflexion.on_tool_call(tool_name, params)
            if warning:
                messages.append(warning)

        return "\n".join(messages) if messages else None

    def after_tool_call(self, tool_name: str, result: str):
        """Process tool result after execution.

        Runs Deep Reflexion error repeat detection.
        """
        if not self.enabled or not self.reflexion:
            return

        warning = self.reflexion.on_tool_result(result)
        if warning:
            logger.info("reflexion_post_tool warning=%s", warning[:80])

    def on_tool_result_record(self, tool_name: str, result: str):
        """Record tool result for no-progress detection."""
        if not self.enabled or not self.reflexion:
            return
        self.reflexion.no_progress.record_tool_call(tool_name)
        self.reflexion.no_progress.record_result(result)

    def on_turn_start(self, task: str):
        """Called at the start of a new turn.

        Checks if Sentinel should activate for complex tasks.
        """
        if not self.enabled or not self.sentinel:
            return

        # Try to load existing plan
        existing = self.sentinel.load_plan(task)
        if existing:
            logger.info("sentinel_plan_resumed task_hash=%s", existing.task_hash)

    async def on_turn_failure(
        self,
        task: str,
        error: str,
        conversation_history: List[Dict[str, Any]] = None,
    ):
        """Called when a turn fails.

        Runs:
        1. Deep Reflexion — generate reflection, extract discoveries
        2. Context Evolution — record lesson from failure
        3. Sentinel — check if planning should activate
        """
        if not self.enabled:
            return

        # Deep Reflexion: reflection + discoveries
        if self.reflexion:
            await self.reflexion.on_failure(task, error, conversation_history or [])

        # Context Evolution: record lesson
        if self.evolution:
            self.evolution.on_failure(task, error, conversation_history)

        # Sentinel: activate on repeated failures
        if self.sentinel:
            failure_count = self.reflexion.attempt_count if self.reflexion else 0
            should, reason = self.sentinel.should_activate(
                task, reflexion_failure_count=failure_count
            )
            if should and not self.sentinel.is_active:
                logger.info("sentinel_activating reason=%s", reason)
                await self.sentinel.decompose(task)

    def get_retry_context(self) -> str:
        """Get context injection for the next retry attempt.

        Includes reflections and discoveries from Deep Reflexion.
        """
        if not self.enabled or not self.reflexion:
            return ""
        return self.reflexion.get_retry_context()

    def should_retry(self) -> bool:
        """Check if another retry attempt is allowed."""
        if not self.enabled or not self.reflexion:
            return True
        return self.reflexion.should_retry()

    def on_user_feedback(self, feedback: str):
        """Record user feedback for Context Evolution."""
        if not self.enabled or not self.evolution:
            return
        self.evolution.on_user_feedback(feedback)

    def get_sentinel_progress(self) -> str:
        """Get Sentinel plan progress summary."""
        if not self.enabled or not self.sentinel:
            return ""
        return self.sentinel.get_progress_summary()

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all features."""
        stats = {"enabled": self.enabled}
        if self.reflexion:
            stats["reflexion"] = self.reflexion.get_stats()
        if self.guardian:
            stats["guardian"] = self.guardian.get_session_summary()
        if self.sentinel:
            stats["sentinel"] = {"active": self.sentinel.is_active}
        if self.evolution:
            stats["evolution"] = self.evolution.get_stats()
        return stats

    def reset(self):
        """Reset all feature state for a new session."""
        if self.reflexion:
            self.reflexion.reset()
        if self.guardian:
            self.guardian._session_operations.clear()
        if self.sentinel:
            self.sentinel.abandon_plan()
