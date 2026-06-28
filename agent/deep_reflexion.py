"""Deep Reflexion — Dual-layer reflective architecture for hermes-omni.

Inspired by OmniAgent's reflexion system, adapted for hermes-omni's architecture.

Features:
1. Tool Loop Detection — SHA-256 hashing of repeated tool calls
2. Error Repeat Detection — Same error occurring multiple times
3. No-Progress Detection — Tool overuse + result similarity
4. Discovery Extraction — Carry forward file knowledge across retries
5. Self-Reflection — LLM-powered failure analysis on retry
6. Failure Prevention — Trajectory repetition, error action repetition, loop pseudo-termination

Integration point: Wraps around the existing conversation loop to add
reflexion capabilities without modifying core agent code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReflexionState:
    """Tracks state across reflexion attempts."""

    reflections: List[str] = field(default_factory=list)
    discoveries: str = ""
    recent_tool_calls: List[str] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)
    tool_name_history: List[str] = field(default_factory=list)
    result_hashes: List[str] = field(default_factory=list)
    stuck_detected_count: int = 0
    loop_detection_window: int = 10
    loop_repeat_threshold: int = 3

    def reset_per_attempt(self):
        """Reset state that should be fresh per attempt."""
        self.recent_tool_calls = []
        self.recent_errors = []
        self.tool_name_history = []
        self.result_hashes = []
        # Keep reflections and discoveries across retries

    def reset_all(self):
        """Full reset."""
        self.reflections = []
        self.discoveries = ""
        self.recent_tool_calls = []
        self.recent_errors = []
        self.tool_name_history = []
        self.result_hashes = []
        self.stuck_detected_count = 0


class ToolLoopDetector:
    """Detects when the agent is stuck calling the same tool repeatedly."""

    def __init__(self, window: int = 10, threshold: int = 3):
        self.window = window
        self.threshold = threshold
        self._recent_calls: List[str] = []

    def check(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Check if this tool call is a repeat. Returns True if loop detected."""
        call_sig = hashlib.sha256(
            json.dumps({"tool": tool_name, "params": params}, sort_keys=True).encode()
        ).hexdigest()

        self._recent_calls.append(call_sig)
        if len(self._recent_calls) > self.window:
            self._recent_calls = self._recent_calls[-self.window:]

        if len(self._recent_calls) >= self.threshold:
            recent_tail = self._recent_calls[-self.threshold:]
            if len(set(recent_tail)) == 1 and recent_tail[0] == call_sig:
                logger.warning(
                    "tool_loop_detected tool=%s repeat=%d",
                    tool_name, self.threshold,
                )
                return True
        return False

    def reset(self):
        self._recent_calls = []


class ErrorRepeatDetector:
    """Detects when the same error keeps occurring."""

    def __init__(self, max_tracked: int = 20):
        self.max_tracked = max_tracked
        self._recent_errors: List[str] = []

    def check(self, error_msg: str) -> bool:
        """Returns True if this error has occurred before."""
        normalized = error_msg.strip()
        if normalized in self._recent_errors:
            logger.warning("error_repeat_detected error=%s", normalized[:80])
            return True
        self._recent_errors.append(normalized)
        if len(self._recent_errors) > self.max_tracked:
            self._recent_errors = self._recent_errors[-self.max_tracked:]
        return False

    def reset(self):
        self._recent_errors = []


class NoProgressDetector:
    """Detects when the agent is making no meaningful progress."""

    def __init__(self, tool_overuse_threshold: int = 6, window: int = 10):
        self.tool_overuse_threshold = tool_overuse_threshold
        self.window = window
        self._tool_name_history: List[str] = []
        self._result_hashes: List[str] = []

    def record_tool_call(self, tool_name: str):
        self._tool_name_history.append(tool_name)

    def record_result(self, result: str):
        result_hash = hashlib.sha256(result[:500].encode()).hexdigest()[:16]
        self._result_hashes.append(result_hash)

    def check(self) -> Optional[str]:
        """Check for no-progress. Returns warning message or None."""
        # Check 1: Tool overuse
        recent = self._tool_name_history[-self.window:]
        if len(recent) >= self.tool_overuse_threshold:
            counts: Dict[str, int] = {}
            for t in recent:
                counts[t] = counts.get(t, 0) + 1
            for tool_name, count in counts.items():
                if count >= self.tool_overuse_threshold:
                    return (
                        f"[NO PROGRESS] Tool '{tool_name}' called {count} times in "
                        f"last {len(recent)} calls without completing the task. "
                        f"Try a fundamentally different approach."
                    )

        # Check 2: Result similarity
        recent_hashes = self._result_hashes[-3:]
        if len(recent_hashes) >= 3 and len(set(recent_hashes)) == 1:
            return (
                "[NO PROGRESS] Last 3 tool results are nearly identical. "
                "You may be stuck in a repetitive pattern. "
                "Try a completely different strategy."
            )

        return None

    def reset(self):
        self._tool_name_history = []
        self._result_hashes = []


class DiscoveryExtractor:
    """Extracts key file paths and knowledge from conversation history.

    Carries forward discoveries across retries so the agent doesn't
    have to re-discover the same files.
    """

    @staticmethod
    def extract(conversation_history: List[Dict[str, Any]]) -> str:
        """Extract file paths and key info from tool calls."""
        discoveries = []
        seen = set()

        for msg in conversation_history:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue

                if name == "read_file" and "path" in args:
                    path = args["path"]
                    if path not in seen:
                        discoveries.append(f"- read_file: {path}")
                        seen.add(path)
                elif name == "write_file" and "path" in args:
                    path = args["path"]
                    if path not in seen:
                        discoveries.append(f"- write_file: {path}")
                        seen.add(path)
                elif name == "edit_file" and "path" in args:
                    path = args["path"]
                    if path not in seen:
                        discoveries.append(f"- edit_file: {path}")
                        seen.add(path)
                elif name in ("grep", "find") and "pattern" in args:
                    key = f"{name}:{args.get('pattern', '')}"
                    if key not in seen:
                        discoveries.append(
                            f"- {name}: pattern={args.get('pattern', '')}, "
                            f"path={args.get('path', '.')}"
                        )
                        seen.add(key)
                elif name == "bash" and "command" in args:
                    cmd = args["command"][:100]
                    key = f"bash:{cmd}"
                    if key not in seen:
                        discoveries.append(f"- bash: {cmd}")
                        seen.add(key)

        return "\n".join(discoveries) if discoveries else ""


class ReflectionGenerator:
    """Generates self-reflections on failed attempts using LLM."""

    def __init__(self, llm_call_fn=None):
        """
        Args:
            llm_call_fn: async function(messages, temperature, max_tokens) -> response
        """
        self._llm_call = llm_call_fn

    async def generate(
        self,
        task: str,
        error: str,
        conversation_history: List[Dict[str, Any]],
        safety_context: str = "",
    ) -> Optional[str]:
        """Generate a reflection on why the task failed."""
        if not self._llm_call:
            return None

        history_summary = self._summarize_attempt(conversation_history)

        prompt = f"""You attempted the following task but failed:
Task: {task}
Failure reason: {error or 'max iterations reached'}

Here is a summary of what you tried:
{history_summary}{safety_context}

Write 2-3 concise sentences explaining:
1. What went wrong
2. What you should do differently next time

Do NOT write code or tool calls. Only provide the reflection."""

        try:
            from agent.llm_providers import chat_completion

            response = await self._llm_call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            reflection = (getattr(response, "content", None) or "").strip()
            if len(reflection) < 20:
                return None
            logger.info("reflection_generated length=%d", len(reflection))
            return reflection
        except Exception as e:
            logger.warning("reflection_failed: %s", e)
            return None

    @staticmethod
    def _summarize_attempt(history: List[Dict[str, Any]]) -> str:
        """Summarize conversation history for reflection prompt."""
        events = []
        for msg in history[-30:]:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "unknown")
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                            args_str = json.dumps(args, ensure_ascii=False)[:150]
                        except Exception:
                            args_str = "?"
                        events.append(f"Called {name}({args_str})")
                elif content:
                    events.append(f"Assistant: {content[:200]}")
            elif role == "tool":
                if content and content.startswith("Error:"):
                    events.append(f"Tool error: {content[:200]}")
                elif content:
                    events.append(f"Tool result: {content[:200]}...")

        summary = "\n".join(events)
        return summary[:5000] if len(summary) > 5000 else summary


class DeepReflexion:
    """Orchestrates all reflexion components.

    Usage:
        reflexion = DeepReflexion(llm_call_fn=my_llm_func)

        # Per tool call
        if reflexion.on_tool_call("bash", {"command": "rm -rf /"}):
            print("Loop detected!")

        # Per tool result
        if reflexion.on_tool_result("Error: permission denied"):
            print("Repeat error!")

        # After failed attempt
        await reflexion.on_failure(task, error, history)

        # Before retry
        context = reflexion.get_retry_context()

        # Check if should retry
        if reflexion.should_retry():
            # retry...
    """

    def __init__(
        self,
        llm_call_fn=None,
        max_retries: int = 3,
        loop_window: int = 10,
        loop_threshold: int = 3,
        tool_overuse_threshold: int = 6,
    ):
        self.max_retries = max_retries
        self.attempt_count = 0

        self.loop_detector = ToolLoopDetector(
            window=loop_window, threshold=loop_threshold
        )
        self.error_detector = ErrorRepeatDetector()
        self.no_progress = NoProgressDetector(
            tool_overuse_threshold=tool_overuse_threshold
        )
        self.discovery_extractor = DiscoveryExtractor()
        self.reflection_generator = ReflectionGenerator(llm_call_fn)

        self.reflections: List[str] = []
        self.discoveries: str = ""
        self.stuck_count: int = 0

    def on_tool_call(self, tool_name: str, params: Dict[str, Any]) -> Optional[str]:
        """Check a tool call before execution.

        Returns:
            Warning message if loop/no-progress detected, None otherwise.
        """
        # Loop detection
        if self.loop_detector.check(tool_name, params):
            self.stuck_count += 1
            return (
                f"[LOOP DETECTED] Tool '{tool_name}' with the same parameters "
                f"has been called {self.loop_detector.threshold} times consecutively. "
                f"This approach is not working. Try a different tool or parameters."
            )

        # No-progress detection
        self.no_progress.record_tool_call(tool_name)
        warning = self.no_progress.check()
        if warning:
            self.stuck_count += 1
            return warning

        return None

    def on_tool_result(self, result: str) -> Optional[str]:
        """Check a tool result after execution.

        Returns:
            Warning if error repeat detected, None otherwise.
        """
        self.no_progress.record_result(result)

        if result and result.startswith("Error:"):
            if self.error_detector.check(result):
                return (
                    "\n\n[ERROR REPEAT] This same error occurred earlier. "
                    "The previous approach failed. Try a completely different strategy."
                )
        return None

    async def on_failure(
        self,
        task: str,
        error: str,
        conversation_history: List[Dict[str, Any]],
    ):
        """Handle a failed attempt — generate reflection and extract discoveries."""
        self.attempt_count += 1

        # Extract discoveries from this attempt
        attempt_discoveries = self.discovery_extractor.extract(conversation_history)
        if attempt_discoveries:
            self.discoveries = attempt_discoveries
            logger.info(
                "discoveries_extracted count=%d",
                attempt_discoveries.count("\n") + 1,
            )

        # Generate reflection
        reflection = await self.reflection_generator.generate(
            task=task,
            error=error,
            conversation_history=conversation_history,
        )
        if reflection:
            self.reflections.append(reflection)

    def get_retry_context(self) -> str:
        """Build context injection for the next retry attempt."""
        if not self.reflections and not self.discoveries:
            return ""

        lines = ["\n## Previous Attempt Context"]

        if self.discoveries:
            lines.append(
                "You attempted this task before and discovered these files/paths. "
                "Do NOT re-search or re-read them — use this knowledge directly:"
            )
            lines.append(self.discoveries)

        if self.reflections:
            lines.append("### What went wrong and what to do differently:")
            for i, reflection in enumerate(self.reflections, 1):
                lines.append(f"{i}. {reflection}")

        lines.append("")
        return "\n".join(lines)

    def should_retry(self) -> bool:
        """Check if we should retry the task."""
        return self.attempt_count < self.max_retries

    def reset(self):
        """Full reset."""
        self.attempt_count = 0
        self.reflections = []
        self.discoveries = ""
        self.stuck_count = 0
        self.loop_detector.reset()
        self.error_detector.reset()
        self.no_progress.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Get reflexion statistics."""
        return {
            "attempts": self.attempt_count,
            "reflections": len(self.reflections),
            "stuck_detections": self.stuck_count,
            "has_discoveries": bool(self.discoveries),
        }
