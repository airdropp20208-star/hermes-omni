"""Agent Harness — parallel execution + dependency graph + self-check.

WHAT IS A HARNESS?
------------------
A "harness" is the runtime wrapper around an LLM that turns it into an
agent. It handles:
1. Tool execution loop (call LLM → parse tool calls → execute → feed back)
2. Parallel tool execution (independent tools run concurrently)
3. Dependency graph (tool B depends on tool A's output)
4. Self-check (verify output before returning to user)
5. Progressive loading (load only needed context)

Hermes already has conversation_loop.py (the main harness), but it's
synchronous — tools execute one at a time. AgentHarness adds:

- **Parallel execution**: independent tool calls run in ThreadPoolExecutor
- **Dependency tracking**: if tool B needs tool A's output, B waits for A
- **Self-check**: after all tools complete, verify the response
- **Progressive loading**: load skills/context only when needed

This is inspired by OmniAgent's "HyperHarness" (parallel execution engine)
but reimplemented cleanly in agent/unified/.

USAGE
-----
    from agent.unified.harness import AgentHarness, ToolCall, ToolResult

    harness = AgentHarness(llm_call=my_llm, max_parallel=4)

    # Agent generates tool calls — harness executes them
    result = harness.run(
        user_message="Read file A and file B, then compare them",
        tool_calls=[
            ToolCall(tool="read_file", args={"path": "A"}),
            ToolCall(tool="read_file", args={"path": "B"}),  # parallel with A
            ToolCall(tool="compare", args={"a": "$1", "b": "$2"}, depends_on=[0, 1]),
        ],
    )
    # → A and B read in parallel, compare waits for both
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    """One tool call in a harness execution plan."""

    id: int
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)  # IDs of calls this depends on
    timeout: float = 60.0
    optional: bool = False  # if True, failure doesn't block dependents


@dataclass
class ToolResult:
    """Result of one tool call."""

    call_id: int
    tool: str
    success: bool
    result: str = ""
    error: str = ""
    elapsed_ms: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0


@dataclass
class HarnessResult:
    """Final result of a harness run."""

    tool_results: list[ToolResult] = field(default_factory=list)
    total_elapsed_ms: int = 0
    parallel_speedup: float = 1.0  # estimated speedup vs sequential
    all_succeeded: bool = True
    errors: list[str] = field(default_factory=list)
    self_check_passed: bool = True
    self_check_notes: str = ""


# --------------------------------------------------------------------------- #
# AgentHarness
# --------------------------------------------------------------------------- #


class AgentHarness:
    """Parallel tool execution harness with dependency tracking + self-check.

    Features:
    1. Parallel execution of independent tool calls (ThreadPoolExecutor)
    2. Dependency graph: tool B waits for tool A if depends_on=[A_id]
    3. Progressive loading: context loaded only when needed
    4. Self-check: verify all results before returning
    5. Timeout handling per tool call
    6. Optional tools: failure doesn't block dependents
    """

    def __init__(
        self,
        *,
        tool_executor: Callable[[str, dict[str, Any]], str] | None = None,
        max_parallel: int = 4,
        self_check_fn: Callable[[list[ToolResult], str], tuple[bool, str]] | None = None,
    ) -> None:
        self._executor_fn = tool_executor
        self._max_parallel = max(1, min(max_parallel, 16))
        self._self_check_fn = self_check_fn
        self._lock = threading.RLock()

    def run(
        self,
        *,
        tool_calls: list[ToolCall],
        user_message: str = "",
    ) -> HarnessResult:
        """Execute a batch of tool calls with parallel + dependency support.

        Args:
            tool_calls: list of ToolCall with optional depends_on
            user_message: original user message (for self-check)

        Returns:
            HarnessResult with all tool results + self-check status
        """
        started = time.time()
        if not tool_calls:
            return HarnessResult(total_elapsed_ms=0)

        # Build dependency graph.
        results: dict[int, ToolResult] = {}
        completed_ids: set[int] = set()
        failed_ids: set[int] = set()
        futures: dict[int, Future] = {}

        with ThreadPoolExecutor(max_workers=self._max_parallel) as pool:
            pending = list(tool_calls)
            while pending or futures:
                # Submit all calls whose deps are met.
                to_submit = []
                for call in pending:
                    deps_met = True
                    for dep_id in call.depends_on:
                        if dep_id in failed_ids and not self._is_optional_dep(call, dep_id):
                            # Dependency failed — skip this call.
                            results[call.id] = ToolResult(
                                call_id=call.id,
                                tool=call.tool,
                                success=False,
                                error=f"Dependency {dep_id} failed",
                            )
                            failed_ids.add(call.id)
                            deps_met = False
                            break
                        elif dep_id not in completed_ids:
                            deps_met = False
                            break
                    if deps_met:
                        to_submit.append(call)

                # Submit ready calls.
                for call in to_submit:
                    pending.remove(call)
                    # Resolve $N references in args (use previous results).
                    resolved_args = self._resolve_refs(call.args, results)
                    future = pool.submit(
                        self._execute_tool,
                        call=call,
                        args=resolved_args,
                    )
                    futures[call.id] = future

                # Wait for at least one to complete.
                if futures:
                    done, _ = self._wait_any(futures)
                    for call_id in done:
                        future = futures.pop(call_id)
                        result = future.result()
                        results[call_id] = result
                        if result.success:
                            completed_ids.add(call_id)
                        else:
                            failed_ids.add(call_id)

                # If nothing submitted and nothing pending, break.
                if not to_submit and not futures and not pending:
                    break
                # If nothing submitted but pending exists, wait for futures.
                if not to_submit and futures:
                    continue
                # If nothing submitted, nothing pending, nothing futures → deadlock.
                if not to_submit and not futures and pending:
                    # Mark remaining as failed (circular dep or missing dep).
                    for call in pending:
                        results[call.id] = ToolResult(
                            call_id=call.id,
                            tool=call.tool,
                            success=False,
                            error="Deadlock: unresolvable dependencies",
                        )
                        failed_ids.add(call.id)
                    pending.clear()

        elapsed = int((time.time() - started) * 1000)

        # Calculate parallel speedup estimate.
        sequential_time = sum(r.elapsed_ms for r in results.values())
        parallel_speedup = sequential_time / max(1, elapsed)

        # Self-check.
        self_check_passed = True
        self_check_notes = ""
        if self._self_check_fn is not None:
            try:
                self_check_passed, self_check_notes = self._self_check_fn(
                    list(results.values()), user_message
                )
            except Exception:
                self_check_passed = False
                self_check_notes = "Self-check failed"

        all_succeeded = all(r.success for r in results.values())
        errors = [r.error for r in results.values() if r.error]

        return HarnessResult(
            tool_results=list(results.values()),
            total_elapsed_ms=elapsed,
            parallel_speedup=parallel_speedup,
            all_succeeded=all_succeeded,
            errors=errors,
            self_check_passed=self_check_passed,
            self_check_notes=self_check_notes,
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _execute_tool(self, *, call: ToolCall, args: dict[str, Any]) -> ToolResult:
        """Execute one tool call."""
        started = time.time()
        if self._executor_fn is None:
            return ToolResult(
                call_id=call.id,
                tool=call.tool,
                success=False,
                error="No tool executor configured",
                elapsed_ms=0,
            )
        try:
            result = self._executor_fn(call.tool, args)
            elapsed = int((time.time() - started) * 1000)
            # Check if result indicates failure.
            success = True
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and parsed.get("error"):
                        success = False
                except Exception:
                    pass
            return ToolResult(
                call_id=call.id,
                tool=call.tool,
                success=success,
                result=result if isinstance(result, str) else json.dumps(result, default=str),
                elapsed_ms=elapsed,
                started_at=started,
                completed_at=time.time(),
            )
        except Exception as exc:
            elapsed = int((time.time() - started) * 1000)
            return ToolResult(
                call_id=call.id,
                tool=call.tool,
                success=False,
                error=repr(exc),
                elapsed_ms=elapsed,
                started_at=started,
                completed_at=time.time(),
            )

    def _resolve_refs(
        self,
        args: dict[str, Any],
        results: dict[int, ToolResult],
    ) -> dict[str, Any]:
        """Resolve $N references in args to actual results.

        Example: {"a": "$0"} → {"a": results[0].result}
        """
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                try:
                    ref_id = int(value[1:])
                    if ref_id in results:
                        resolved[key] = results[ref_id].result
                    else:
                        resolved[key] = value  # keep as-is if not found
                except ValueError:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _is_optional_dep(call: ToolCall, dep_id: int) -> bool:
        """Check if a failed dependency is optional."""
        # For now, all deps are required unless call.optional is True.
        return call.optional

    @staticmethod
    def _wait_any(futures: dict[int, Future]) -> tuple[set[int], set[int]]:
        """Wait for any future to complete. Returns (done_ids, pending_ids)."""
        done_ids: set[int] = set()
        pending_ids: set[int] = set()
        for call_id, future in futures.items():
            if future.done():
                done_ids.add(call_id)
            else:
                pending_ids.add(call_id)
        if not done_ids and futures:
            # Wait for first to complete.
            for future in as_completed(futures.values(), timeout=120):
                for call_id, f in futures.items():
                    if f is future:
                        done_ids.add(call_id)
                        break
                break
        return done_ids, pending_ids


# --------------------------------------------------------------------------- #
# Convenience: auto-plan tool calls from user message
# --------------------------------------------------------------------------- #


def auto_plan_tools(
    *,
    user_message: str,
    available_tools: list[str],
    llm_call: Callable[[str, str], str] | None = None,
) -> list[ToolCall]:
    """Use LLM to plan tool calls from a user message.

    Returns a list of ToolCall with dependency information.
    Falls back to empty list if LLM unavailable.
    """
    if llm_call is None:
        return []
    system = (
        "You are the tool-planning layer. Given a user message and available "
        "tools, plan which tools to call and in what order. Specify "
        "dependencies (tool B depends on tool A if B needs A's output).\n\n"
        "Return STRICT JSON:\n"
        '{"calls": [{"tool": "name", "args": {}, "depends_on": [0]}]}\n'
        "\nUse $N in args to reference result of call N (0-indexed).\n"
        "Independent calls (no deps) will run in PARALLEL."
    )
    user = (
        f"User: {user_message}\n\n"
        f"Available tools: {', '.join(available_tools[:30])}\n\n"
        "Plan tool calls now."
    )
    try:
        raw = llm_call(system, user)
        # Parse JSON.
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        data = json.loads(text)
        calls = []
        for i, c in enumerate(data.get("calls", [])):
            calls.append(
                ToolCall(
                    id=i,
                    tool=str(c.get("tool", "")),
                    args=c.get("args", {}),
                    depends_on=[int(d) for d in c.get("depends_on", [])],
                )
            )
        return calls
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_harness: AgentHarness | None = None


def get_harness() -> AgentHarness | None:
    return _harness


def configure_harness(
    *,
    tool_executor: Callable[[str, dict[str, Any]], str] | None = None,
    max_parallel: int = 4,
    self_check_fn: Callable[[list[ToolResult], str], tuple[bool, str]] | None = None,
) -> AgentHarness:
    global _harness
    _harness = AgentHarness(
        tool_executor=tool_executor,
        max_parallel=max_parallel,
        self_check_fn=self_check_fn,
    )
    return _harness


def run_harness(
    *,
    tool_calls: list[ToolCall],
    user_message: str = "",
) -> dict[str, Any]:
    """Public API: run harness. Returns dict with results."""
    if _harness is None:
        return {"success": False, "error": "harness not configured"}
    result = _harness.run(tool_calls=tool_calls, user_message=user_message)
    return {
        "success": result.all_succeeded and result.self_check_passed,
        "tool_results": [
            {
                "call_id": r.call_id,
                "tool": r.tool,
                "success": r.success,
                "result": r.result[:500],
                "error": r.error,
                "elapsed_ms": r.elapsed_ms,
            }
            for r in result.tool_results
        ],
        "total_elapsed_ms": result.total_elapsed_ms,
        "parallel_speedup": round(result.parallel_speedup, 2),
        "self_check_passed": result.self_check_passed,
        "self_check_notes": result.self_check_notes,
        "errors": result.errors,
    }


def harness_stats() -> dict[str, Any]:
    """Public API: get harness config."""
    if _harness is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "max_parallel": _harness._max_parallel,
        "has_executor": _harness._executor_fn is not None,
        "has_self_check": _harness._self_check_fn is not None,
    }
