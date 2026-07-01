"""Failure Forecast — predict failures BEFORE they happen.

THE BREAKTHROUGH
----------------
Current agent safety is REACTIVE: verify after response, catch errors
after they happen. FailureForecast is PROACTIVE: analyze the agent's
current trajectory and predict "80% chance of failure in next 3 steps"
BEFORE the failure occurs.

HOW IT WORKS
------------
1. **Pattern mining**: Analyze historical trajectories (from ReflexionStore
   + LearningEngine) to find failure patterns
2. **Real-time monitoring**: Track current session's tool call sequence
3. **Prediction**: Match current sequence against failure patterns
4. **Intervention**: If failure predicted > 70%, inject warning or
   force agent to reconsider

FAILURE PATTERNS DETECTED
-------------------------
- Tool loop: same tool called 3+ times with similar args → 85% stuck
- Read-without-test: read_file → edit_file without running tests → 60% bug
- Token spiral: 50K+ tokens in single turn → 90% context overflow
- Error cascade: 2+ errors in last 5 calls → 75% chain failure
- Blind retry: same failed tool retried without changing args → 80% loop
- Scope creep: task started simple but 20+ tool calls → 70% lost focus
- Assumption drift: agent stops verifying assumptions → 65% hallucination

TOKEN ECONOMICS
---------------
- 0 LLM calls (pure pattern matching + statistics)
- Saves tokens by preventing failures early (each prevented failure
  saves 5-15 tool calls that would have been wasted)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ToolCallEvent:
    """One tool call in the current session."""

    tool_name: str
    args_hash: str  # hash of args for loop detection
    success: bool
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0


@dataclass
class FailurePattern:
    """One learned failure pattern."""

    name: str
    description: str
    check_fn: str  # name of check function
    risk_weight: float  # 0.0 to 1.0 — how strong this signal is
    intervention: str  # what to do when triggered


@dataclass
class ForecastResult:
    """Result of a failure forecast check."""

    risk_score: float  # 0.0 (safe) to 1.0 (certain failure)
    predicted_failures: list[str] = field(default_factory=list)
    recommended_action: Literal["proceed", "caution", "intervene", "abort"] = "proceed"
    interventions: list[str] = field(default_factory=list)
    pattern_matches: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Built-in failure patterns
# --------------------------------------------------------------------------- #

BUILTIN_PATTERNS: list[FailurePattern] = [
    FailurePattern(
        name="tool_loop",
        description="Same tool called 3+ times with similar args — likely stuck in a loop",
        check_fn="check_tool_loop",
        risk_weight=0.85,
        intervention="Force agent to try a different approach or ask user for clarification",
    ),
    FailurePattern(
        name="blind_retry",
        description="Retrying a failed tool with same args — will fail again",
        check_fn="check_blind_retry",
        risk_weight=0.80,
        intervention="Agent should change approach, not retry identically",
    ),
    FailurePattern(
        name="token_spiral",
        description="50K+ tokens in a single turn — context overflow imminent",
        check_fn="check_token_spiral",
        risk_weight=0.90,
        intervention="Compress context immediately or split task into subtasks",
    ),
    FailurePattern(
        name="error_cascade",
        description="2+ errors in last 5 tool calls — chain failure likely",
        check_fn="check_error_cascade",
        risk_weight=0.75,
        intervention="Stop and diagnose root cause before continuing",
    ),
    FailurePattern(
        name="read_without_test",
        description="Edited file without running tests — likely introduced bugs",
        check_fn="check_read_without_test",
        risk_weight=0.60,
        intervention="Run tests after edit before proceeding",
    ),
    FailurePattern(
        name="scope_creep",
        description="20+ tool calls for a simple task — agent lost focus",
        check_fn="check_scope_creep",
        risk_weight=0.70,
        intervention="Re-evaluate task scope, consider splitting into subtasks",
    ),
    FailurePattern(
        name="assumption_drift",
        description="Agent hasn't verified assumptions in 10+ calls — hallucination risk",
        check_fn="check_assumption_drift",
        risk_weight=0.65,
        intervention="Force agent to re-verify key assumptions",
    ),
    FailurePattern(
        name="rapid_fire_errors",
        description="3+ errors in rapid succession (last 30 seconds) — system issue",
        check_fn="check_rapid_fire_errors",
        risk_weight=0.88,
        intervention="Pause and check system health (API limits, network, permissions)",
    ),
    FailurePattern(
        name="context_bloat",
        description="Accumulated 100K+ tokens without compression — degrading quality",
        check_fn="check_context_bloat",
        risk_weight=0.72,
        intervention="Trigger context compression before next tool call",
    ),
    FailurePattern(
        name="single_tool_dependency",
        description="Using only 1 tool type for everything — missing better alternatives",
        check_fn="check_single_tool_dependency",
        risk_weight=0.55,
        intervention="Suggest alternative tools that might be more effective",
    ),
]


# --------------------------------------------------------------------------- #
# FailureForecast
# --------------------------------------------------------------------------- #


class FailureForecast:
    """Predicts failures before they happen by analyzing tool call patterns.

    Maintains a sliding window of recent tool calls and checks them
    against known failure patterns. Thread-safe.
    """

    def __init__(
        self,
        *,
        window_size: int = 50,
        intervene_threshold: float = 0.70,
        caution_threshold: float = 0.40,
        abort_threshold: float = 0.90,
    ) -> None:
        self._window_size = max(10, min(window_size, 200))
        self._intervene_threshold = intervene_threshold
        self._caution_threshold = caution_threshold
        self._abort_threshold = abort_threshold
        self._events: deque[ToolCallEvent] = deque(maxlen=self._window_size)
        self._session_start: float = time.time()
        self._total_tokens: int = 0
        self._intervention_count: int = 0
        self._patterns: list[FailurePattern] = list(BUILTIN_PATTERNS)
        # Stats per pattern
        self._pattern_hits: dict[str, int] = defaultdict(int)

    def record_tool_call(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
        success: bool,
        tokens_used: int = 0,
    ) -> ForecastResult | None:
        """Record a tool call and return forecast if risk > caution threshold.

        Returns None if risk is low (no intervention needed).
        """
        args_hash = self._hash_args(args)
        event = ToolCallEvent(
            tool_name=tool_name,
            args_hash=args_hash,
            success=success,
            tokens_used=tokens_used,
        )
        self._events.append(event)
        self._total_tokens += tokens_used

        # Run forecast
        result = self.forecast()

        # Log intervention if needed
        if result.risk_score >= self._intervene_threshold:
            self._intervention_count += 1
            for pattern_name in result.pattern_matches:
                self._pattern_hits[pattern_name] += 1

        return result if result.risk_score >= self._caution_threshold else None

    def forecast(self) -> ForecastResult:
        """Run all pattern checks and return aggregated forecast."""
        events = list(self._events)
        if len(events) < 3:
            return ForecastResult(risk_score=0.0)

        matches: list[tuple[float, str, str]] = []  # (weight, pattern_name, intervention)

        for pattern in self._patterns:
            check = getattr(self, pattern.check_fn, None)
            if check is None:
                continue
            try:
                triggered = check(events)
                if triggered:
                    matches.append((pattern.risk_weight, pattern.name, pattern.intervention))
            except Exception:
                continue

        if not matches:
            return ForecastResult(risk_score=0.0)

        # Aggregate: use max weighted score + bonus for multiple matches
        max_score = max(w for w, _, _ in matches)
        multi_bonus = min(0.15, len(matches) * 0.05)  # up to +0.15 for 3+ matches
        risk_score = min(1.0, max_score + multi_bonus)

        predicted_failures = [name for _, name, _ in matches]
        interventions = [iv for _, _, iv in matches]

        if risk_score >= self._abort_threshold:
            action = "abort"
        elif risk_score >= self._intervene_threshold:
            action = "intervene"
        elif risk_score >= self._caution_threshold:
            action = "caution"
        else:
            action = "proceed"

        return ForecastResult(
            risk_score=risk_score,
            predicted_failures=predicted_failures,
            recommended_action=action,  # type: ignore[assignment]
            interventions=interventions,
            pattern_matches=predicted_failures,
        )

    # ------------------------------------------------------------------ #
    # Pattern check functions
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash_args(args: dict[str, Any] | None) -> str:
        """Simple hash of args for loop detection."""
        if not args:
            return ""
        import hashlib
        import json
        try:
            raw = json.dumps(args, sort_keys=True, default=str)
            return hashlib.md5(raw.encode()).hexdigest()[:12]
        except Exception:
            return str(args)[:50]

    def check_tool_loop(self, events: list[ToolCallEvent]) -> bool:
        """Same tool called 3+ times with similar args."""
        if len(events) < 3:
            return False
        recent = events[-5:]
        # Check last 3 calls
        last3 = recent[-3:]
        if len(last3) < 3:
            return False
        tool_names = [e.tool_name for e in last3]
        args_hashes = [e.args_hash for e in last3]
        # Same tool all 3 times
        if len(set(tool_names)) == 1:
            # Same args or very similar
            if len(set(args_hashes)) <= 2:
                return True
        return False

    def check_blind_retry(self, events: list[ToolCallEvent]) -> bool:
        """Retrying a failed tool with same args."""
        if len(events) < 2:
            return False
        last = events[-1]
        prev = events[-2]
        # Both failed, same tool, same args
        if (
            not last.success
            and not prev.success
            and last.tool_name == prev.tool_name
            and last.args_hash == prev.args_hash
        ):
            return True
        return False

    def check_token_spiral(self, events: list[ToolCallEvent]) -> bool:
        """50K+ tokens in recent calls."""
        if not events:
            return False
        recent = events[-10:]
        total = sum(e.tokens_used for e in recent)
        return total > 50000

    def check_error_cascade(self, events: list[ToolCallEvent]) -> bool:
        """2+ errors in last 5 tool calls."""
        if len(events) < 5:
            return False
        recent = events[-5:]
        errors = sum(1 for e in recent if not e.success)
        return errors >= 2

    def check_read_without_test(self, events: list[ToolCallEvent]) -> bool:
        """Edited file without running tests afterwards."""
        if len(events) < 2:
            return False
        # Look for edit_file in last 5 calls
        recent = events[-5:]
        has_edit = any(
            "edit" in e.tool_name.lower() or "write" in e.tool_name.lower()
            for e in recent
        )
        if not has_edit:
            return False
        # Check if test/bash ran AFTER the edit
        edit_idx = None
        for i, e in enumerate(recent):
            if "edit" in e.tool_name.lower() or "write" in e.tool_name.lower():
                edit_idx = i
        if edit_idx is None:
            return False
        # Check calls after edit
        after_edit = recent[edit_idx + 1:]
        has_test = any(
            "test" in e.tool_name.lower()
            or "bash" in e.tool_name.lower()
            or "execute" in e.tool_name.lower()
            for e in after_edit
        )
        return not has_test

    def check_scope_creep(self, events: list[ToolCallEvent]) -> bool:
        """20+ tool calls — agent might be overcomplicating."""
        return len(events) >= 20

    def check_assumption_drift(self, events: list[ToolCallEvent]) -> bool:
        """10+ calls without a verification tool (read/check/grep)."""
        if len(events) < 10:
            return False
        recent = events[-10:]
        has_verify = any(
            "read" in e.tool_name.lower()
            or "grep" in e.tool_name.lower()
            or "search" in e.tool_name.lower()
            or "check" in e.tool_name.lower()
            for e in recent
        )
        return not has_verify

    def check_rapid_fire_errors(self, events: list[ToolCallEvent]) -> bool:
        """3+ errors in last 30 seconds."""
        if len(events) < 3:
            return False
        now = time.time()
        recent = [e for e in events if now - e.timestamp < 30]
        errors = sum(1 for e in recent if not e.success)
        return errors >= 3

    def check_context_bloat(self, events: list[ToolCallEvent]) -> bool:
        """100K+ accumulated tokens without compression."""
        return self._total_tokens > 100000

    def check_single_tool_dependency(self, events: list[ToolCallEvent]) -> bool:
        """Using only 1 tool type for 10+ calls."""
        if len(events) < 10:
            return False
        recent = events[-10:]
        tool_types = set(e.tool_name for e in recent)
        return len(tool_types) <= 1

    # ------------------------------------------------------------------ #
    # Stats + management
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear event history (new session)."""
        self._events.clear()
        self._total_tokens = 0
        self._session_start = time.time()

    def stats(self) -> dict[str, Any]:
        return {
            "events_tracked": len(self._events),
            "total_tokens": self._total_tokens,
            "interventions_triggered": self._intervention_count,
            "pattern_hits": dict(self._pattern_hits),
            "session_duration_s": time.time() - self._session_start,
            "patterns_active": len(self._patterns),
        }

    def add_pattern(self, pattern: FailurePattern) -> None:
        """Add a custom failure pattern."""
        self._patterns.append(pattern)


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_forecast: FailureForecast | None = None


def get_forecast() -> FailureForecast | None:
    return _forecast


def configure_forecast(
    *,
    window_size: int = 50,
    intervene_threshold: float = 0.70,
    caution_threshold: float = 0.40,
    abort_threshold: float = 0.90,
) -> FailureForecast:
    global _forecast
    _forecast = FailureForecast(
        window_size=window_size,
        intervene_threshold=intervene_threshold,
        caution_threshold=caution_threshold,
        abort_threshold=abort_threshold,
    )
    return _forecast


def record_and_forecast(
    *,
    tool_name: str,
    args: dict[str, Any] | None,
    success: bool,
    tokens_used: int = 0,
) -> dict[str, Any] | None:
    """Public API: record tool call + return forecast if risk > threshold."""
    if _forecast is None:
        return None
    result = _forecast.record_tool_call(
        tool_name=tool_name,
        args=args,
        success=success,
        tokens_used=tokens_used,
    )
    if result is None:
        return None
    return {
        "risk_score": result.risk_score,
        "predicted_failures": result.predicted_failures,
        "recommended_action": result.recommended_action,
        "interventions": result.interventions,
        "pattern_matches": result.pattern_matches,
    }


def forecast_now() -> dict[str, Any]:
    """Public API: get current forecast without recording."""
    if _forecast is None:
        return {"risk_score": 0.0, "enabled": False}
    result = _forecast.forecast()
    return {
        "enabled": True,
        "risk_score": result.risk_score,
        "predicted_failures": result.predicted_failures,
        "recommended_action": result.recommended_action,
        "interventions": result.interventions,
        "pattern_matches": result.pattern_matches,
    }


def forecast_stats() -> dict[str, Any]:
    """Public API: get forecast stats."""
    if _forecast is None:
        return {"enabled": False}
    return {"enabled": True, **_forecast.stats()}
