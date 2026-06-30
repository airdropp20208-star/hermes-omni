"""Cost Tracker — token accounting + budget caps per phase/session.

WHY THIS EXISTS
---------------
v3 cognitive modules add 5-15 LLM calls per request (slow thinking,
verifier, ensemble, etc.). Without tracking, you fly blind — no idea
what each phase costs, no way to set budgets, no alerts when near limit.

This module wraps every LLM call to count tokens, attribute them to a
phase, and enforce per-session budgets. Persisted to
~/.hermes/unified/cost_log.jsonl for audit + analytics.

USAGE
-----
    from agent.unified.cost_tracker import tracked_llm_call, get_cost_summary

    # Wrap any LLM call:
    response = tracked_llm_call("plan", lambda: llm(system, user))

    # Get summary:
    summary = get_cost_summary()
    # → {"total_tokens": 125000, "by_phase": {"plan": 5000, ...},
    #    "budget_remaining": 75000, "budget_exceeded": False}
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class CostRecord:
    """One LLM call's cost record."""

    phase: str  # "plan" | "critique" | "reflect" | "verify" | "ensemble" | "guardian" | ...
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    elapsed_ms: int = 0
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    cache_hit: bool = False  # True if response came from cache (free)


@dataclass
class BudgetConfig:
    """Per-session budget configuration."""

    total_token_budget: int = 1_000_000  # 1M tokens default
    per_phase_budget: dict[str, int] = field(default_factory=lambda: {
        "plan": 50_000,
        "critique": 30_000,
        "reflect": 50_000,
        "verify": 100_000,
        "ensemble": 200_000,
        "guardian": 50_000,
        "slow_thinking": 200_000,
        "cognitive_tree": 100_000,
        "hypothesis": 80_000,
        "default": 50_000,
    })
    warn_at_fraction: float = 0.8  # warn when 80% of budget used
    hard_stop: bool = True  # block calls when budget exceeded


# --------------------------------------------------------------------------- #
# CostTracker
# --------------------------------------------------------------------------- #


class CostTracker:
    """Token counter + budget enforcer. Thread-safe."""

    def __init__(
        self,
        *,
        log_path: str | Path | None = None,
        budget: BudgetConfig | None = None,
    ) -> None:
        if log_path is None:
            from hermes_constants import get_hermes_home

            log_path = get_hermes_home() / "unified" / "cost_log.jsonl"
        self._log_path = Path(log_path).expanduser()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._budget = budget or BudgetConfig()
        self._lock = threading.RLock()
        self._records: list[CostRecord] = []
        self._by_phase: dict[str, int] = defaultdict(int)
        self._total = 0
        self._session_id = ""

    def set_session(self, session_id: str) -> None:
        with self._lock:
            self._session_id = session_id or ""

    def record(
        self,
        *,
        phase: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "",
        elapsed_ms: int = 0,
        cache_hit: bool = False,
    ) -> CostRecord:
        """Record a single LLM call. Returns the record."""
        total = prompt_tokens + completion_tokens
        if cache_hit:
            # Cache hits are free — don't count toward budget.
            total = 0
        rec = CostRecord(
            phase=phase,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            model=model,
            elapsed_ms=elapsed_ms,
            session_id=self._session_id,
            cache_hit=cache_hit,
        )
        with self._lock:
            self._records.append(rec)
            self._by_phase[phase] += total
            self._total += total
            # Persist (append-only JSONL).
            try:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(rec), ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass
        return rec

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return max(1, len(text) // 4)

    def check_budget(self, phase: str = "") -> dict[str, Any]:
        """Check if a call would exceed budget. Returns status dict."""
        with self._lock:
            total_used = self._total
            phase_used = self._by_phase.get(phase, 0)
        total_budget = self._budget.total_token_budget
        phase_budget = self._budget.per_phase_budget.get(
            phase, self._budget.per_phase_budget.get("default", 50_000)
        )
        total_remaining = total_budget - total_used
        phase_remaining = phase_budget - phase_used
        total_fraction = total_used / total_budget if total_budget > 0 else 0.0
        phase_fraction = phase_used / phase_budget if phase_budget > 0 else 0.0
        return {
            "phase": phase,
            "total_used": total_used,
            "total_budget": total_budget,
            "total_remaining": max(0, total_remaining),
            "total_fraction": total_fraction,
            "phase_used": phase_used,
            "phase_budget": phase_budget,
            "phase_remaining": max(0, phase_remaining),
            "phase_fraction": phase_fraction,
            "warn_total": total_fraction >= self._budget.warn_at_fraction,
            "warn_phase": phase_fraction >= self._budget.warn_at_fraction,
            "exceeded_total": total_used >= total_budget,
            "exceeded_phase": phase_used >= phase_budget,
        }

    def should_block(self, phase: str = "") -> bool:
        """Return True if the call should be blocked (hard stop)."""
        if not self._budget.hard_stop:
            return False
        status = self.check_budget(phase)
        return status["exceeded_total"] or status["exceeded_phase"]

    def summary(self) -> dict[str, Any]:
        """Get full cost summary."""
        with self._lock:
            by_phase = dict(self._by_phase)
            total = self._total
            records_count = len(self._records)
        return {
            "total_tokens": total,
            "total_budget": self._budget.total_token_budget,
            "total_remaining": max(0, self._budget.total_token_budget - total),
            "total_fraction": total / self._budget.total_token_budget if self._budget.total_token_budget > 0 else 0.0,
            "by_phase": by_phase,
            "records_count": records_count,
            "session_id": self._session_id,
            "budget_exceeded": total >= self._budget.total_token_budget,
        }

    def reset(self) -> None:
        """Clear in-memory counters (does NOT delete persisted log)."""
        with self._lock:
            self._records.clear()
            self._by_phase.clear()
            self._total = 0


# --------------------------------------------------------------------------- #
# Tracked LLM call wrapper
# --------------------------------------------------------------------------- #


def tracked_llm_call(
    phase: str,
    fn: Callable[[], T],
    *,
    model: str = "",
    estimate_tokens: bool = True,
) -> T:
    """Wrap an LLM call with cost tracking.

    Args:
        phase: "plan" | "critique" | "reflect" | "verify" | "ensemble" | ...
        fn: callable() -> response (str or object with .usage)
        model: model name for logging
        estimate_tokens: if True and response has no usage, estimate from text

    Returns:
        The original response from fn().
    """
    tracker = get_tracker()
    if tracker is None:
        return fn()
    # Check budget before call.
    if tracker.should_block(phase):
        # Budget exceeded — return empty result instead of calling LLM.
        # The caller should handle empty/None gracefully (cognitive
        # modules all fail-open).
        return ""  # type: ignore[return-value]
    started = time.time()
    try:
        response = fn()
    except Exception:
        # Still record the attempt (0 tokens).
        tracker.record(
            phase=phase,
            prompt_tokens=0,
            completion_tokens=0,
            model=model,
            elapsed_ms=int((time.time() - started) * 1000),
        )
        raise
    elapsed_ms = int((time.time() - started) * 1000)

    # Extract token usage if available.
    prompt_tokens = 0
    completion_tokens = 0
    cache_hit = False
    try:
        # OpenAI-style response object.
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            # Check if cached (OpenAI returns cached_tokens in prompt_tokens_details).
            details = getattr(usage, "prompt_tokens_details", None)
            if details and getattr(details, "cached_tokens", 0):
                cache_hit = True
    except Exception:
        pass

    # If no usage info, estimate from response text.
    if estimate_tokens and prompt_tokens == 0 and completion_tokens == 0:
        try:
            text = response if isinstance(response, str) else str(response)
            completion_tokens = tracker.estimate_tokens(text)
        except Exception:
            pass

    tracker.record(
        phase=phase,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
        elapsed_ms=elapsed_ms,
        cache_hit=cache_hit,
    )
    return response


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_tracker: CostTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> CostTracker | None:
    return _tracker


def configure_tracker(
    *,
    log_path: str | Path | None = None,
    budget: BudgetConfig | None = None,
) -> CostTracker:
    global _tracker
    with _tracker_lock:
        _tracker = CostTracker(log_path=log_path, budget=budget or BudgetConfig())
        return _tracker


def get_cost_summary() -> dict[str, Any]:
    """Public API: get current cost summary."""
    if _tracker is None:
        return {"enabled": False}
    return _tracker.summary()


def check_cost_budget(phase: str = "") -> dict[str, Any]:
    """Public API: check budget for a phase."""
    if _tracker is None:
        return {"enabled": False, "exceeded": False}
    return _tracker.check_budget(phase)


def reset_cost_tracker() -> None:
    """Public API: reset counters (start new session budget)."""
    if _tracker is not None:
        _tracker.reset()
