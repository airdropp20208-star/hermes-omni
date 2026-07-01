"""Metacognitive Monitor — self-doubt and confidence calibration.

THE PROBLEM
-----------
LLMs are notoriously miscalibrated. When asked "are you sure?", they
say "yes" with high confidence regardless of whether they're right.
This is dangerous in an agent: the agent confidently executes a wrong
plan, fails, retries the same plan, fails again, loops forever.

MetacognitiveMonitor fixes this by:
1. **Tracking prediction accuracy** — after each action, compare the
   agent's stated confidence to the actual outcome. Build a calibration
   curve.
2. **Applying calibration** — when the agent says "80% confident",
   apply the calibration curve. If the agent's historical accuracy at
   "80% confident" was actually 55%, downgrade the confidence to 55%.
3. **Triggering self-doubt** — if calibrated confidence is below a
   threshold, force the agent to reconsider: generate alternatives,
   seek more evidence, or ask the user.

This is "metacognition" in the literal sense: thinking about thinking.
The agent monitors its own reasoning quality and adjusts.

WHEN IT RUNS
------------
- After every action with a stated confidence (CognitiveTree, plans)
- After every tool result (for failure detection)
- Periodically to recompute the calibration curve

The monitor is mostly passive — it observes and adjusts. The only
active intervention is when calibrated confidence drops below the
threshold, in which case it returns a "self_doubt" signal that the
conversation loop can act on.

TOKEN ECONOMICS
---------------
- 0 LLM calls for tracking and calibration (pure statistics)
- 1 LLM call per self-doubt event (to generate alternatives)
- Self-doubt events are rare (only when calibrated confidence < 0.5)

Net: near-zero overhead, large benefit when it triggers.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ConfidenceRecord:
    """One data point in the calibration history."""

    stated_confidence: float  # what the agent claimed (0.0 to 1.0)
    actual_outcome: Literal["success", "failure", "partial"]
    tool_name: str = ""
    task_category: str = ""  # "diagnostic", "creative", "factual", etc.
    timestamp: float = field(default_factory=time.time)


@dataclass
class CalibrationCurve:
    """Binned calibration data. Bin width = 0.1 (10 bins from 0.0 to 1.0)."""

    bins: dict[int, list[bool]] = field(default_factory=lambda: defaultdict(list))
    # bin_index (0-9) → list of success booleans

    def add(self, stated: float, success: bool) -> None:
        bin_idx = max(0, min(9, int(stated * 10)))
        self.bins[bin_idx].append(success)

    def calibrated_confidence(self, stated: float) -> float:
        """Given a stated confidence, return the historical accuracy
        in that bin. Falls back to the stated value if no data."""
        bin_idx = max(0, min(9, int(stated * 10)))
        outcomes = self.bins.get(bin_idx, [])
        if len(outcomes) < 5:
            # Not enough data — trust the stated value but apply a
            # small discount for unknown miscalibration.
            return stated * 0.95
        return sum(outcomes) / len(outcomes)

    def sample_count(self) -> int:
        return sum(len(v) for v in self.bins.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": {str(k): v for k, v in self.bins.items()},
            "sample_count": self.sample_count(),
        }


@dataclass
class SelfDoubtSignal:
    """A signal that the agent should reconsider."""

    trigger: str  # "low_confidence" | "repeated_failure" | "contradicts_reflexion"
    stated_confidence: float
    calibrated_confidence: float
    recommendation: Literal["proceed_with_caution", "generate_alternatives", "ask_user", "abort"]
    rationale: str
    timestamp: float = field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# MetacognitiveMonitor
# --------------------------------------------------------------------------- #


class MetacognitiveMonitor:
    """Tracks calibration and triggers self-doubt.

    In-memory state. A future version could persist calibration data
    across sessions for per-user-per-model curves.
    """

    def __init__(
        self,
        *,
        llm_call=None,
        self_doubt_threshold: float = 0.5,
        repeated_failure_count: int = 3,
        min_samples_for_calibration: int = 5,
    ) -> None:
        self._llm_call = llm_call
        self._doubt_threshold = max(0.0, min(self_doubt_threshold, 1.0))
        self._repeated_failure = max(2, repeated_failure_count)
        self._min_samples = max(1, min_samples_for_calibration)
        self._curve = CalibrationCurve()
        self._recent_outcomes: list[bool] = []  # last N successes/failures
        self._recent_outcomes_max = 20
        self._per_tool_failure_streaks: dict[str, int] = defaultdict(int)
        self._doubt_history: list[SelfDoubtSignal] = []

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_outcome(
        self,
        *,
        stated_confidence: float,
        actual_outcome: Literal["success", "failure", "partial"],
        tool_name: str = "",
        task_category: str = "",
    ) -> SelfDoubtSignal | None:
        """Record the outcome of an action. Returns a SelfDoubtSignal if
        the monitor recommends reconsideration, else None."""
        success = actual_outcome == "success"
        self._curve.add(stated_confidence, success)
        self._recent_outcomes.append(success)
        if len(self._recent_outcomes) > self._recent_outcomes_max:
            self._recent_outcomes.pop(0)
        # Per-tool failure streak
        if tool_name:
            if success:
                self._per_tool_failure_streaks[tool_name] = 0
            else:
                self._per_tool_failure_streaks[tool_name] += 1

        # Check for self-doubt triggers.
        signal = self._check_for_doubt(
            stated_confidence=stated_confidence,
            tool_name=tool_name,
        )
        if signal is not None:
            self._doubt_history.append(signal)
        return signal

    def calibrate(self, stated_confidence: float) -> float:
        """Given a stated confidence, return the calibrated confidence
        based on historical accuracy."""
        return self._curve.calibrated_confidence(stated_confidence)

    # ------------------------------------------------------------------ #
    # Self-doubt triggers
    # ------------------------------------------------------------------ #

    def _check_for_doubt(
        self,
        *,
        stated_confidence: float,
        tool_name: str,
    ) -> SelfDoubtSignal | None:
        # Trigger 1: calibrated confidence below threshold.
        calibrated = self.calibrate(stated_confidence)
        if calibrated < self._doubt_threshold and self._curve.sample_count() >= self._min_samples:
            return SelfDoubtSignal(
                trigger="low_confidence",
                stated_confidence=stated_confidence,
                calibrated_confidence=calibrated,
                recommendation="generate_alternatives" if calibrated < 0.3 else "proceed_with_caution",
                rationale=(
                    f"Stated confidence {stated_confidence:.2f} but historical "
                    f"accuracy in this bin is {calibrated:.2f}."
                ),
            )
        # Trigger 2: repeated failure on the same tool.
        if tool_name and self._per_tool_failure_streaks.get(tool_name, 0) >= self._repeated_failure:
            streak = self._per_tool_failure_streaks[tool_name]
            return SelfDoubtSignal(
                trigger="repeated_failure",
                stated_confidence=stated_confidence,
                calibrated_confidence=calibrated,
                recommendation="generate_alternatives" if streak >= self._repeated_failure + 1 else "ask_user",
                rationale=(
                    f"Tool {tool_name!r} has failed {streak} times in a row. "
                    "Same approach is unlikely to work."
                ),
            )
        # Trigger 3: recent outcomes mostly failures.
        if len(self._recent_outcomes) >= 5:
            recent_success_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
            if recent_success_rate < 0.3:
                return SelfDoubtSignal(
                    trigger="low_confidence",
                    stated_confidence=stated_confidence,
                    calibrated_confidence=calibrated,
                    recommendation="ask_user",
                    rationale=(
                        f"Recent success rate is {recent_success_rate:.0%}. "
                        "The agent may be operating outside its competence."
                    ),
                )
        return None

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        return {
            "calibration_samples": self._curve.sample_count(),
            "calibration_curve": self._curve.to_dict(),
            "recent_success_rate": (
                sum(self._recent_outcomes) / len(self._recent_outcomes)
                if self._recent_outcomes
                else None
            ),
            "recent_sample_count": len(self._recent_outcomes),
            "per_tool_failure_streaks": dict(self._per_tool_failure_streaks),
            "doubt_events_count": len(self._doubt_history),
            "last_doubt": (
                {
                    "trigger": self._doubt_history[-1].trigger,
                    "recommendation": self._doubt_history[-1].recommendation,
                    "rationale": self._doubt_history[-1].rationale,
                }
                if self._doubt_history
                else None
            ),
        }

    def reset(self) -> None:
        """Clear all calibration data. Use when switching models or users."""
        self._curve = CalibrationCurve()
        self._recent_outcomes.clear()
        self._per_tool_failure_streaks.clear()
        self._doubt_history.clear()


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_monitor: MetacognitiveMonitor | None = None


def get_monitor() -> MetacognitiveMonitor | None:
    return _monitor


def configure_monitor(
    *,
    llm_call=None,
    self_doubt_threshold: float = 0.5,
    repeated_failure_count: int = 3,
    min_samples_for_calibration: int = 5,
) -> MetacognitiveMonitor | None:
    global _monitor
    _monitor = MetacognitiveMonitor(
        llm_call=llm_call,
        self_doubt_threshold=self_doubt_threshold,
        repeated_failure_count=repeated_failure_count,
        min_samples_for_calibration=min_samples_for_calibration,
    )
    return _monitor
