"""Slow Thinking — multi-level deep reasoning for hard tasks.

WHY THIS EXISTS
---------------
Top-tier models (Claude 4.5, GLM 5.2, GPT-5) "think slow" — they
generate 10K+ tokens of internal reasoning before answering. This is
WHY they're good at math, coding, analysis.

Mid-tier models skip this. They answer fast and wrong.

SlowThinking gives mid-tier models the same capability at runtime:
generate structured reasoning traces of varying depth BEFORE the final
answer.

FOUR LEVELS (user picks per task)
---------------------------------
- **fast** (default): no slow thinking. Direct answer. ~0 extra tokens.
  Use for: chat, simple Q&A, lookup.

- **balanced**: 1 round of structured reasoning (decompose → analyze →
  synthesize). ~1-2K extra tokens. Use for: medium-complexity tasks,
  explanations, simple code.

- **deep**: 2-3 rounds, each refining the previous. Includes
  verification step. ~3-5K extra tokens. Use for: hard problems,
  multi-step coding, analysis with edge cases.

- **max**: 4-5 rounds with explicit self-critique between each round.
  Explores multiple approaches (mini CognitiveTree), picks best,
  verifies. ~8-15K extra tokens. Use for: critical decisions,
  irreversible actions, competition-level problems.

ARCHITECTURE
------------
```
                    ┌─────────────────────────┐
                    │  User request + level   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Round 1: Decompose      │
                    │  - break into subproblems │
                    │  - identify what's known  │
                    │  - identify what's needed │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Round 2: Analyze         │
                    │  - solve each subproblem  │
                    │  - note assumptions       │
                    │  - note uncertainty       │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Round 3: Synthesize      │
                    │  - combine solutions      │
                    │  - check consistency      │
                    │  - identify gaps          │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │ (deep/max only)                     │
              ▼                                     ▼
   ┌────────────────────┐              ┌────────────────────┐
   │  Round 4: Critique │              │  (max only)        │
   │  - find flaws      │              │  Round 5: Refine   │
   │  - test edge cases │              │  - apply critique  │
   └─────────┬──────────┘              │  - final verify    │
             │                         └─────────┬──────────┘
             ▼                                   ▼
        ┌────────────────────────────────────────────┐
        │  Final Answer                              │
        │  (only after reasoning trace is complete)  │
        └────────────────────────────────────────────┘
```

TOKEN ECONOMICS
---------------
Cost is INTENTIONAL — slow thinking trades tokens for quality.
- fast: 0 extra
- balanced: +1-2K tokens
- deep: +3-5K tokens
- max: +8-15K tokens

User chooses. For trivial chat, fast. For "design a distributed
database", max. The config sets the DEFAULT level; user can override
per-request via `reasoning_level` parameter.

INTEGRATION
-----------
- Runs BEFORE the final answer is generated
- Reasoning trace is stored (optional) for audit/debugging
- Verifier runs AFTER slow thinking (catches flaws in the final answer)
- For `max` level, integrates CognitiveTree to explore approaches
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Levels
# --------------------------------------------------------------------------- #


class ThinkingLevel(IntEnum):
    """Higher = deeper reasoning = more tokens."""

    FAST = 0
    BALANCED = 10
    DEEP = 20
    MAX = 30


def parse_level(s: str | ThinkingLevel | int) -> ThinkingLevel:
    if isinstance(s, ThinkingLevel):
        return s
    if isinstance(s, int):
        return ThinkingLevel(s)
    s_lower = str(s).lower().strip()
    if s_lower in ("fast", "0", "off", "none"):
        return ThinkingLevel.FAST
    if s_lower in ("balanced", "medium", "10"):
        return ThinkingLevel.BALANCED
    if s_lower in ("deep", "20"):
        return ThinkingLevel.DEEP
    if s_lower in ("max", "maximum", "30", "ultra"):
        return ThinkingLevel.MAX
    return ThinkingLevel.FAST


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ReasoningRound:
    """One round of slow thinking."""

    round_num: int
    phase: str  # "decompose" | "analyze" | "synthesize" | "critique" | "refine"
    content: str
    elapsed_ms: int = 0


@dataclass
class SlowThinkingResult:
    """Output of a slow thinking session."""

    level: ThinkingLevel
    rounds: list[ReasoningRound] = field(default_factory=list)
    final_answer: str = ""
    total_elapsed_ms: int = 0
    total_llm_calls: int = 0
    reasoning_trace: str = ""  # concatenated rounds for storage


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_DECOMPOSE_SYSTEM = (
    "You are the decomposition phase of slow thinking. Break the problem "
    "into smaller sub-problems. For each, identify:\n"
    "- what's known (given in the request)\n"
    "- what's unknown (needs to be figured out)\n"
    "- what assumptions are needed\n"
    "- what tools/info might help\n\n"
    "Be thorough. A good decomposition makes the rest easy.\n\n"
    "Return plain text (not JSON). Use markdown headers for sub-problems."
)

_ANALYZE_SYSTEM = (
    "You are the analysis phase of slow thinking. Given the decomposition, "
    "solve each sub-problem. Show your work. Note any:\n"
    "- assumptions you're making\n"
    "- uncertainty in your solution\n"
    "- alternative approaches considered\n\n"
    "Return plain text (not JSON). Use markdown headers for each sub-problem."
)

_SYNTHESIZE_SYSTEM = (
    "You are the synthesis phase of slow thinking. Combine the sub-problem "
    "solutions into a coherent whole. Check:\n"
    "- consistency between sub-problems\n"
    "- gaps not addressed\n"
    "- edge cases missed\n\n"
    "Return plain text (not JSON). End with a 'CONCLUSION:' section."
)

_CRITIQUE_SYSTEM = (
    "You are the self-critique phase of slow thinking. Review the synthesized "
    "solution. Find FLAWS:\n"
    "- logical errors\n"
    "- unstated assumptions\n"
    "- missing edge cases\n"
    "- overconfident claims\n\n"
    "Be harsh. Better to find flaws now than after output.\n\n"
    "Return plain text. List each flaw as a bullet point."
)

_REFINE_SYSTEM = (
    "You are the refinement phase of slow thinking. Apply the critique to "
    "the synthesized solution. Fix every flaw identified. If a flaw can't "
    "be fixed, note why.\n\n"
    "Return plain text. End with 'REFINED CONCLUSION:'."
)

_FINAL_ANSWER_SYSTEM = (
    "You are producing the final answer after slow thinking. Given the "
    "reasoning trace, produce a clear, accurate response to the original "
    "request. Do NOT include the reasoning trace in the output — just the "
    "answer the user wants.\n\n"
    "Return plain text."
)

# max-level: explore multiple approaches (mini CognitiveTree)
_EXPLORE_APPROACHES_SYSTEM = (
    "You are the exploration phase of MAX slow thinking. Generate 3 DISTINCT "
    "approaches to solve the problem. For each:\n"
    "- name the approach\n"
    "- describe how it works\n"
    "- list pros and cons\n"
    "- estimate difficulty\n\n"
    "Then pick the best approach and explain why.\n\n"
    "Return plain text."
)


# --------------------------------------------------------------------------- #
# SlowThinking engine
# --------------------------------------------------------------------------- #


class SlowThinkingEngine:
    """Multi-level deep reasoning engine."""

    def __init__(
        self,
        *,
        llm_call=None,
        default_level: ThinkingLevel = ThinkingLevel.FAST,
        store_traces: bool = False,
    ) -> None:
        self._llm_call = llm_call
        self._default_level = default_level
        self._store_traces = store_traces
        self._traces: list[SlowThinkingResult] = []

    def think(
        self,
        *,
        request: str,
        context: str = "",
        level: ThinkingLevel | str | int | None = None,
    ) -> SlowThinkingResult:
        """Run slow thinking at the given level. Returns result with final answer."""
        if level is None:
            level = self._default_level
        elif not isinstance(level, ThinkingLevel):
            level = parse_level(level)

        if level == ThinkingLevel.FAST or self._llm_call is None:
            # Fast path: direct answer, no slow thinking.
            return SlowThinkingResult(
                level=ThinkingLevel.FAST,
                final_answer="",
                total_elapsed_ms=0,
                total_llm_calls=0,
            )

        started = time.time()
        rounds: list[ReasoningRound] = []
        calls = 0

        # MAX level: explore approaches first.
        if level == ThinkingLevel.MAX:
            r, content = self._run_round(
                round_num=0,
                phase="explore",
                system=_EXPLORE_APPROACHES_SYSTEM,
                user=f"Request: {request}\n\nContext:\n{context or '(none)'}",
            )
            rounds.append(r)
            calls += 1
            # Use the chosen approach as additional context for decomposition.
            context = f"{context}\n\n[Chosen approach from exploration]:\n{content}"

        # Round 1: Decompose.
        r1, decomp = self._run_round(
            round_num=1,
            phase="decompose",
            system=_DECOMPOSE_SYSTEM,
            user=f"Request: {request}\n\nContext:\n{context or '(none)'}",
        )
        rounds.append(r1)
        calls += 1

        # Round 2: Analyze.
        r2, analysis = self._run_round(
            round_num=2,
            phase="analyze",
            system=_ANALYZE_SYSTEM,
            user=f"Request: {request}\n\nDecomposition:\n{decomp}",
        )
        rounds.append(r2)
        calls += 1

        # Round 3: Synthesize.
        r3, synthesis = self._run_round(
            round_num=3,
            phase="synthesize",
            system=_SYNTHESIZE_SYSTEM,
            user=f"Request: {request}\n\nDecomposition:\n{decomp}\n\nAnalysis:\n{analysis}",
        )
        rounds.append(r3)
        calls += 1

        current_solution = synthesis

        # DEEP and MAX: critique + refine.
        if level in (ThinkingLevel.DEEP, ThinkingLevel.MAX):
            r4, critique = self._run_round(
                round_num=4,
                phase="critique",
                system=_CRITIQUE_SYSTEM,
                user=f"Request: {request}\n\nSynthesized solution:\n{synthesis}",
            )
            rounds.append(r4)
            calls += 1

            r5, refined = self._run_round(
                round_num=5,
                phase="refine",
                system=_REFINE_SYSTEM,
                user=f"Request: {request}\n\nSynthesized solution:\n{synthesis}\n\nCritique:\n{critique}",
            )
            rounds.append(r5)
            calls += 1
            current_solution = refined

        # MAX: second critique pass for extra rigor.
        if level == ThinkingLevel.MAX:
            r6, critique2 = self._run_round(
                round_num=6,
                phase="critique",
                system=_CRITIQUE_SYSTEM,
                user=f"Request: {request}\n\nRefined solution:\n{current_solution}",
            )
            rounds.append(r6)
            calls += 1
            if critique2.strip():
                r7, refined2 = self._run_round(
                    round_num=7,
                    phase="refine",
                    system=_REFINE_SYSTEM,
                    user=f"Request: {request}\n\nRefined solution:\n{current_solution}\n\nCritique:\n{critique2}",
                )
                rounds.append(r7)
                calls += 1
                current_solution = refined2

        # Final answer: synthesize from the reasoning trace.
        reasoning_trace = "\n\n".join(
            f"## Round {r.round_num} ({r.phase})\n{r.content}" for r in rounds
        )
        r_final, final = self._run_round(
            round_num=len(rounds) + 1,
            phase="final",
            system=_FINAL_ANSWER_SYSTEM,
            user=f"Original request: {request}\n\nReasoning trace:\n{reasoning_trace}\n\nProduce the final answer now.",
        )
        calls += 1

        result = SlowThinkingResult(
            level=level,
            rounds=rounds,
            final_answer=final,
            total_elapsed_ms=int((time.time() - started) * 1000),
            total_llm_calls=calls,
            reasoning_trace=reasoning_trace,
        )
        if self._store_traces:
            self._traces.append(result)
        return result

    def list_traces(self) -> list[dict[str, Any]]:
        return [
            {
                "level": r.level.name,
                "rounds": len(r.rounds),
                "final_answer_preview": r.final_answer[:200],
                "total_llm_calls": r.total_llm_calls,
                "total_elapsed_ms": r.total_elapsed_ms,
            }
            for r in self._traces
        ]

    def _run_round(
        self,
        *,
        round_num: int,
        phase: str,
        system: str,
        user: str,
    ) -> tuple[ReasoningRound, str]:
        started = time.time()
        try:
            raw = self._llm_call(system, user)
            content = raw.strip() if raw else ""
        except Exception as exc:
            content = f"[{phase} failed: {exc!r}]"
        elapsed = int((time.time() - started) * 1000)
        return (
            ReasoningRound(round_num=round_num, phase=phase, content=content, elapsed_ms=elapsed),
            content,
        )


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_engine: SlowThinkingEngine | None = None


def get_slow_thinking() -> SlowThinkingEngine | None:
    return _engine


def configure_slow_thinking(
    *,
    llm_call=None,
    default_level: ThinkingLevel | str | int = ThinkingLevel.FAST,
    store_traces: bool = False,
) -> SlowThinkingEngine | None:
    global _engine
    if llm_call is None:
        _engine = None
        return None
    if not isinstance(default_level, ThinkingLevel):
        default_level = parse_level(default_level)
    _engine = SlowThinkingEngine(
        llm_call=llm_call,
        default_level=default_level,
        store_traces=store_traces,
    )
    return _engine


def think(
    *,
    request: str,
    context: str = "",
    level: ThinkingLevel | str | int | None = None,
) -> SlowThinkingResult:
    """Public API: run slow thinking. Returns result with final_answer."""
    if _engine is None:
        return SlowThinkingResult(level=ThinkingLevel.FAST, final_answer="")
    return _engine.think(request=request, context=context, level=level)
