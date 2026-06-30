"""Context Distiller — compression that keeps INSIGHTS, not just summaries.

THE PROBLEM
-----------
Standard context compression (which Hermes has via `agent/context_compressor.py`)
works by SUMMARIZING: "the user asked X, the agent did Y, the result was Z".
This loses the *insights* — the transferable lessons, the dead-ends explored,
the user's actual preferences.

A senior engineer's notebook doesn't say "tried 3 approaches, one worked".
It says "approach A failed because of [specific reason]; approach B works
but is slow; approach C is what we use in production". The INSIGHTS are
what matter for future work.

ContextDistiller produces structured distillations that preserve:
1. **Decisions made** — what was chosen and why (not just what was tried)
2. **Dead-ends** — what was tried and rejected, with the rejection reason
3. **User preferences** — what the user liked/disliked/corrected
4. **Open questions** — things that were unclear and never resolved
5. **Transferable lessons** — insights that apply to similar future tasks

This is NOT a replacement for context compression. It runs ALONGSIDE it:
- Compressor reduces token count (operates on raw conversation)
- Distiller extracts structured knowledge (operates on the compressed result)

WHEN IT RUNS
------------
- Periodically (every N turns, configurable)
- When context is about to be compressed
- When a session is being closed

The distilled context is injected into the system prompt for future turns,
so the agent "remembers" insights even after the raw conversation is gone.

TOKEN ECONOMICS
---------------
- 1 LLM call per distillation (every N turns)
- The distilled block is small (~500 tokens) and replaces ~2000-5000 tokens
  of raw conversation in future context
- Net: significant savings for long sessions, plus better reasoning because
  insights are explicit instead of buried in narrative
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class DistilledContext:
    """Structured distillation of a conversation segment."""

    decisions: list[str] = field(default_factory=list)
    # ["Chose approach B over A because A requires sudo"]
    dead_ends: list[str] = field(default_factory=list)
    # ["Tried editing the YAML directly — config loader doesn't support that"]
    user_preferences: list[str] = field(default_factory=list)
    # ["User prefers concise responses", "User wants code reviewed before commit"]
    open_questions: list[str] = field(default_factory=list)
    # ["Never confirmed whether the production DB has the same schema"]
    transferable_lessons: list[str] = field(default_factory=list)
    # ["The cron job fails silently when /tmp is full — check disk first"]
    facts_established: list[str] = field(default_factory=list)
    # ["Repo uses Python 3.11", "Tests run via pytest, not unittest"]
    created_at: float = field(default_factory=time.time)
    source_turns: tuple[int, int] = (0, 0)  # which turns this covers
    raw_token_estimate: int = 0  # tokens in the source conversation
    distilled_token_estimate: int = 0

    def to_prompt_block(self) -> str:
        """Render as a markdown block for system prompt injection."""
        if not any([self.decisions, self.dead_ends, self.user_preferences,
                    self.open_questions, self.transferable_lessons, self.facts_established]):
            return ""
        lines = ["<distilled-context>"]
        if self.decisions:
            lines.append("Decisions:")
            for d in self.decisions:
                lines.append(f"  - {d}")
        if self.dead_ends:
            lines.append("Dead-ends (don't retry):")
            for d in self.dead_ends:
                lines.append(f"  - {d}")
        if self.user_preferences:
            lines.append("User preferences:")
            for p in self.user_preferences:
                lines.append(f"  - {p}")
        if self.facts_established:
            lines.append("Established facts:")
            for f in self.facts_established:
                lines.append(f"  - {f}")
        if self.transferable_lessons:
            lines.append("Transferable lessons:")
            for l in self.transferable_lessons:
                lines.append(f"  - {l}")
        if self.open_questions:
            lines.append("Open questions (consider resolving):")
            for q in self.open_questions:
                lines.append(f"  - {q}")
        lines.append("</distilled-context>")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_DISTILL_SYSTEM = (
    "You are the context distillation layer of an AI agent. You receive "
    "a segment of conversation and extract STRUCTURED KNOWLEDGE that will "
    "be useful for future turns. Do NOT summarize the narrative — extract "
    "INSIGHTS.\n\n"
    "Focus on:\n"
    "- decisions: what was chosen and WHY (not just what was tried)\n"
    "- dead_ends: what was tried and rejected, with the rejection reason\n"
    "- user_preferences: what the user liked/disliked/corrected\n"
    "- open_questions: things that were unclear and never resolved\n"
    "- transferable_lessons: insights that apply to similar future tasks\n"
    "- facts_established: confirmed facts about the environment/task\n\n"
    "Be CONCISE. Each item should be one sentence. Skip empty categories.\n\n"
    "Return STRICT JSON:\n"
    "{\n"
    '  "decisions": ["..."],\n'
    '  "dead_ends": ["..."],\n'
    '  "user_preferences": ["..."],\n'
    '  "open_questions": ["..."],\n'
    '  "transferable_lessons": ["..."],\n'
    '  "facts_established": ["..."]\n'
    "}"
)


# --------------------------------------------------------------------------- #
# ContextDistiller
# --------------------------------------------------------------------------- #


class ContextDistiller:
    """Extracts structured insights from conversation segments.

    Maintains a rolling distillation: each call adds to the accumulated
    knowledge. Old distillations can be merged (1 LLM call to merge 2
    distillations into 1) to keep the block bounded.
    """

    def __init__(
        self,
        *,
        llm_call=None,
        distill_every_n_turns: int = 10,
        max_distilled_items: int = 30,
        merge_threshold: int = 50,
    ) -> None:
        self._llm_call = llm_call
        self._distill_every = max(3, distill_every_n_turns)
        self._max_items = max(10, max_distilled_items)
        self._merge_threshold = max(20, merge_threshold)
        self._current = DistilledContext()
        self._turns_since_distill = 0

    def maybe_distill(
        self,
        *,
        turn_count: int,
        conversation_segment: str,
        turn_start: int,
        turn_end: int,
    ) -> DistilledContext | None:
        """Distill if enough turns have passed. Returns the new distillation
        (or None if not enough turns yet)."""
        if self._llm_call is None:
            return None
        if turn_count - self._turns_since_distill < self._distill_every:
            return None
        self._turns_since_distill = turn_count
        return self.distill(
            conversation_segment=conversation_segment,
            turn_start=turn_start,
            turn_end=turn_end,
        )

    def distill(
        self,
        *,
        conversation_segment: str,
        turn_start: int,
        turn_end: int,
    ) -> DistilledContext | None:
        """Force a distillation of the given conversation segment."""
        if self._llm_call is None:
            return None
        try:
            user = (
                f"Conversation segment (turns {turn_start}-{turn_end}):\n"
                f"{conversation_segment}\n\n"
                "Extract structured knowledge now."
            )
            raw = self._llm_call(_DISTILL_SYSTEM, user)
            data = self._parse_json(raw)
            if data is None:
                return None
            new = DistilledContext(
                decisions=[str(s).strip() for s in data.get("decisions", []) if str(s).strip()],
                dead_ends=[str(s).strip() for s in data.get("dead_ends", []) if str(s).strip()],
                user_preferences=[str(s).strip() for s in data.get("user_preferences", []) if str(s).strip()],
                open_questions=[str(s).strip() for s in data.get("open_questions", []) if str(s).strip()],
                transferable_lessons=[str(s).strip() for s in data.get("transferable_lessons", []) if str(s).strip()],
                facts_established=[str(s).strip() for s in data.get("facts_established", []) if str(s).strip()],
                source_turns=(turn_start, turn_end),
                raw_token_estimate=len(conversation_segment) // 4,
                distilled_token_estimate=0,
            )
            new.distilled_token_estimate = len(new.to_prompt_block()) // 4

            # Merge with current accumulated distillation.
            self._current = self._merge(self._current, new)

            # If too many items, trigger a compression merge via LLM.
            total_items = (
                len(self._current.decisions)
                + len(self._current.dead_ends)
                + len(self._current.user_preferences)
                + len(self._current.open_questions)
                + len(self._current.transferable_lessons)
                + len(self._current.facts_established)
            )
            if total_items > self._merge_threshold:
                self._current = self._compress_via_llm(self._current)
            return new
        except Exception:
            return None

    def get_current(self) -> DistilledContext:
        return self._current

    def get_prompt_block(self) -> str:
        """The current distilled context, rendered for system prompt."""
        return self._current.to_prompt_block()

    def reset(self) -> None:
        self._current = DistilledContext()
        self._turns_since_distill = 0

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge(a: DistilledContext, b: DistilledContext) -> DistilledContext:
        """Simple set-union merge of two distillations. Deduplicates by exact
        string match. Does NOT call LLM — that's _compress_via_llm."""
        def dedupe(items: list[str]) -> list[str]:
            seen: set[str] = set()
            result: list[str] = []
            for item in items:
                key = item.lower().strip()
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        return DistilledContext(
            decisions=dedupe(a.decisions + b.decisions),
            dead_ends=dedupe(a.dead_ends + b.dead_ends),
            user_preferences=dedupe(a.user_preferences + b.user_preferences),
            open_questions=dedupe(a.open_questions + b.open_questions),
            transferable_lessons=dedupe(a.transferable_lessons + b.transferable_lessons),
            facts_established=dedupe(a.facts_established + b.facts_established),
            source_turns=(min(a.source_turns[0], b.source_turns[0]), max(a.source_turns[1], b.source_turns[1])),
            raw_token_estimate=a.raw_token_estimate + b.raw_token_estimate,
            distilled_token_estimate=0,
        )

    def _compress_via_llm(self, ctx: DistilledContext) -> DistilledContext:
        """Call LLM to compress an over-large distillation down to essentials.
        Keeps only the most important items per category."""
        if self._llm_call is None:
            return ctx
        try:
            system = (
                "You compress a structured context distillation down to its "
                "most important items. Keep at most "
                f"{self._max_items // 6} items per category. Drop items that "
                "are redundant, outdated, or low-value. Return the same JSON "
                "schema, but smaller."
            )
            user = (
                "Current distillation:\n"
                + json.dumps({
                    "decisions": ctx.decisions,
                    "dead_ends": ctx.dead_ends,
                    "user_preferences": ctx.user_preferences,
                    "open_questions": ctx.open_questions,
                    "transferable_lessons": ctx.transferable_lessons,
                    "facts_established": ctx.facts_established,
                }, ensure_ascii=False, indent=2)
                + "\n\nReturn the compressed JSON."
            )
            raw = self._llm_call(system, user)
            data = self._parse_json(raw)
            if data is None:
                return ctx
            return DistilledContext(
                decisions=[str(s).strip() for s in data.get("decisions", []) if str(s).strip()][: self._max_items // 6],
                dead_ends=[str(s).strip() for s in data.get("dead_ends", []) if str(s).strip()][: self._max_items // 6],
                user_preferences=[str(s).strip() for s in data.get("user_preferences", []) if str(s).strip()][: self._max_items // 6],
                open_questions=[str(s).strip() for s in data.get("open_questions", []) if str(s).strip()][: self._max_items // 6],
                transferable_lessons=[str(s).strip() for s in data.get("transferable_lessons", []) if str(s).strip()][: self._max_items // 6],
                facts_established=[str(s).strip() for s in data.get("facts_established", []) if str(s).strip()][: self._max_items // 6],
                source_turns=ctx.source_turns,
                raw_token_estimate=ctx.raw_token_estimate,
                distilled_token_estimate=0,
            )
        except Exception:
            return ctx

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_distiller: ContextDistiller | None = None


def get_distiller() -> ContextDistiller | None:
    return _distiller


def configure_distiller(
    *,
    llm_call=None,
    distill_every_n_turns: int = 10,
    max_distilled_items: int = 30,
    merge_threshold: int = 50,
) -> ContextDistiller | None:
    global _distiller
    if llm_call is None:
        _distiller = None
        return None
    _distiller = ContextDistiller(
        llm_call=llm_call,
        distill_every_n_turns=distill_every_n_turns,
        max_distilled_items=max_distilled_items,
        merge_threshold=merge_threshold,
    )
    return _distiller
