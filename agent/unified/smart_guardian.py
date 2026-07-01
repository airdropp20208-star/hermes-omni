r"""Smart Guardian — LLM-as-judge replacement for the pattern-based guardian.

The legacy guardian (agent/guardian.py + agent/unified/policy.py) blocks
actions using regex patterns. This is the wrong abstraction for a
reasoning-first agent:

- Regex cannot understand context. `rm -rf node_modules/` is safe;
  `rm -rf /` is not. The pattern `rm\s+-rf` cannot tell them apart.
- Regex cannot learn. Once written, the rules never improve.
- Regex over-blocks creative workflows. A senior engineer doing
  `git push --force` to their own branch knows what they're doing.

The Smart Guardian inverts the model:

- The PolicyEngine (pattern-based) still runs FIRST as a fast pre-filter
  for known-catastrophic patterns (rm -rf /, fork bomb, mkfs, etc.).
  These get BLOCKED at the pattern layer with zero LLM cost.
- Anything that passes the pattern layer but is classified as
  CONSEQUENTIAL or IRREVERSIBLE by the DecisionFramework gets sent to
  the LLM judge for a context-aware risk assessment.
- The judge sees: tool name, args, plan, classification reason, and the
  most recent reflexion lessons for similar situations. It returns one
  of: ALLOW / WARN / REQUIRE_USER_CONFIRM / BLOCK.
- The judge's decision is cached by a content-hash of its inputs, so
  repeated similar actions do not re-incur LLM cost.

This is the bridge between "unlimited mode" (no safety at all) and
"pattern-block mode" (regex can't reason). It gives the agent the
*appearance* of unlimited autonomy while actually maintaining a
context-aware safety net that scales with the LLM's intelligence.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Literal

from .decision import Classification, DecisionClass
from .reasoning import ReasoningPlan


class GuardianVerdict(str, Enum):
    """What the smart guardian decided about an action."""

    ALLOW = "allow"  # proceed, no warning
    WARN = "warn"  # proceed, but log/notify
    REQUIRE_USER_CONFIRM = "require_user_confirm"  # pause and ask the user
    BLOCK = "block"  # refuse; this is final


@dataclass(frozen=True)
class RiskAssessment:
    """A single LLM-judge assessment."""

    verdict: GuardianVerdict
    risk_level: Literal["negligible", "low", "moderate", "high", "severe"]
    reasoning: str
    conditions: str  # if non-empty, action is allowed only if these conditions hold
    similar_past_lessons: str  # what past reflexion said about similar actions
    created_at: float = field(default_factory=time.time)
    elapsed_ms: int = 0
    cache_hit: bool = False
    cache_key: str = ""

    @property
    def blocks(self) -> bool:
        return self.verdict is GuardianVerdict.BLOCK

    @property
    def requires_user(self) -> bool:
        return self.verdict is GuardianVerdict.REQUIRE_USER_CONFIRM


_JUDGE_SYSTEM_PROMPT = (
    "You are the Guardian of an autonomous AI agent. Your job is to "
    "decide whether the agent should be allowed to execute a planned "
    "action. You see the action, the agent's reasoning plan, the "
    "classification rationale, and any relevant lessons from past "
    "similar actions.\n\n"
    "Be calibrated, not paranoid. The goal is to allow legitimate work "
    "while preventing genuine disasters. A senior engineer would not "
    "ask permission for `git push origin feature-branch`, but would "
    "pause for `git push --force origin main`.\n\n"
    "Return STRICT JSON only. Schema:\n"
    "{\n"
    '  "verdict": "allow" | "warn" | "require_user_confirm" | "block",\n'
    '  "risk_level": "negligible" | "low" | "moderate" | "high" | "severe",\n'
    '  "reasoning": "one or two sentences explaining the verdict",\n'
    '  "conditions": "if non-empty, the action may proceed ONLY if these '
    'conditions are true (e.g., \'branch is not main/master\')",\n'
    '  "similar_past_lessons": "if past lessons informed this verdict, '
    'briefly note how; otherwise empty"\n'
    "}"
)


def _build_judge_user_prompt(
    *,
    tool_name: str,
    args: dict[str, Any] | None,
    plan: ReasoningPlan | None,
    classification: Classification,
    reflexion_context: str,
) -> str:
    plan_block = plan.as_prompt_block() if plan else "(no plan generated — classification was TRIVIAL or LLM unavailable)"
    args_str = json.dumps(args, ensure_ascii=False, default=str) if args else "(no args)"
    return (
        f"Tool: {tool_name}\n"
        f"Classification: {classification.cls.name} — {classification.reason}\n"
        f"Arguments: {args_str}\n\n"
        f"Plan:\n{plan_block}\n\n"
        f"Relevant past lessons:\n{reflexion_context or '(none)'}\n\n"
        "Return the verdict JSON now."
    )


def _cache_key(
    *,
    tool_name: str,
    args: dict[str, Any] | None,
    classification_reason: str,
    plan_decision: str,
) -> str:
    """Stable content-hash for caching judge verdicts.

    The cache key intentionally excludes the plan text itself (which
    varies with LLM temperature) — only the plan's *decision* field is
    included, so two plans with the same decision but different prose
    share a cache entry.
    """
    parts = [
        tool_name or "",
        json.dumps(args or {}, ensure_ascii=False, default=str, sort_keys=True),
        classification_reason,
        plan_decision,
    ]
    raw = "\0".join(parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


class SmartGuardian:
    """LLM-as-judge guardian for CONSEQUENTIAL+ actions.

    Lifecycle:
        1. PolicyEngine (pattern-based) runs FIRST — handles known
           catastrophic patterns with zero LLM cost.
        2. DecisionFramework classifies the action.
        3. If TRIVIAL or STANDARD, SmartGuardian returns ALLOW immediately.
        4. If CONSEQUENTIAL or IRREVERSIBLE, SmartGuardian calls the LLM
           judge (with caching).
        5. If LLM unavailable, SmartGuardian degrades to ALLOW (fail open).

    The cache is bounded (default 512 entries, LRU eviction). Verdicts
    have a TTL (default 1 hour) so stale verdicts don't permanently
    allow risky patterns.
    """

    def __init__(
        self,
        *,
        llm_call=None,
        reflexion_recall=None,
        cache_size: int = 512,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self._llm_call = llm_call
        # reflexion_recall(query, limit) -> str (formatted lessons)
        self._reflexion_recall = reflexion_recall or (lambda query, limit=3: "")
        self._cache_size = max(8, cache_size)
        self._cache_ttl = max(60, cache_ttl_seconds)
        self._cache: dict[str, tuple[float, RiskAssessment]] = {}
        self._lock = RLock()

    def assess(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
        plan: ReasoningPlan | None,
        classification: Classification,
    ) -> RiskAssessment:
        """Assess the risk of executing this action.

        Always returns a RiskAssessment. Never raises.
        """
        # Fast path: TRIVIAL and STANDARD actions skip the judge entirely.
        # The LLM judge is reserved for actions where the cost of a wrong
        # decision exceeds the cost of an LLM call.
        if classification.cls < DecisionClass.CONSEQUENTIAL:
            return RiskAssessment(
                verdict=GuardianVerdict.ALLOW,
                risk_level="negligible" if classification.is_trivial else "low",
                reasoning=f"Action classified as {classification.cls.name}; below guardian LLM-judge threshold.",
                conditions="",
                similar_past_lessons="",
            )

        # Fail open if no LLM available. We log this in the verdict so the
        # conversation loop / user can see that safety was degraded.
        if self._llm_call is None:
            return RiskAssessment(
                verdict=GuardianVerdict.WARN,
                risk_level="moderate",
                reasoning=(
                    f"Action classified as {classification.cls.name} but no LLM "
                    "judge is configured. Proceeding with degraded safety."
                ),
                conditions="",
                similar_past_lessons="",
            )

        # Cache lookup.
        key = _cache_key(
            tool_name=tool_name,
            args=args,
            classification_reason=classification.reason,
            plan_decision=(plan.decision if plan else "no_plan"),
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                ts, verdict = cached
                if (time.time() - ts) <= self._cache_ttl:
                    return RiskAssessment(
                        verdict=verdict.verdict,
                        risk_level=verdict.risk_level,
                        reasoning=verdict.reasoning,
                        conditions=verdict.conditions,
                        similar_past_lessons=verdict.similar_past_lessons,
                        elapsed_ms=0,
                        cache_hit=True,
                        cache_key=key,
                    )
                # expired
                self._cache.pop(key, None)

        # LLM judge call.
        try:
            reflexion_context = self._reflexion_recall(
                f"{tool_name} {classification.matched_pattern}",
                limit=3,
            )
            user_prompt = _build_judge_user_prompt(
                tool_name=tool_name,
                args=args,
                plan=plan,
                classification=classification,
                reflexion_context=reflexion_context,
            )
            started = time.time()
            raw = self._llm_call(_JUDGE_SYSTEM_PROMPT, user_prompt)
            elapsed = int((time.time() - started) * 1000)
            data = self._parse_json(raw)
            if data is None:
                # LLM returned garbage — fail open with a warning.
                return RiskAssessment(
                    verdict=GuardianVerdict.WARN,
                    risk_level="moderate",
                    reasoning="LLM judge returned unparseable output; proceeding with warning.",
                    conditions="",
                    similar_past_lessons="",
                    elapsed_ms=elapsed,
                )
            verdict = GuardianVerdict(str(data.get("verdict", "allow")).strip().lower())
            if verdict not in GuardianVerdict:
                verdict = GuardianVerdict.WARN
            risk_level = str(data.get("risk_level", "moderate")).strip().lower()
            if risk_level not in {"negligible", "low", "moderate", "high", "severe"}:
                risk_level = "moderate"
            assessment = RiskAssessment(
                verdict=verdict,
                risk_level=risk_level,  # type: ignore[arg-type]
                reasoning=str(data.get("reasoning", "")).strip(),
                conditions=str(data.get("conditions", "")).strip(),
                similar_past_lessons=str(data.get("similar_past_lessons", "")).strip(),
                elapsed_ms=elapsed,
                cache_key=key,
            )
        except Exception as exc:
            # Any unexpected failure → fail open with warning.
            return RiskAssessment(
                verdict=GuardianVerdict.WARN,
                risk_level="moderate",
                reasoning=f"Guardian LLM call failed: {exc!r}. Proceeding with warning.",
                conditions="",
                similar_past_lessons="",
            )

        # Cache write with LRU eviction.
        with self._lock:
            self._cache[key] = (time.time(), assessment)
            while len(self._cache) > self._cache_size:
                # Evict oldest by timestamp.
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                self._cache.pop(oldest_key, None)

        return assessment

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

_guardian: SmartGuardian | None = None


def get_guardian() -> SmartGuardian:
    global _guardian
    if _guardian is None:
        _guardian = SmartGuardian()
    return _guardian


def configure_guardian(
    *,
    llm_call=None,
    reflexion_recall=None,
    cache_size: int = 512,
    cache_ttl_seconds: int = 3600,
) -> SmartGuardian:
    global _guardian
    _guardian = SmartGuardian(
        llm_call=llm_call,
        reflexion_recall=reflexion_recall,
        cache_size=cache_size,
        cache_ttl=cache_ttl_seconds,
    )
    return _guardian
