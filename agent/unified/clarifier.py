"""Clarifier — detect ambiguity + ask clarifying questions.

WHY THIS EXISTS
---------------
Agent thường guess khi user message ambiguous → guess sai → rework.
Ví dụ: "fix the bug" — bug nào? ở đâu? fix kiểu gì?

Top-tier agents detect ambiguity và ask 1 câu clarify trước khi hành động.
Hermes-Omni cần cùng khả năng.

PIPELINE
--------
1. Detect ambiguity signals (heuristic + LLM)
2. Generate clarifying question (1 LLM call)
3. Return question to user (instead of acting on guess)

WHEN TO RUN
-----------
- BEFORE run_cognitive_pipeline (cheaper to ask than to guess wrong)
- Heuristic detection (cheap) → if ambiguous, LLM detection (1 call)
- Only ask if ambiguity_score > threshold (config)

AMBIGUITY SIGNALS
-----------------
- Vague references: "it", "this", "that", "the bug", "the file"
- Missing scope: "fix everything", "improve performance" (improve how much?)
- Multiple interpretations: "deploy the app" (which env? which version?)
- Underspecified: "make it better" (better in what way?)
- Pronoun without antecedent: "should I do it?" (do what?)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Heuristic ambiguity signals
# --------------------------------------------------------------------------- #


_AMBIGUOUS_PATTERNS = [
    # Vague references
    (re.compile(r"\b(it|this|that|these|those)\b", re.IGNORECASE), 0.3, "vague reference"),
    (re.compile(r"\bthe (bug|issue|problem|error|file|thing)\b", re.IGNORECASE), 0.4, "vague 'the X'"),
    # Missing scope
    (re.compile(r"\b(fix|solve|handle)\s+(everything|all|it)\b", re.IGNORECASE), 0.6, "no scope"),
    (re.compile(r"\b(improve|optimize|better|enhance)\b", re.IGNORECASE), 0.5, "improve without metric"),
    (re.compile(r"\bmake\s+it\s+(better|faster|nicer)\b", re.IGNORECASE), 0.6, "make it better"),
    # Multiple interpretations
    (re.compile(r"\bdeploy\b", re.IGNORECASE), 0.4, "deploy (which env?)"),
    (re.compile(r"\btest\b(?!\s+(case|suite|file))", re.IGNORECASE), 0.3, "test (what?)"),
    (re.compile(r"\bupdate\s+(the\s+)?(code|app|server)\b", re.IGNORECASE), 0.4, "update (how?)"),
    # Pronoun without antecedent (heuristic: sentence starts with "should I" or "can you")
    (re.compile(r"^(should i|can you|could you|would you)\b", re.IGNORECASE), 0.3, "request without object"),
    # Very short messages
    (re.compile(r"^.{1,15}$"), 0.4, "very short message"),
]


@dataclass
class AmbiguityAssessment:
    """Result of ambiguity detection."""

    is_ambiguous: bool
    ambiguity_score: float  # 0.0 to 1.0
    signals: list[str] = field(default_factory=list)
    clarifying_question: str = ""
    method: Literal["heuristic", "llm", "none"] = "none"


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_DETECT_AMBIGUITY_SYSTEM = (
    "You are the ambiguity-detection layer of an AI agent. Decide if the "
    "user's message is ambiguous enough to warrant a clarifying question "
    "before acting.\n\n"
    "Ambiguous if:\n"
    "- Missing scope/quantity (e.g., 'improve performance' — by how much?)\n"
    "- Vague references (e.g., 'fix the bug' — which bug?)\n"
    "- Multiple valid interpretations (e.g., 'deploy' — to which env?)\n"
    "- Missing context the agent cannot infer\n\n"
    "NOT ambiguous if:\n"
    "- Clear and specific\n"
    "- Context from previous messages makes it clear\n"
    "- It's a follow-up that references prior turn\n\n"
    "Return STRICT JSON:\n"
    "{\n"
    '  "is_ambiguous": true | false,\n'
    '  "ambiguity_score": 0.0 to 1.0,\n'
    '  "signals": ["reason 1", "reason 2"],\n'
    '  "clarifying_question": "if ambiguous, the single best question to ask; empty otherwise"\n'
    "}"
)


# --------------------------------------------------------------------------- #
# Clarifier
# --------------------------------------------------------------------------- #


class Clarifier:
    """Detect ambiguity and generate clarifying questions."""

    def __init__(
        self,
        *,
        llm_call=None,
        heuristic_threshold: float = 0.5,
        llm_threshold: float = 0.4,
        always_use_llm: bool = False,
    ) -> None:
        self._llm_call = llm_call
        self._heuristic_threshold = max(0.0, min(heuristic_threshold, 1.0))
        self._llm_threshold = max(0.0, min(llm_threshold, 1.0))
        self._always_use_llm = always_use_llm

    def assess(self, *, user_message: str, context: str = "") -> AmbiguityAssessment:
        """Assess if a message is ambiguous. Returns assessment."""
        if not user_message or not user_message.strip():
            return AmbiguityAssessment(
                is_ambiguous=False, ambiguity_score=0.0, method="none"
            )
        # Phase 1: heuristic detection (free).
        score, signals = self._heuristic_detect(user_message)
        if score >= self._heuristic_threshold:
            # Heuristic says ambiguous — confirm with LLM if available.
            if self._llm_call is not None:
                llm_result = self._llm_assess(user_message, context)
                if llm_result is not None:
                    return llm_result
            # No LLM — use heuristic result, generate generic question.
            return AmbiguityAssessment(
                is_ambiguous=True,
                ambiguity_score=score,
                signals=signals,
                clarifying_question=self._generate_generic_question(signals, user_message),
                method="heuristic",
            )
        # Phase 2: if always_use_llm or score is borderline, run LLM.
        if self._always_use_llm and self._llm_call is not None:
            llm_result = self._llm_assess(user_message, context)
            if llm_result is not None:
                return llm_result
        # Not ambiguous.
        return AmbiguityAssessment(
            is_ambiguous=False,
            ambiguity_score=score,
            signals=signals,
            method="heuristic",
        )

    def _heuristic_detect(self, message: str) -> tuple[float, list[str]]:
        """Cheap regex-based detection. Returns (score, signals)."""
        score = 0.0
        signals: list[str] = []
        for pattern, weight, label in _AMBIGUOUS_PATTERNS:
            if pattern.search(message):
                score = max(score, weight)
                signals.append(label)
        return score, signals

    def _llm_assess(self, message: str, context: str) -> AmbiguityAssessment | None:
        """LLM-based detection. Returns None on failure."""
        try:
            user = (
                f"User message:\n{message}\n\n"
                f"Conversation context:\n{context or '(none)'}\n\n"
                "Assess ambiguity. Return JSON now."
            )
            raw = self._llm_call(_DETECT_AMBIGUITY_SYSTEM, user)
            data = self._parse_json(raw)
            if data is None:
                return None
            is_amb = bool(data.get("is_ambiguous", False))
            try:
                score = float(data.get("ambiguity_score", 0.0))
            except (TypeError, ValueError):
                score = 0.5 if is_amb else 0.0
            signals = [str(s).strip() for s in data.get("signals", []) if str(s).strip()]
            question = str(data.get("clarifying_question", "")).strip()
            return AmbiguityAssessment(
                is_ambiguous=is_amb and score >= self._llm_threshold,
                ambiguity_score=max(0.0, min(1.0, score)),
                signals=signals,
                clarifying_question=question if is_amb else "",
                method="llm",
            )
        except Exception:
            return None

    @staticmethod
    def _generate_generic_question(signals: list[str], message: str) -> str:
        """Generate a generic clarifying question from heuristic signals."""
        if not signals:
            return "Could you provide more details about what you'd like me to do?"
        if "vague reference" in signals or "vague 'the X'" in signals:
            return "Could you clarify which specific item you're referring to?"
        if "no scope" in signals:
            return "What specific scope or outcome would you like?"
        if "improve without metric" in signals or "make it better" in signals:
            return "In what specific way would you like it improved, and by how much?"
        if "deploy (which env?)" in signals:
            return "Which environment would you like me to deploy to (staging, production, etc.)?"
        return "Could you provide more context or specifics about your request?"

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

_clarifier: Clarifier | None = None


def get_clarifier() -> Clarifier | None:
    return _clarifier


def configure_clarifier(
    *,
    llm_call=None,
    heuristic_threshold: float = 0.5,
    llm_threshold: float = 0.4,
    always_use_llm: bool = False,
) -> Clarifier | None:
    global _clarifier
    _clarifier = Clarifier(
        llm_call=llm_call,
        heuristic_threshold=heuristic_threshold,
        llm_threshold=llm_threshold,
        always_use_llm=always_use_llm,
    )
    return _clarifier


def assess_ambiguity(*, user_message: str, context: str = "") -> dict[str, Any]:
    """Public API: assess if a message is ambiguous.

    Returns dict with: is_ambiguous, ambiguity_score, signals,
    clarifying_question, method.
    """
    if _clarifier is None:
        return {"enabled": False, "is_ambiguous": False}
    result = _clarifier.assess(user_message=user_message, context=context)
    return {
        "enabled": True,
        "is_ambiguous": result.is_ambiguous,
        "ambiguity_score": result.ambiguity_score,
        "signals": result.signals,
        "clarifying_question": result.clarifying_question,
        "method": result.method,
    }
