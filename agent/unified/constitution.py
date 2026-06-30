"""Constitutional Principles — value alignment layer.

DIFFERENTIATOR vs TOP-TIER MODELS
---------------------------------
Claude has Constitutional AI baked in at TRAIN time (fixed principles).
GLM has Chinese-regulatory alignment baked in.

Hermes-Omni lets the USER define principles at RUNTIME. This means:
- A doctor can configure "Always cite medical sources; never prescribe"
- A lawyer can configure "Never give legal advice; recommend bar association"
- A developer can configure "Always test code before claiming it works"
- A parent can configure "Refuse to discuss adult topics with minors"

The constitution is a markdown file at ~/.hermes/constitution.md with
~5-10 principles. The Verifier checks BOTH correctness AND alignment.

WHEN IT RUNS
------------
- Integrated into Verifier: each critique pass also checks constitution
- Standalone: agent can call `constitution_check(text)` explicitly
- On session start: principles loaded into system prompt

TOKEN ECONOMICS
---------------
- 0 LLM calls to load principles (file read)
- 1 LLM call per constitution check (batched with verifier critique
  when both enabled — same prompt, just add principles section)

Net: ~0 extra cost when combined with verifier.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Default constitution (user can override)
# --------------------------------------------------------------------------- #

_DEFAULT_CONSTITUTION = """# Hermes Agent Constitution

## Honesty
- Never fabricate facts, citations, or sources.
- If unsure, say "I'm not sure" rather than guessing.
- Distinguish clearly between facts and opinions.

## Helpfulness
- Address the full request, not just the easy parts.
- Provide actionable, specific guidance — not generic platitudes.
- When refusing, explain why and suggest alternatives.

## Harm Prevention
- Do not provide instructions for weapons, drugs, or malware.
- Do not help with attacks on individuals or systems.
- Refuse deceptive requests even when framed as hypothetical.

## Accuracy
- Cite sources for specific factual claims.
- Test code before claiming it works.
- Acknowledge uncertainty in predictions.

## Respect
- Do not discriminate based on protected characteristics.
- Respect user privacy — do not ask for unnecessary personal info.
- Default to formal tone unless user requests otherwise.
"""


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class Principle:
    """One constitutional principle."""

    category: str  # "Honesty", "Helpfulness", etc.
    text: str


@dataclass
class ConstitutionCheck:
    """Result of checking a response against the constitution."""

    violations: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    aligned: bool = True
    score: float = 1.0  # 0.0 (severe violations) to 1.0 (fully aligned)
    explanation: str = ""


# --------------------------------------------------------------------------- #
# Constitution loader
# --------------------------------------------------------------------------- #


def load_constitution(path: str | Path | None = None) -> list[Principle]:
    """Load principles from ~/.hermes/constitution.md (or given path).

    Format: markdown with ## headings as categories, bullet points as
    principles. Falls back to default constitution if file missing.
    """
    if path is None:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "constitution.md"
    path = Path(path).expanduser()
    text = None
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = None
    if not text or not text.strip():
        text = _DEFAULT_CONSTITUTION
    return _parse_constitution(text)


def _parse_constitution(text: str) -> list[Principle]:
    """Parse markdown into Principle list."""
    principles: list[Principle] = []
    current_category = "General"
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("## "):
            current_category = line[3:].strip()
        elif line.strip().startswith("- "):
            principle_text = line.strip()[2:].strip()
            if principle_text:
                principles.append(Principle(category=current_category, text=principle_text))
    if not principles:
        # Fallback: treat whole text as one principle.
        principles.append(Principle(category="General", text="Follow the spirit of the constitution document."))
    return principles


def constitution_to_prompt_block(principles: list[Principle] | None = None) -> str:
    """Render principles as a system prompt block."""
    if principles is None:
        principles = load_constitution()
    if not principles:
        return ""
    lines = ["<constitution>", "You MUST follow these principles:"]
    by_category: dict[str, list[str]] = {}
    for p in principles:
        by_category.setdefault(p.category, []).append(p.text)
    for category, texts in by_category.items():
        lines.append(f"\n{category}:")
        for t in texts:
            lines.append(f"  - {t}")
    lines.append("</constitution>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Constitution checker
# --------------------------------------------------------------------------- #


_CHECK_SYSTEM = (
    "You are the constitution-alignment layer of an AI agent. Check the "
    "agent's response against the constitutional principles. Report:\n"
    "- violations: principles the response clearly breaks\n"
    "- concerns: principles the response might break (borderline)\n"
    "- aligned: true if no violations\n"
    "- score: 1.0 (fully aligned) to 0.0 (severe violations)\n\n"
    "Be strict but reasonable — a response that 'technically' follows a "
    "principle but violates its spirit should still flag a concern.\n\n"
    "Return STRICT JSON:\n"
    "{\n"
    '  "violations": ["principle that was broken"],\n'
    '  "concerns": ["borderline case"],\n'
    '  "aligned": true | false,\n'
    '  "score": 0.0 to 1.0,\n'
    '  "explanation": "one-paragraph summary"\n'
    "}"
)


class ConstitutionChecker:
    """Checks agent responses against constitutional principles."""

    def __init__(
        self,
        *,
        llm_call=None,
        constitution_path: str | Path | None = None,
    ) -> None:
        self._llm_call = llm_call
        self._path = constitution_path
        self._principles: list[Principle] | None = None

    def _get_principles(self) -> list[Principle]:
        if self._principles is None:
            self._principles = load_constitution(self._path)
        return self._principles

    def refresh(self) -> None:
        """Force reload of constitution file."""
        self._principles = load_constitution(self._path)

    def check(self, *, user_request: str, response: str, context: str = "") -> ConstitutionCheck:
        """Check a response against the constitution."""
        if self._llm_call is None:
            return ConstitutionCheck()
        principles = self._get_principles()
        principles_text = constitution_to_prompt_block(principles)
        try:
            user = (
                f"{principles_text}\n\n"
                f"User request:\n{user_request}\n\n"
                f"Agent response to check:\n{response}\n\n"
                f"Context:\n{context or '(none)'}\n\n"
                "Check alignment with the constitution. Return JSON now."
            )
            raw = self._llm_call(_CHECK_SYSTEM, user)
            data = self._parse_json(raw)
            if data is None:
                return ConstitutionCheck()
            violations = [str(s).strip() for s in data.get("violations", []) if str(s).strip()]
            concerns = [str(s).strip() for s in data.get("concerns", []) if str(s).strip()]
            aligned = bool(data.get("aligned", True))
            try:
                score = max(0.0, min(1.0, float(data.get("score", 1.0))))
            except (TypeError, ValueError):
                score = 1.0 if aligned else 0.5
            return ConstitutionCheck(
                violations=violations,
                concerns=concerns,
                aligned=aligned and not violations,
                score=score,
                explanation=str(data.get("explanation", "")).strip(),
            )
        except Exception:
            return ConstitutionCheck()

    def get_prompt_block(self) -> str:
        """Return principles as a prompt block (for system prompt injection)."""
        return constitution_to_prompt_block(self._get_principles())

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

_checker: ConstitutionChecker | None = None


def get_constitution_checker() -> ConstitutionChecker | None:
    return _checker


def configure_constitution_checker(
    *,
    llm_call=None,
    constitution_path: str | Path | None = None,
) -> ConstitutionChecker | None:
    global _checker
    _checker = ConstitutionChecker(
        llm_call=llm_call,
        constitution_path=constitution_path,
    )
    return _checker


def constitution_check(
    *,
    user_request: str,
    response: str,
    context: str = "",
) -> ConstitutionCheck:
    """Public API: check a response against the constitution."""
    if _checker is None:
        return ConstitutionCheck()
    return _checker.check(user_request=user_request, response=response, context=context)


def get_constitution_prompt_block() -> str:
    """Public API: return constitution as a system prompt block."""
    if _checker is None:
        return constitution_to_prompt_block(load_constitution())
    return _checker.get_prompt_block()
