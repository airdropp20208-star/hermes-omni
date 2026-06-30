"""Decision Framework — classifies actions by consequence level.

The classifier is the first gate in the reasoning protocol: it decides HOW MUCH
reasoning an action deserves before the agent is allowed to execute it.

Design principles
-----------------
1. **Cheap first, expensive last.** Classification is pure-Python heuristics
   (regex + tool-name matching). It runs in microseconds and never blocks.
2. **Read-only is free.** Pure inspection tools (read_file, ls, grep, list)
   are TRIVIAL — no plan needed. The LLM's intelligence is not bottlenecked
   by framework overhead for safe operations.
3. **Side-effects escalate.** Anything that writes, executes, sends, or
   deletes is at least STANDARD. Anything irreversible (rm -rf, force push,
   drop table, send email to many recipients) is CONSEQUENTIAL or
   IRREVERSIBLE.
4. **Conservative on unknowns.** If we cannot prove an action is safe, we
   treat it as STANDARD so the reasoning protocol at least runs a brief
   plan. False-positive caution is cheap; false-negative recklessness is
   expensive.

The four decision classes map to behavior in ReasoningProtocol:

    TRIVIAL        → execute immediately, no plan
    STANDARD       → brief plan (goal + approach), execute
    CONSEQUENTIAL  → full plan (goal + approach + risks + reversibility),
                     self-critique, execute if sound
    IRREVERSIBLE   → full plan + critique + explicit acknowledgement of
                     irreversibility. Auto-execute only if user has trusted
                     the agent (config flag); otherwise surface to user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class DecisionClass(IntEnum):
    """How much reasoning an action deserves. Higher = more reasoning."""

    TRIVIAL = 0
    STANDARD = 10
    CONSEQUENTIAL = 20
    IRREVERSIBLE = 30


# Tool-name patterns that are pure read-only (no state change).
_READ_ONLY_TOOL_PATTERNS = (
    re.compile(r"^read", re.IGNORECASE),
    re.compile(r"^list", re.IGNORECASE),
    re.compile(r"^ls$", re.IGNORECASE),
    re.compile(r"^grep", re.IGNORECASE),
    re.compile(r"^glob", re.IGNORECASE),
    re.compile(r"^find", re.IGNORECASE),
    re.compile(r"^stat", re.IGNORECASE),
    re.compile(r"^head", re.IGNORECASE),
    re.compile(r"^tail", re.IGNORECASE),
    re.compile(r"^cat", re.IGNORECASE),
    re.compile(r"^wc", re.IGNORECASE),
    re.compile(r"^unified_recall", re.IGNORECASE),
    re.compile(r"^unified_reflexion_list", re.IGNORECASE),
    re.compile(r"^unified_framework_status", re.IGNORECASE),
    re.compile(r"^reasoning_", re.IGNORECASE),
    re.compile(r"search", re.IGNORECASE),  # web_search, file_search, tool_search
    re.compile(r"status", re.IGNORECASE),
)

# Tool-name patterns that always execute something (commands, code).
_EXECUTE_TOOL_PATTERNS = (
    re.compile(r"^bash", re.IGNORECASE),
    re.compile(r"^sh$", re.IGNORECASE),
    re.compile(r"^shell", re.IGNORECASE),
    re.compile(r"^execute", re.IGNORECASE),
    re.compile(r"^terminal", re.IGNORECASE),
    re.compile(r"^run", re.IGNORECASE),
    re.compile(r"^subprocess", re.IGNORECASE),
)

# Tool-name patterns that send data outside the sandbox (irreversible-ish).
_SEND_TOOL_PATTERNS = (
    re.compile(r"^send", re.IGNORECASE),
    re.compile(r"^publish", re.IGNORECASE),
    re.compile(r"^deploy", re.IGNORECASE),
    re.compile(r"^push", re.IGNORECASE),
    re.compile(r"^email", re.IGNORECASE),
    re.compile(r"^message", re.IGNORECASE),
    re.compile(r"^notify", re.IGNORECASE),
)

# Argument-value regex patterns that signal IRREVERSIBLE operations
# regardless of tool name. These are the catastrophic patterns that
# unlimited mode disabled — we re-enable them as *classification signals*
# (not blocks). The LLM is given a chance to justify them; if it can't,
# the framework asks the user.
_IRREVERSIBLE_VALUE_PATTERNS = (
    re.compile(r"\brm\s+-rf?\s+/(?:\s|$|[^/\s])", re.IGNORECASE),  # rm -rf /<path>
    re.compile(r"\bmkfs\.", re.IGNORECASE),
    re.compile(r"\bdd\s+if=.*of=/dev/", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;:", re.IGNORECASE),  # fork bomb
    re.compile(r"\bchmod\s+-R\s+[0-7]{3,4}\s+/(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bchown\s+-R\s+\S+\s+/(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bgit\s+push\s+.*--force", re.IGNORECASE),
    re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\s+(TABLE|DATABASE)\b", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\s+\w+\s*;(?:\s|$)", re.IGNORECASE),  # unfiltered DELETE
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bhalt\b", re.IGNORECASE),
    re.compile(r"\bpoweroff\b", re.IGNORECASE),
)

# Argument-value patterns that signal CONSEQUENTIAL (significant but
# potentially reversible) operations.
_CONSEQUENTIAL_VALUE_PATTERNS = (
    re.compile(r"\brm\s+-r", re.IGNORECASE),  # rm -r but not -rf /<root>
    re.compile(r"\bgit\s+push\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bapt(?:-get)?\s+install", re.IGNORECASE),
    re.compile(r"\bpip\s+install", re.IGNORECASE),
    re.compile(r"\bnpm\s+install", re.IGNORECASE),
    re.compile(r"\buv\s+(pip\s+)?install", re.IGNORECASE),
    re.compile(r"\bCREATE\s+(TABLE|DATABASE|INDEX)\b", re.IGNORECASE),
    re.compile(r"\bALTER\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+\w+\s+SET\b", re.IGNORECASE),
    re.compile(r"\bwget\b", re.IGNORECASE),
    re.compile(r"\bcurl\s+.*\|\s*(?:sh|bash)", re.IGNORECASE),  # pipe to shell
)


@dataclass(frozen=True)
class Classification:
    """Result of classifying a tool call."""

    cls: DecisionClass
    reason: str
    matched_pattern: str = ""

    @property
    def is_trivial(self) -> bool:
        return self.cls is DecisionClass.TRIVIAL

    @property
    def requires_plan(self) -> bool:
        return self.cls >= DecisionClass.STANDARD

    @property
    def requires_critique(self) -> bool:
        return self.cls >= DecisionClass.CONSEQUENTIAL

    @property
    def requires_acknowledgement(self) -> bool:
        return self.cls is DecisionClass.IRREVERSIBLE


class DecisionFramework:
    """Classify a tool call into a DecisionClass.

    Stateless and side-effect free. Safe to call from any thread.
    """

    def classify(self, tool_name: str, args: dict[str, Any] | None) -> Classification:
        name = (tool_name or "").strip()
        safe_args = args if isinstance(args, dict) else {}

        # 1. Check argument values for IRREVERSIBLE patterns first — these
        #    override everything else because the value itself is the risk,
        #    not the tool name.
        for key, value in safe_args.items():
            text = self._stringify(value)
            if not text:
                continue
            for pattern in _IRREVERSIBLE_VALUE_PATTERNS:
                match = pattern.search(text)
                if match:
                    return Classification(
                        cls=DecisionClass.IRREVERSIBLE,
                        reason=f"Argument {key!r} matched irreversible pattern: {pattern.pattern}",
                        matched_pattern=pattern.pattern,
                    )

        # 2. Check argument values for CONSEQUENTIAL patterns.
        for key, value in safe_args.items():
            text = self._stringify(value)
            if not text:
                continue
            for pattern in _CONSEQUENTIAL_VALUE_PATTERNS:
                match = pattern.search(text)
                if match:
                    return Classification(
                        cls=DecisionClass.CONSEQUENTIAL,
                        reason=f"Argument {key!r} matched consequential pattern: {pattern.pattern}",
                        matched_pattern=pattern.pattern,
                    )

        # 3. Tool-name based classification.
        # 3a. Send/publish/deploy tools → CONSEQUENTIAL (external side effects).
        for pattern in _SEND_TOOL_PATTERNS:
            if pattern.search(name):
                return Classification(
                    cls=DecisionClass.CONSEQUENTIAL,
                    reason=f"Tool name {name!r} matches send/deploy pattern",
                    matched_pattern=pattern.pattern,
                )

        # 3b. Execute/bash/run tools → STANDARD (unless args escalated above).
        for pattern in _EXECUTE_TOOL_PATTERNS:
            if pattern.search(name):
                return Classification(
                    cls=DecisionClass.STANDARD,
                    reason=f"Tool name {name!r} matches execute pattern",
                    matched_pattern=pattern.pattern,
                )

        # 3c. Read-only tools → TRIVIAL.
        for pattern in _READ_ONLY_TOOL_PATTERNS:
            if pattern.search(name):
                return Classification(
                    cls=DecisionClass.TRIVIAL,
                    reason=f"Tool name {name!r} matches read-only pattern",
                    matched_pattern=pattern.pattern,
                )

        # 4. Default: STANDARD. Conservative — better to plan once too many
        #    than once too few. The plan is cheap (single LLM call) and the
        #    LLM can self-short-circuit if the action is obviously safe.
        return Classification(
            cls=DecisionClass.STANDARD,
            reason="Unknown tool — defaulting to STANDARD (brief plan)",
        )

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        try:
            import json

            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
