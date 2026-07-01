"""Guardian policy engine for unified Hermes.

The engine combines OmniAgent-style guard/sentinel ideas with AgentScope-style
permission checks. It is deliberately declarative and side-effect free so it can
run from Hermes' existing plugin hook before any tool is executed.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


@dataclass(frozen=True)
class PolicyRule:
    """One guardian rule.

    ``tool`` supports shell wildcards. ``arg_patterns`` maps argument names to
    regular expressions. A missing argument never matches unless the regex is
    ``.*`` and the value is coerced to an empty string.
    """

    name: str
    decision: Decision
    message: str
    tool: str = "*"
    arg_patterns: dict[str, str] = field(default_factory=dict)
    priority: int = 100

    def matches(self, tool_name: str, args: dict[str, Any]) -> bool:
        if not fnmatch.fnmatchcase(tool_name, self.tool):
            return False
        for key, pattern in self.arg_patterns.items():
            value = args.get(key, "")
            if not re.search(pattern, str(value), flags=re.IGNORECASE | re.MULTILINE):
                return False
        return True


@dataclass(frozen=True)
class GuardianPolicy:
    """Result of evaluating a tool call."""

    decision: Decision
    message: str = ""
    rule: str = ""

    @property
    def blocked(self) -> bool:
        return self.decision == Decision.BLOCK


class PolicyEngine:
    """Evaluate policy rules in priority order."""

    def __init__(self, rules: Iterable[PolicyRule] = ()) -> None:
        self.rules = sorted(list(rules), key=lambda item: item.priority)

    @classmethod
    def default(cls) -> "PolicyEngine":
        """Conservative defaults for destructive code/shell/file operations."""
        destructive = r"\b(rm\s+-rf\s+/(?:\s|$)|mkfs\.|dd\s+if=|:(){:|:&};:|chmod\s+-R\s+777\s+/|chown\s+-R\s+[^\s]+\s+/)"
        secret_paths = r"(^|/)(\.ssh|\.gnupg|\.aws|\.config/gcloud|\.netrc|\.git-credentials)(/|$)"
        return cls(
            [
                PolicyRule(
                    name="block-catastrophic-shell",
                    decision=Decision.BLOCK,
                    tool="execute_code",
                    arg_patterns={"code": destructive},
                    message="Unified Guardian blocked a catastrophic shell command.",
                    priority=10,
                ),
                PolicyRule(
                    name="block-catastrophic-bash",
                    decision=Decision.BLOCK,
                    tool="bash*",
                    arg_patterns={"command": destructive},
                    message="Unified Guardian blocked a catastrophic shell command.",
                    priority=10,
                ),
                PolicyRule(
                    name="warn-secret-path-read",
                    decision=Decision.WARN,
                    tool="*read*",
                    arg_patterns={"path": secret_paths},
                    message="Reading a sensitive credential path; redact before sharing results.",
                    priority=50,
                ),
            ]
        )

    @classmethod
    def from_patterns(cls, patterns: Iterable[str] = ()) -> "PolicyEngine":
        engine = cls.default()
        extra = []
        for idx, pattern in enumerate(part.strip() for part in patterns if str(part).strip()):
            extra.append(
                PolicyRule(
                    name=f"configured-block-tool-{idx}",
                    decision=Decision.BLOCK,
                    tool=pattern,
                    message=f"Tool '{pattern}' is blocked by unified guardian configuration.",
                    priority=5,
                )
            )
        return cls([*extra, *engine.rules])

    @classmethod
    def from_env(cls) -> "PolicyEngine":
        raw = os.getenv("HERMES_UNIFIED_BLOCK_TOOLS", "").strip()
        patterns = [part.strip() for part in raw.split(",") if part.strip()] if raw else []
        return cls.from_patterns(patterns)

    def evaluate(self, tool_name: str, args: dict[str, Any] | None) -> GuardianPolicy:
        safe_args = args if isinstance(args, dict) else {}
        best_warning: GuardianPolicy | None = None
        for rule in self.rules:
            if not rule.matches(tool_name, safe_args):
                continue
            policy = GuardianPolicy(rule.decision, rule.message, rule.name)
            if policy.decision == Decision.BLOCK:
                return policy
            if policy.decision == Decision.WARN and best_warning is None:
                best_warning = policy
        return best_warning or GuardianPolicy(Decision.ALLOW)
