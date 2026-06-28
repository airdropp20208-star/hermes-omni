"""Guardian Agent — Pre-execution risk review and safety gate.

Inspired by OmniAgent's Guardian, adapted for hermes-omni.

Features:
1. Pre-execution review — catch common mistakes before they happen
2. Risk annotation — label operation risk level (low/medium/high/critical)
3. Self-correction suggestions — provide specific fix advice, not just blocks
4. Four-layer dynamic security scanning:
   - Layer 1: Pattern-based risk detection (fast, no LLM)
   - Layer 2: LLM-powered intelligent review (slower, more accurate)
   - Layer 3: Interactive approval (user confirmation for high-risk)
   - Layer 4: Execution sandbox (isolate dangerous operations)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Risk Patterns ──────────────────────────────────────────────────

_HIGH_RISK_BASH_PATTERNS = [
    (r"\brm\s+(-[rfRF]+\s+)?/([\s;|&]|$)", "Removing root filesystem", "critical"),
    (r"\brm\s+(-[rfRF]+\s+)?~([\s;|&]|$)", "Removing home directory", "critical"),
    (r"\bsudo\s+", "Elevated privileges", "high"),
    (r"\bchmod\s+[0-7]{3,4}\s+/([\s;|&]|$)", "Changing root permissions", "critical"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "System shutdown/reboot", "critical"),
    (r"\bdd\s+.*of=/dev/", "Direct device write", "critical"),
    (r"\bmkfs\b", "Filesystem formatting", "critical"),
    (r"\b(iptables|ufw)\b", "Firewall modification", "high"),
    (r"\b(curl|wget)\b.*\|\s*(bash|sh)\b", "Piped remote script execution", "high"),
    (r"\bpython\s+-c\s+.*(?:import\s+os|subprocess|shutil)", "Dynamic Python execution", "medium"),
    (r"\b(git\s+push\s+--force|git\s+reset\s+--hard)\b", "Destructive git operation", "high"),
    (r"\bfind\b.*-exec\s+rm\b", "find + rm combination", "high"),
    (r"\brm\s+-rf\s+", "Recursive force delete", "high"),
    (r"\beval\s+\$", "Eval with variable expansion", "high"),
    (r"\bnc\s+.*-e\s+", "Netcat with exec", "critical"),
    (r"\bcrontab\s+-r\b", "Remove all cron jobs", "high"),
]

_WRITE_TOOLS = {"write_file", "edit_file", "save_json", "bash", "Write", "Edit"}


@dataclass
class ReviewResult:
    """Result of reviewing a pending tool call."""

    risk_level: str = "low"  # "low" | "medium" | "high" | "critical"
    passed: bool = True
    findings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "passed": self.passed,
            "findings": self.findings,
            "suggestions": self.suggestions,
        }


class PatternScanner:
    """Layer 1: Fast pattern-based risk detection."""

    def scan_bash(self, command: str) -> Tuple[str, List[str]]:
        """Scan a bash command for risk patterns.

        Returns:
            (risk_level, findings) — "low" if no patterns matched
        """
        findings = []
        max_risk = "low"

        risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        for pattern, description, risk in _HIGH_RISK_BASH_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                findings.append(f"{description}: `{command[:80]}`")
                if risk_order.get(risk, 0) > risk_order.get(max_risk, 0):
                    max_risk = risk

        return max_risk, findings

    def scan_write_operation(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Scan write operations for risk patterns."""
        findings = []
        max_risk = "low"

        if tool_name in ("write_file", "Write"):
            path = params.get("path", "")
            content = params.get("content", "")

            # Check for writing to sensitive paths
            sensitive_paths = [
                "/etc/passwd", "/etc/shadow", "/etc/sudoers",
                "/.ssh/", "/.aws/", "/.kube/",
                "~/.bashrc", "~/.zshrc", "~/.profile",
            ]
            for sp in sensitive_paths:
                if sp in path:
                    findings.append(f"Writing to sensitive path: {path}")
                    max_risk = "high"
                    break

            # Check for writing dangerous content
            dangerous_patterns = [
                (r"eval\s+", "Contains eval"),
                (r"exec\s*\(", "Contains exec"),
                (r"subprocess\.(?:call|run|Popen)", "Contains subprocess"),
                (r"os\.system\s*\(", "Contains os.system"),
                (r"__import__\s*\(", "Contains dynamic import"),
            ]
            for pattern, desc in dangerous_patterns:
                if re.search(pattern, content):
                    findings.append(f"Dangerous content: {desc}")
                    if max_risk == "low":
                        max_risk = "medium"

        elif tool_name in ("edit_file", "Edit"):
            path = params.get("path", "")
            # Check for editing sensitive files
            sensitive_patterns = [
                r"/etc/", r"\.ssh/", r"\.aws/", r"\.kube/",
                r"pyproject\.toml$", r"package\.json$",
            ]
            for sp in sensitive_patterns:
                if re.search(sp, path):
                    findings.append(f"Editing sensitive file: {path}")
                    max_risk = "medium"
                    break

        return max_risk, findings


class GuardianAgent:
    """Pre-execution risk review and safety gate.

    Usage:
        guardian = GuardianAgent()

        # Before executing a tool call
        result = await guardian.review("bash", {"command": "rm -rf /tmp/junk"})
        if not result.passed and result.risk_level == "critical":
            print("BLOCKED:", result.findings)
        elif result.findings:
            print("WARNING:", result.suggestions)
    """

    def __init__(self, llm_call_fn=None, auto_block_critical: bool = True):
        """
        Args:
            llm_call_fn: Optional async function for LLM-powered review
            auto_block_critical: Auto-block critical risk operations
        """
        self.pattern_scanner = PatternScanner()
        self._llm_call = llm_call_fn
        self.auto_block_critical = auto_block_critical
        self._session_operations: List[Dict[str, Any]] = []

    async def review(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: str = "",
    ) -> ReviewResult:
        """Review a tool call before execution.

        Runs Layer 1 (pattern scan) always, Layer 2 (LLM review) optionally.

        Returns:
            ReviewResult with risk level, findings, and suggestions.
        """
        findings = []
        suggestions = []
        max_risk = "low"

        # ── Layer 1: Pattern-based scan ──
        if tool_name == "bash":
            command = params.get("command", "")
            risk, bash_findings = self.pattern_scanner.scan_bash(command)
            findings.extend(bash_findings)
            max_risk = self._higher_risk(max_risk, risk)

        elif tool_name in _WRITE_TOOLS:
            risk, write_findings = self.pattern_scanner.scan_write_operation(
                tool_name, params
            )
            findings.extend(write_findings)
            max_risk = self._higher_risk(max_risk, risk)

        # ── Layer 2: LLM-powered review (for high-impact operations) ──
        if max_risk in ("high", "critical") and self._llm_call:
            llm_result = await self._llm_review(tool_name, params, context)
            if llm_result:
                if llm_result.get("risk_level"):
                    max_risk = self._higher_risk(max_risk, llm_result["risk_level"])
                if llm_result.get("findings"):
                    findings.extend(llm_result["findings"])
                if llm_result.get("suggestions"):
                    suggestions.extend(llm_result["suggestions"])

        # ── Determine pass/fail ──
        passed = True
        if max_risk == "critical" and self.auto_block_critical:
            passed = False
        elif max_risk == "high":
            # High risk: warn but don't auto-block (user can approve)
            suggestions.append(
                "⚠️ High-risk operation. Consider if this is really necessary."
            )

        # Record operation
        self._session_operations.append({
            "tool": tool_name,
            "params_summary": str(params)[:200],
            "risk_level": max_risk,
            "findings_count": len(findings),
            "passed": passed,
        })

        return ReviewResult(
            risk_level=max_risk,
            passed=passed,
            findings=findings,
            suggestions=suggestions,
        )

    async def _llm_review(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: str,
    ) -> Optional[Dict[str, Any]]:
        """Layer 2: LLM-powered intelligent review."""
        if not self._llm_call:
            return None

        prompt = f"""Review this tool call for safety risks:

Tool: {tool_name}
Parameters: {json.dumps(params, ensure_ascii=False)[:500]}
Context: {context[:500] if context else "No context"}

Respond in JSON only:
{{
  "risk_level": "low|medium|high|critical",
  "findings": ["list of concerns"],
  "suggestions": ["list of safety suggestions"]
}}"""

        try:
            response = await self._llm_call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            content = (getattr(response, "content", None) or "").strip()
            json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.debug("guardian_llm_review_failed: %s", e)

        return None

    @staticmethod
    def _higher_risk(a: str, b: str) -> str:
        """Return the higher of two risk levels."""
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return a if order.get(a, 0) >= order.get(b, 0) else b

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of all reviewed operations this session."""
        if not self._session_operations:
            return {}

        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        blocked_count = 0
        for op in self._session_operations:
            risk = op.get("risk_level", "low")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
            if not op.get("passed", True):
                blocked_count += 1

        return {
            "total_operations": len(self._session_operations),
            "risk_counts": risk_counts,
            "blocked_count": blocked_count,
        }
