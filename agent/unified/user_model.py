"""User Model — build a profile of the user for personalization.

WHY THIS EXISTS
---------------
Hermes treats every user the same. Top-tier assistants (Claude, GLM)
adapt to user expertise/style over time. A senior dev wants terse
technical answers; a beginner wants explanations. Without user modeling,
the agent either over-explains (annoying pros) or under-explains
(confusing beginners).

This module builds a profile from:
1. Explicit preferences (user tells agent: "I prefer concise answers")
2. Behavioral signals (correction frequency, question depth, vocabulary)
3. Recurring request patterns (always asks about Python, never about JS)

The profile is injected into the system prompt so the agent adapts.

PROFILE STRUCTURE
----------------
- expertise_level: "beginner" | "intermediate" | "advanced" | "expert"
- communication_style: "concise" | "detailed" | "technical" | "casual"
- preferred_language: "en" | "vi" | "zh" | ...
- domains: ["python", "kubernetes", "machine_learning", ...]
- recurring_requests: ["debug", "explain", "deploy", ...]
- correction_count: int (how often user corrects agent)
- avg_message_length: int
- preferences: list[str] (explicit user-stated preferences)

PERSISTENCE
-----------
Profile stored at ~/.hermes/unified/user_profile.json. Updated
continuously as agent observes user behavior.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class UserProfile:
    """User profile for personalization."""

    expertise_level: Literal["beginner", "intermediate", "advanced", "expert"] = "intermediate"
    communication_style: Literal["concise", "detailed", "technical", "casual"] = "technical"
    preferred_language: str = ""
    domains: list[str] = field(default_factory=list)  # ["python", "kubernetes", ...]
    recurring_requests: list[str] = field(default_factory=list)  # ["debug", "deploy", ...]
    correction_count: int = 0
    total_messages: int = 0
    avg_message_length: int = 0
    preferences: list[str] = field(default_factory=list)  # explicit
    last_updated: float = field(default_factory=time.time)
    # Internal counters (not serialized as preferences).
    _domain_counter: Counter = field(default_factory=Counter)
    _request_counter: Counter = field(default_factory=Counter)
    _message_lengths: list[int] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Render as a system prompt block."""
        if self.total_messages < 3:
            # Not enough data — don't inject (avoid premature assumptions).
            return ""
        lines = ["<user-profile>"]
        lines.append(f"Expertise: {self.expertise_level}")
        lines.append(f"Communication style: {self.communication_style}")
        if self.preferred_language:
            lines.append(f"Preferred language: {self.preferred_language}")
        if self.domains:
            lines.append(f"Domains: {', '.join(self.domains[:5])}")
        if self.recurring_requests:
            lines.append(f"Common requests: {', '.join(self.recurring_requests[:5])}")
        if self.preferences:
            lines.append("Explicit preferences:")
            for p in self.preferences[:5]:
                lines.append(f"  - {p}")
        lines.append(f"Adapt your responses to this user profile.</user-profile>")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Domain + request detection heuristics
# --------------------------------------------------------------------------- #


_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "python": {"python", "pip", "django", "flask", "fastapi", "pytest", "pandas", "numpy"},
    "javascript": {"javascript", "js", "node", "npm", "react", "vue", "angular", "typescript"},
    "rust": {"rust", "cargo", "tokio", "serde"},
    "go": {"golang", "go ", "goroutine", "channel"},
    "kubernetes": {"kubernetes", "k8s", "kubectl", "helm", "pod"},
    "docker": {"docker", "container", "dockerfile", "compose"},
    "machine_learning": {"ml", "ai", "neural", "training", "model", "pytorch", "tensorflow"},
    "web": {"html", "css", "http", "rest", "api", "frontend"},
    "database": {"sql", "postgres", "mysql", "sqlite", "mongodb", "query"},
    "devops": {"ci", "cd", "pipeline", "deploy", "ansible", "terraform"},
    "security": {"security", "vulnerability", "cve", "exploit", "encryption"},
    "git": {"git", "commit", "branch", "merge", "rebase", "pr", "pull request"},
}

_REQUEST_KEYWORDS: dict[str, set[str]] = {
    "debug": {"debug", "error", "bug", "fix", "broken", "fail", "crash", "traceback"},
    "explain": {"explain", "how", "why", "what", "understand", "elaborate"},
    "deploy": {"deploy", "ship", "release", "publish", "rollout"},
    "refactor": {"refactor", "clean", "restructure", "simplify"},
    "test": {"test", "pytest", "unittest", "coverage", "tdd"},
    "review": {"review", "feedback", "critique", "check"},
    "create": {"create", "build", "make", "generate", "scaffold"},
    "document": {"document", "docs", "readme", "explain"},
    "optimize": {"optimize", "perf", "performance", "speed", "fast"},
    "learn": {"learn", "teach", "tutorial", "guide", "example"},
}


def _detect_domains(message: str) -> list[str]:
    """Detect domain keywords in a message."""
    msg_lower = message.lower()
    found = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            found.append(domain)
    return found


def _detect_requests(message: str) -> list[str]:
    """Detect request types in a message."""
    msg_lower = message.lower()
    found = []
    for req, keywords in _REQUEST_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            found.append(req)
    return found


_EXPERTISE_SIGNALS = {
    "beginner": {"help", "how do i", "what is", "explain like", "i'm new", "i am new", "beginner"},
    "expert": {"architecture", "optimization", "internals", "edge case", "race condition",
                "memory leak", "concurrency", "distributed", "byzantine"},
}


def _estimate_expertise(message: str, current: str) -> str:
    """Estimate expertise from message. Returns current if no signal."""
    msg_lower = message.lower()
    for level, signals in _EXPERTISE_SIGNALS.items():
        if any(s in msg_lower for s in signals):
            return level
    return current


# --------------------------------------------------------------------------- #
# UserModel
# --------------------------------------------------------------------------- #


class UserModel:
    """Builds and maintains a user profile."""

    def __init__(self, *, profile_path: str | Path | None = None) -> None:
        if profile_path is None:
            from hermes_constants import get_hermes_home

            profile_path = get_hermes_home() / "unified" / "user_profile.json"
        self._path = Path(profile_path).expanduser()
        self._profile = self._load()

    def _load(self) -> UserProfile:
        if not self._path.exists():
            return UserProfile()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            # Filter to allowed fields.
            allowed = UserProfile.__dataclass_fields__.keys()
            data = {k: v for k, v in data.items() if k in allowed}
            # Convert _domain_counter / _request_counter back to Counter.
            data["_domain_counter"] = Counter(data.get("_domain_counter", {}))
            data["_request_counter"] = Counter(data.get("_request_counter", {}))
            return UserProfile(**data)
        except Exception:
            return UserProfile()

    def _save(self) -> None:
        try:
            self._profile.last_updated = time.time()
            data = asdict(self._profile)
            # Convert Counter to dict for JSON.
            data["_domain_counter"] = dict(data.get("_domain_counter", {}))
            data["_request_counter"] = dict(data.get("_request_counter", {}))
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def observe_message(self, message: str) -> None:
        """Update profile based on a user message."""
        if not message or not message.strip():
            return
        p = self._profile
        p.total_messages += 1
        # Message length tracking.
        p._message_lengths.append(len(message))
        if len(p._message_lengths) > 100:
            p._message_lengths = p._message_lengths[-50:]
        p.avg_message_length = sum(p._message_lengths) // len(p._message_lengths)
        # Domain + request detection.
        for domain in _detect_domains(message):
            p._domain_counter[domain] += 1
        for req in _detect_requests(message):
            p._request_counter[req] += 1
        # Update top domains/requests.
        p.domains = [d for d, _ in p._domain_counter.most_common(10)]
        p.recurring_requests = [r for r, _ in p._request_counter.most_common(10)]
        # Expertise estimation.
        p.expertise_level = _estimate_expertise(message, p.expertise_level)  # type: ignore[assignment]
        # Communication style: long messages → detailed, short → concise.
        if p.avg_message_length > 500:
            p.communication_style = "detailed"  # type: ignore[assignment]
        elif p.avg_message_length < 50 and p.total_messages > 5:
            p.communication_style = "concise"  # type: ignore[assignment]
        self._save()

    def record_correction(self, correction_text: str = "") -> None:
        """Record that the user corrected the agent."""
        self._profile.correction_count += 1
        # Extract explicit preference if present.
        # Heuristic: "I prefer X", "I want X", "don't X", "always X"
        m = re.search(
            r"(?:i prefer|i want|i like|always|never|don't|do not)\s+(.+?)(?:[.,\n]|$)",
            correction_text,
            re.IGNORECASE,
        )
        if m:
            pref = m.group(1).strip()[:200]
            if pref and pref not in self._profile.preferences:
                self._profile.preferences.append(pref)
                if len(self._profile.preferences) > 20:
                    self._profile.preferences = self._profile.preferences[-20:]
        self._save()

    def set_explicit_preference(self, preference: str) -> None:
        """User explicitly states a preference."""
        preference = preference.strip()
        if preference and preference not in self._profile.preferences:
            self._profile.preferences.append(preference)
            self._save()

    def get_profile(self) -> UserProfile:
        return self._profile

    def get_prompt_block(self) -> str:
        """Return profile as system prompt block (empty if insufficient data)."""
        return self._profile.to_prompt_block()

    def reset(self) -> None:
        """Clear profile (start fresh)."""
        self._profile = UserProfile()
        self._save()


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_model: UserModel | None = None


def get_user_model() -> UserModel | None:
    return _model


def configure_user_model(*, profile_path: str | Path | None = None) -> UserModel:
    global _model
    _model = UserModel(profile_path=profile_path)
    return _model


def observe_user_message(message: str) -> None:
    """Public API: observe a user message for profile building."""
    if _model is None:
        return
    _model.observe_message(message)


def record_user_correction(correction_text: str = "") -> None:
    """Public API: record that user corrected the agent."""
    if _model is None:
        return
    _model.record_correction(correction_text)


def get_user_profile_block() -> str:
    """Public API: get profile as system prompt block."""
    if _model is None:
        return ""
    return _model.get_prompt_block()


def user_model_stats() -> dict[str, Any]:
    """Public API: get profile stats."""
    if _model is None:
        return {"enabled": False}
    p = _model.get_profile()
    return {
        "enabled": True,
        "total_messages": p.total_messages,
        "expertise_level": p.expertise_level,
        "communication_style": p.communication_style,
        "domains": p.domains[:5],
        "recurring_requests": p.recurring_requests[:5],
        "correction_count": p.correction_count,
        "preferences_count": len(p.preferences),
    }
