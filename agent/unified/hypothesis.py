"""Hypothesis Engine — hypothesis-test-revise loop for diagnostic tasks.

WHY THIS EXISTS
---------------
CognitiveTree helps the agent pick between approaches BEFORE acting.
But many tasks are diagnostic: "why is the build failing?", "why is
the API returning 500?", "why is the cron job not running?". For these,
the agent doesn't need to pick between plans — it needs to form a
hypothesis, test it, and revise based on evidence.

This is the scientific method applied to agent reasoning:
    1. Observe a symptom
    2. Form a hypothesis (or N hypotheses)
    3. Design a test that would distinguish between them
    4. Run the test
    5. Revise: confirm, refute, or refine
    6. Repeat until confidence is high enough to act

Without this, agents fall into two failure modes:
    A. **Action-first**: jump straight to "fix" without understanding
       the root cause. Fix is wrong, problem persists, agent loops.
    B. **Analysis paralysis**: gather evidence forever, never act.

HypothesisEngine balances these by forcing the agent to commit to a
testable hypothesis, then act on the result.

WHEN IT RUNS
------------
Triggered by the agent explicitly calling `hypothesis_form` (a tool),
OR by the conversation loop detecting a "diagnostic" user message:
- "why ...", "what's wrong with ...", "debug ...", "diagnose ..."
- After 2+ failed attempts at the same task (suggests root cause unknown)

It does NOT run for every action — only diagnostic situations. This
keeps token cost under control.

TOKEN ECONOMICS
---------------
- 1 LLM call to form N hypotheses (batched)
- 0 LLM calls to test (tests are tool calls, which the agent does anyway)
- 1 LLM call to revise after each test
- Typically 2-3 revisions before acting

Total: ~5-8 LLM calls for a diagnostic task, vs. 10-20 for an
action-first agent that keeps retrying the wrong fix.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


class HypothesisStatus(str, Enum):
    UNTESTED = "untested"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    SUPERSEDED = "superseded"  # replaced by a refined hypothesis


@dataclass
class Hypothesis:
    """A testable explanation for an observed symptom."""

    hypothesis_id: str
    statement: str  # "The build fails because of a missing import in utils.py"
    prior_probability: float = 0.5  # 0.0 to 1.0
    posterior_probability: float = 0.5
    status: HypothesisStatus = HypothesisStatus.UNTESTED
    test_design: str = ""  # "grep for the import in utils.py"
    test_result: str = ""  # "import found at line 5"
    test_outcome: Literal["confirm", "refute", "inconclusive"] = "inconclusive"
    created_at: float = field(default_factory=time.time)
    tested_at: float = 0.0
    revisions: int = 0


@dataclass
class DiagnosticSession:
    """A diagnostic investigation. Tracks the full hypothesis-test-revise loop."""

    session_id: str
    symptom: str  # the observed problem
    context: str  # background information
    hypotheses: list[Hypothesis] = field(default_factory=list)
    iterations: int = 0
    final_conclusion: str = ""
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    closed: bool = False

    def active_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.status in (HypothesisStatus.UNTESTED, HypothesisStatus.TESTING)]

    def best_hypothesis(self) -> Hypothesis | None:
        active = self.active_hypotheses()
        if not active:
            confirmed = [h for h in self.hypotheses if h.status == HypothesisStatus.CONFIRMED]
            if confirmed:
                return max(confirmed, key=lambda h: h.posterior_probability)
            return None
        return max(active, key=lambda h: h.posterior_probability)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_FORM_HYPOTHESES_SYSTEM = (
    "You are the diagnostic layer of an AI agent. The agent has observed "
    "a symptom and needs to form testable hypotheses. Generate {n} DISTINCT "
    "hypotheses that explain the symptom. Each must be testable with a "
    "concrete tool call.\n\n"
    "Bayesian prior: rate how likely each hypothesis is BEFORE testing, "
    "based on base rates (common causes are more likely than exotic ones).\n\n"
    "Return STRICT JSON: an array of {n} objects. Schema:\n"
    "{\n"
    '  "statement": "concise hypothesis statement",\n'
    '  "prior_probability": 0.0 to 1.0,\n'
    '  "test_design": "specific tool call that would confirm or refute this",\n'
    '  "expected_if_true": "what the test would show if hypothesis is true",\n'
    '  "expected_if_false": "what the test would show if hypothesis is false"\n'
    "}"
)

_REVISE_SYSTEM = (
    "You are the diagnostic layer of an AI agent. A hypothesis has been "
    "tested. Apply Bayesian updating: given the prior probability and the "
    "test outcome, compute the posterior probability.\n\n"
    "If the test was inconclusive, propose a refined hypothesis with a "
    "more specific test. If the test refuted the hypothesis, propose a "
    "new alternative. If confirmed, set posterior near 1.0.\n\n"
    "Return STRICT JSON:\n"
    "{\n"
    '  "posterior_probability": 0.0 to 1.0,\n'
    '  "interpretation": "what the test result means",\n'
    '  "next_action": "act" | "test_further" | "abandon",\n'
    '  "refined_hypothesis": "if test_further, a new/refined hypothesis; empty otherwise",\n'
    '  "refined_test_design": "if test_further, the next test to run; empty otherwise",\n'
    '  "conclusion": "if act, the conclusion to act on; empty otherwise"\n'
    "}"
)


# --------------------------------------------------------------------------- #
# HypothesisEngine
# --------------------------------------------------------------------------- #


class HypothesisEngine:
    """Hypothesis-test-revise loop for diagnostic tasks.

    Stateless across sessions — each diagnostic gets a fresh session.
    Sessions are tracked in-memory (could be persisted to disk in a
    future version).
    """

    def __init__(
        self,
        *,
        llm_call=None,
        n_hypotheses: int = 3,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8,
    ) -> None:
        self._llm_call = llm_call
        self._n_hypotheses = max(2, min(n_hypotheses, 5))
        self._max_iterations = max(2, min(max_iterations, 10))
        self._confidence_threshold = max(0.5, min(confidence_threshold, 0.95))
        self._sessions: dict[str, DiagnosticSession] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start_session(self, *, symptom: str, context: str = "") -> str:
        """Start a new diagnostic session. Returns session_id."""
        import uuid

        session_id = f"diag_{uuid.uuid4().hex[:12]}"
        session = DiagnosticSession(
            session_id=session_id,
            symptom=symptom,
            context=context,
        )
        # Generate initial hypotheses.
        if self._llm_call is not None:
            hypotheses = self._form_hypotheses(symptom=symptom, context=context)
            session.hypotheses = hypotheses
        self._sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> DiagnosticSession | None:
        return self._sessions.get(session_id)

    def record_test_result(
        self,
        *,
        session_id: str,
        hypothesis_id: str,
        test_result: str,
    ) -> dict[str, Any] | None:
        """Record the result of testing a hypothesis. Returns next-step guidance."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        hypothesis = next((h for h in session.hypotheses if h.hypothesis_id == hypothesis_id), None)
        if hypothesis is None:
            return None
        if self._llm_call is None:
            return None

        hypothesis.test_result = test_result
        hypothesis.tested_at = time.time()
        hypothesis.status = HypothesisStatus.TESTING
        session.iterations += 1

        # Revise via LLM.
        revision = self._revise(hypothesis=hypothesis, symptom=session.symptom)
        if revision is None:
            return None

        hypothesis.posterior_probability = revision.get("posterior_probability", hypothesis.prior_probability)
        next_action = revision.get("next_action", "test_further")

        if next_action == "act" or hypothesis.posterior_probability >= self._confidence_threshold:
            hypothesis.status = HypothesisStatus.CONFIRMED
            session.confidence = hypothesis.posterior_probability
            session.final_conclusion = revision.get("conclusion", hypothesis.statement)
            session.closed = True
            return {
                "session_id": session_id,
                "hypothesis_id": hypothesis_id,
                "status": "confirmed",
                "posterior": hypothesis.posterior_probability,
                "conclusion": session.final_conclusion,
                "next_action": "act",
            }
        if next_action == "abandon" or session.iterations >= self._max_iterations:
            hypothesis.status = HypothesisStatus.REFUTED
            # Try next untested hypothesis or close session.
            remaining = session.active_hypotheses()
            if not remaining:
                session.closed = True
                return {
                    "session_id": session_id,
                    "hypothesis_id": hypothesis_id,
                    "status": "abandoned",
                    "posterior": hypothesis.posterior_probability,
                    "next_action": "give_up",
                }
            return {
                "session_id": session_id,
                "hypothesis_id": hypothesis_id,
                "status": "refuted",
                "posterior": hypothesis.posterior_probability,
                "next_action": "try_next",
                "next_hypothesis_id": remaining[0].hypothesis_id,
                "next_test": remaining[0].test_design,
            }
        # test_further: add refined hypothesis.
        refined = revision.get("refined_hypothesis", "").strip()
        if refined:
            import uuid

            new_h = Hypothesis(
                hypothesis_id=f"h_{uuid.uuid4().hex[:8]}",
                statement=refined,
                prior_probability=hypothesis.posterior_probability,
                posterior_probability=hypothesis.posterior_probability,
                status=HypothesisStatus.UNTESTED,
                test_design=revision.get("refined_test_design", ""),
                revisions=hypothesis.revisions + 1,
            )
            session.hypotheses.append(new_h)
            hypothesis.status = HypothesisStatus.SUPERSEDED
            return {
                "session_id": session_id,
                "hypothesis_id": new_h.hypothesis_id,
                "status": "refined",
                "prior": new_h.prior_probability,
                "next_action": "test_further",
                "next_test": new_h.test_design,
            }
        return {
            "session_id": session_id,
            "hypothesis_id": hypothesis_id,
            "status": "inconclusive",
            "posterior": hypothesis.posterior_probability,
            "next_action": "try_next",
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "symptom": s.symptom[:200],
                "iterations": s.iterations,
                "closed": s.closed,
                "confidence": s.confidence,
                "hypotheses_count": len(s.hypotheses),
            }
            for s in self._sessions.values()
        ]

    # ------------------------------------------------------------------ #
    # LLM calls
    # ------------------------------------------------------------------ #

    def _form_hypotheses(self, *, symptom: str, context: str) -> list[Hypothesis]:
        import uuid

        system = _FORM_HYPOTHESES_SYSTEM.format(n=self._n_hypotheses)
        user = (
            f"Symptom: {symptom}\n\n"
            f"Context:\n{context or '(none provided)'}\n\n"
            f"Generate {self._n_hypotheses} testable hypotheses now."
        )
        try:
            raw = self._llm_call(system, user)
            data = self._parse_json_array(raw)
        except Exception:
            data = None
        if not data:
            return []
        hypotheses: list[Hypothesis] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            try:
                prior = float(entry.get("prior_probability", 0.5))
            except (TypeError, ValueError):
                prior = 0.5
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"h_{uuid.uuid4().hex[:8]}",
                    statement=str(entry.get("statement", "")).strip(),
                    prior_probability=max(0.0, min(1.0, prior)),
                    posterior_probability=max(0.0, min(1.0, prior)),
                    test_design=str(entry.get("test_design", "")).strip(),
                )
            )
        return hypotheses

    def _revise(self, *, hypothesis: Hypothesis, symptom: str) -> dict[str, Any] | None:
        system = _REVISE_SYSTEM
        user = (
            f"Symptom: {symptom}\n\n"
            f"Hypothesis: {hypothesis.statement}\n"
            f"Prior probability: {hypothesis.prior_probability:.2f}\n"
            f"Test design: {hypothesis.test_design}\n"
            f"Test result: {hypothesis.test_result}\n\n"
            "Apply Bayesian updating and return the JSON."
        )
        try:
            raw = self._llm_call(system, user)
            data = self._parse_json(raw)
        except Exception:
            data = None
        if not data:
            return None
        return data

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

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

    @staticmethod
    def _parse_json_array(raw: str) -> list[dict[str, Any]] | None:
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
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, list):
                    return data
            except Exception:
                return None
        return None


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_engine: HypothesisEngine | None = None


def get_hypothesis_engine() -> HypothesisEngine | None:
    return _engine


def configure_hypothesis_engine(
    *,
    llm_call=None,
    n_hypotheses: int = 3,
    max_iterations: int = 5,
    confidence_threshold: float = 0.8,
) -> HypothesisEngine | None:
    global _engine
    if llm_call is None:
        _engine = None
        return None
    _engine = HypothesisEngine(
        llm_call=llm_call,
        n_hypotheses=n_hypotheses,
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold,
    )
    return _engine
