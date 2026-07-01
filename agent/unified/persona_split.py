"""Agent Persona Split — 5 specialist personas instead of 5 copies.

THE BREAKTHROUGH
----------------
Ensemble mode runs 3 copies of the SAME model with the SAME prompt.
Result: 3 similar answers → judge picks randomly. Not much better
than single model.

AgentPersonaSplit runs 5 DIFFERENT personas, each with a unique
perspective:
1. Architect — system design, scalability, maintainability
2. Optimizer — performance, efficiency, resource usage
3. Security Auditor — vulnerabilities, attack vectors, data safety
4. User Advocate — UX, simplicity, end-user needs
5. Devil's Advocate — challenge assumptions, find edge cases

Each persona has a specialized system prompt that makes it focus on
its area. Results are genuinely different → judge synthesizes a
comprehensive answer.

The judge sees all 5 perspectives and creates a response that
incorporates insights from ALL personas — something no single
perspective would produce alone.

USAGE
-----
    from agent.unified.persona_split import persona_solve

    result = persona_solve(
        request="Design a REST API for a todo app",
        llm_call=my_llm,
    )
    # → 5 perspectives + synthesized answer
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


# --------------------------------------------------------------------------- #
# Persona definitions
# --------------------------------------------------------------------------- #


PERSONAS: dict[str, dict[str, str]] = {
    "architect": {
        "name": "Architect",
        "icon": "🏗️",
        "system_prompt": (
            "You are the ARCHITECT persona. Your sole focus is SYSTEM DESIGN.\n\n"
            "When analyzing a request, you consider:\n"
            "- Architecture patterns (MVC, microservices, monolith, event-driven)\n"
            "- Scalability (can it handle 10x growth?)\n"
            "- Maintainability (can a new dev understand this in 6 months?)\n"
            "- Separation of concerns\n"
            "- Dependency management\n"
            "- Technical debt implications\n\n"
            "You IGNORE: implementation details, UI aesthetics, performance micro-optimizations.\n"
            "You care about: structure, patterns, long-term sustainability.\n\n"
            "Be specific about architectural decisions and trade-offs."
        ),
        "focus": "architecture, scalability, maintainability",
    },
    "optimizer": {
        "name": "Optimizer",
        "icon": "⚡",
        "system_prompt": (
            "You are the OPTIMIZER persona. Your sole focus is PERFORMANCE.\n\n"
            "When analyzing a request, you consider:\n"
            "- Time complexity (O(n), O(n²), etc.)\n"
            "- Space complexity (memory usage)\n"
            "- Bottlenecks (database queries, network calls, CPU-bound work)\n"
            "- Caching opportunities\n"
            "- Parallelization potential\n"
            "- Resource usage (CPU, memory, disk, network)\n\n"
            "You IGNORE: code style, architecture aesthetics, future maintainability.\n"
            "You care about: speed, efficiency, resource minimization.\n\n"
            "Be specific about performance numbers and optimization techniques."
        ),
        "focus": "performance, efficiency, resources",
    },
    "security_auditor": {
        "name": "Security Auditor",
        "icon": "🛡️",
        "system_prompt": (
            "You are the SECURITY AUDITOR persona. Your sole focus is SECURITY.\n\n"
            "When analyzing a request, you consider:\n"
            "- OWASP Top 10 vulnerabilities\n"
            "- Input validation and sanitization\n"
            "- Authentication and authorization flaws\n"
            "- Data exposure (PII, secrets, tokens)\n"
            "- Injection attacks (SQL, XSS, command)\n"
            "- Race conditions and TOCTOU\n"
            "- Supply chain risks\n\n"
            "You IGNORE: performance, aesthetics, convenience.\n"
            "You care about: safety, data protection, attack prevention.\n\n"
            "Be specific about vulnerabilities and mitigation strategies."
        ),
        "focus": "security, vulnerabilities, data protection",
    },
    "user_advocate": {
        "name": "User Advocate",
        "icon": "👤",
        "system_prompt": (
            "You are the USER ADVOCATE persona. Your sole focus is END-USER EXPERIENCE.\n\n"
            "When analyzing a request, you consider:\n"
            "- Is this easy for a non-technical user to understand?\n"
            "- What will frustrate the user?\n"
            "- Error messages: are they helpful or confusing?\n"
            "- Onboarding: how steep is the learning curve?\n"
            "- Edge cases: what happens when the user makes a mistake?\n"
            "- Accessibility: can everyone use this?\n"
            "- Documentation: is it needed? Is it clear?\n\n"
            "You IGNORE: technical elegance, performance, architecture.\n"
            "You care about: usability, clarity, user satisfaction.\n\n"
            "Be specific about UX issues and improvements."
        ),
        "focus": "usability, clarity, user satisfaction",
    },
    "devils_advocate": {
        "name": "Devil's Advocate",
        "icon": "😈",
        "system_prompt": (
            "You are the DEVIL'S ADVOCATE persona. Your job is to CHALLENGE everything.\n\n"
            "When analyzing a request, you:\n"
            "- Question every assumption (Why this approach? Why not alternative X?)\n"
            "- Find edge cases nobody else considered\n"
            "- Identify what could go WRONG\n"
            "- Challenge the premise (is this even the right problem to solve?)\n"
            "- Consider failure modes (what if the API is down? what if data is corrupt?)\n"
            "- Play 'what if' scenarios (what if scale 100x? what if team is 50 people?)\n\n"
            "You are NOT negative for the sake of it — you find REAL weaknesses.\n"
            "You IGNORE: being agreeable, consensus, hurt feelings.\n"
            "You care about: robustness, edge cases, anti-fragility.\n\n"
            "Be specific about risks and what could go wrong."
        ),
        "focus": "risks, edge cases, anti-fragility",
    },
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class PersonaResponse:
    """One persona's response."""

    persona: str
    name: str
    icon: str
    response: str
    elapsed_ms: int = 0
    error: str = ""


@dataclass
class PersonaResult:
    """Final result of persona split solve."""

    responses: list[PersonaResponse] = field(default_factory=list)
    synthesized: str = ""
    judge_rationale: str = ""
    total_elapsed_ms: int = 0
    personas_used: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Judge prompt
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You are the SYNTHESIS JUDGE. You receive 5 specialist perspectives on "
    "the same request. Each persona focused on a different aspect:\n"
    "1. Architect (structure, scalability)\n"
    "2. Optimizer (performance, efficiency)\n"
    "3. Security Auditor (vulnerabilities, safety)\n"
    "4. User Advocate (usability, clarity)\n"
    "5. Devil's Advocate (risks, edge cases)\n\n"
    "Your job: synthesize ALL perspectives into a single comprehensive answer.\n"
    "Do NOT pick one persona's answer — INTEGRATE insights from all 5.\n"
    "Address conflicts between personas (e.g., security vs usability).\n"
    "Prioritize by context: if the request is about a production system, "
    "weight security + architecture higher. If it's a prototype, weight "
    "user advocate + speed higher.\n\n"
    "Return your synthesized answer directly — no meta-commentary about "
    "the process."
)


# --------------------------------------------------------------------------- #
# PersonaSplit solver
# --------------------------------------------------------------------------- #


class PersonaSplitSolver:
    """Run 5 specialist personas + judge synthesis.

    Uses ThreadPoolExecutor for parallel persona calls.
    """

    def __init__(
        self,
        *,
        llm_call: Callable[[str, str], str] | None = None,
        judge_llm_call: Callable[[str, str], str] | None = None,
        max_workers: int = 5,
        timeout_seconds: float = 60.0,
        personas: list[str] | None = None,
    ) -> None:
        self._llm = llm_call
        self._judge = judge_llm_call or llm_call
        self._max_workers = max(1, min(max_workers, 5))
        self._timeout = max(10.0, timeout_seconds)
        self._persona_keys = personas or list(PERSONAS.keys())

    def solve(
        self,
        *,
        request: str,
        context: str = "",
    ) -> PersonaResult:
        """Run all personas in parallel + judge synthesis."""
        started = time.time()
        if self._llm is None:
            return PersonaResult(total_elapsed_ms=0)

        # Phase 1: Run all personas in parallel
        responses: list[PersonaResponse] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {}
            for key in self._persona_keys:
                persona = PERSONAS[key]
                future = pool.submit(
                    self._run_persona,
                    key=key,
                    persona=persona,
                    request=request,
                    context=context,
                )
                futures[future] = key

            for future in as_completed(futures, timeout=self._timeout):
                key = futures[future]
                try:
                    result = future.result(timeout=self._timeout)
                    responses.append(result)
                except Exception as exc:
                    persona = PERSONAS[key]
                    responses.append(
                        PersonaResponse(
                            persona=key,
                            name=persona["name"],
                            icon=persona["icon"],
                            response="",
                            error=repr(exc),
                        )
                    )

        # Sort by persona order
        order = {k: i for i, k in enumerate(self._persona_keys)}
        responses.sort(key=lambda r: order.get(r.persona, 99))

        # Phase 2: Judge synthesis
        synthesized = ""
        judge_rationale = ""
        if self._judge is not None and any(r.response for r in responses):
            judge_input = self._build_judge_input(request, responses)
            try:
                synthesized = self._judge(JUDGE_SYSTEM, judge_input).strip()
            except Exception:
                synthesized = ""

        elapsed = int((time.time() - started) * 1000)
        return PersonaResult(
            responses=responses,
            synthesized=synthesized,
            judge_rationale=judge_rationale,
            total_elapsed_ms=elapsed,
            personas_used=[r.persona for r in responses],
        )

    def _run_persona(
        self,
        *,
        key: str,
        persona: dict[str, str],
        request: str,
        context: str,
    ) -> PersonaResponse:
        """Run one persona."""
        started = time.time()
        system = persona["system_prompt"]
        user = (
            f"Request: {request}\n\n"
            f"Context: {context or '(none)'}\n\n"
            f"Analyze this from your perspective as {persona['name']}. "
            f"Focus on: {persona['focus']}. Be specific and actionable."
        )
        try:
            response = self._llm(system, user)
            elapsed = int((time.time() - started) * 1000)
            return PersonaResponse(
                persona=key,
                name=persona["name"],
                icon=persona["icon"],
                response=response.strip() if response else "",
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = int((time.time() - started) * 1000)
            return PersonaResponse(
                persona=key,
                name=persona["name"],
                icon=persona["icon"],
                response="",
                error=repr(exc),
                elapsed_ms=elapsed,
            )

    @staticmethod
    def _build_judge_input(request: str, responses: list[PersonaResponse]) -> str:
        """Build judge input from all persona responses."""
        parts = [f"Original request: {request}\n"]
        for r in responses:
            parts.append(f"\n--- {r.icon} {r.name} ---")
            if r.error:
                parts.append(f"(Error: {r.error})")
            else:
                parts.append(r.response[:2000])
        parts.append("\n\nSynthesize a comprehensive answer integrating all perspectives.")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_solver: PersonaSplitSolver | None = None


def get_solver() -> PersonaSplitSolver | None:
    return _solver


def configure_solver(
    *,
    llm_call: Callable[[str, str], str] | None = None,
    judge_llm_call: Callable[[str, str], str] | None = None,
    max_workers: int = 5,
    timeout_seconds: float = 60.0,
    personas: list[str] | None = None,
) -> PersonaSplitSolver:
    global _solver
    _solver = PersonaSplitSolver(
        llm_call=llm_call,
        judge_llm_call=judge_llm_call,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        personas=personas,
    )
    return _solver


def persona_solve(
    *,
    request: str,
    context: str = "",
) -> dict[str, Any]:
    """Public API: run 5-persona split + judge synthesis.

    Returns dict with:
        - synthesized: final answer
        - responses: list of 5 persona responses
        - total_elapsed_ms: total time
    """
    if _solver is None:
        return {"enabled": False, "synthesized": ""}
    result = _solver.solve(request=request, context=context)
    return {
        "enabled": True,
        "synthesized": result.synthesized,
        "responses": [
            {
                "persona": r.persona,
                "name": r.name,
                "icon": r.icon,
                "response_preview": r.response[:300],
                "error": r.error,
                "elapsed_ms": r.elapsed_ms,
            }
            for r in result.responses
        ],
        "total_elapsed_ms": result.total_elapsed_ms,
        "personas_used": result.personas_used,
    }


def persona_stats() -> dict[str, Any]:
    """Public API: get persona solver config."""
    if _solver is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "personas": [PERSONAS[k]["name"] for k in _solver._persona_keys],
        "max_workers": _solver._max_workers,
    }


def list_personas() -> list[dict[str, str]]:
    """Public API: list all available personas."""
    return [
        {
            "key": key,
            "name": p["name"],
            "icon": p["icon"],
            "focus": p["focus"],
        }
        for key, p in PERSONAS.items()
    ]
