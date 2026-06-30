"""Ensemble Reasoning — multi-model generation + LLM judge.

DIFFERENTIATOR vs TOP-TIER LABS
-------------------------------
Anthropic sells Claude. Z.ai sells GLM. OpenAI sells GPT. NONE let you
ensemble their model with competitors — that's bad for their business.

Hermes-Omni can. The user configures 2-3 model providers (already
supported via plugins/model-providers/), and for hard tasks the agent:

1. Sends the request to ALL configured models in parallel
2. Each model generates its own solution independently
3. A judge model (could be one of the same, or a stronger one) compares:
   - picks the best, OR
   - synthesizes a new answer combining the best parts, OR
   - flags disagreement (uncertainty signal to user)

This catches individual model biases. If GLM says X and Claude says Y,
the judge can detect which is right (or that they're both wrong and
synthesize Z).

WHEN IT RUNS
------------
- Opt-in, only for hard tasks (config flag + per-request override)
- Triggered explicitly via `ensemble_solve` tool, OR
- Auto-triggered when:
  - CognitiveTree confidence < 0.5 (low confidence)
  - Verifier fails after 3 iterations
  - User explicitly requests "second opinion"

TOKEN ECONOMICS
---------------
- N model calls (N = number of ensemble models, typically 3)
- 1 judge call
- Total: N+1 calls per hard task

This is 3-4x more expensive than single-model, but only runs for hard
tasks. For trivial chat, single-model path is used.

CONFIGURATION
-------------
User provides a list of "ensemble models" — each is a callable:
    ensemble_models = [
        ("glm-4.5", glm_llm_call),
        ("deepseek-v3", deepseek_llm_call),
        ("qwen3-235b", qwen_llm_call),
    ]

The judge is a separate callable (can be the strongest model, or a
dedicated judge model).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ModelResponse:
    """One model's response in the ensemble."""

    model_name: str
    response: str
    elapsed_ms: int = 0
    error: str = ""


@dataclass
class EnsembleResult:
    """Output of an ensemble solve."""

    responses: list[ModelResponse] = field(default_factory=list)
    decision: Literal["pick", "synthesize", "disagree"] = "pick"
    chosen_model: str = ""
    final_answer: str = ""
    disagreement: float = 0.0  # 0.0 (all same) to 1.0 (all different)
    judge_rationale: str = ""
    total_elapsed_ms: int = 0
    total_llm_calls: int = 0


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM = (
    "You are the judge of a multi-model ensemble. You are given the same "
    "request and N different model responses. Your job:\n"
    "1. Compare the responses on: accuracy, completeness, reasoning quality\n"
    "2. Decide:\n"
    "   - 'pick': one response is clearly best — name it\n"
    "   - 'synthesize': combine the best parts of multiple responses\n"
    "   - 'disagree': responses are irreconcilably different — flag uncertainty\n"
    "3. If 'pick', reproduce the chosen response VERBATIM as final_answer\n"
    "4. If 'synthesize', write a NEW response combining the best parts\n"
    "5. If 'disagree', pick the safest response and note the disagreement\n"
    "6. Score disagreement: 0.0 (all same) to 1.0 (all different)\n\n"
    "Return STRICT JSON:\n"
    "{\n"
    '  "decision": "pick" | "synthesize" | "disagree",\n'
    '  "chosen_model": "model name if pick, empty otherwise",\n'
    '  "final_answer": "the answer to return to user",\n'
    '  "disagreement": 0.0 to 1.0,\n'
    '  "judge_rationale": "one-paragraph explanation"\n'
    "}"
)


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #


# Type for an LLM callable: (system_prompt, user_prompt) -> str
LlmCallable = Callable[[str, str], str]


class EnsembleSolver:
    """Multi-model ensemble with LLM judge.

    The user configures a list of (model_name, llm_callable) pairs and
    a judge callable. The solver runs all models in parallel, then
    asks the judge to pick/synthesize.
    """

    def __init__(
        self,
        *,
        judge_llm_call: LlmCallable | None = None,
        models: list[tuple[str, LlmCallable]] | None = None,
        max_workers: int = 5,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._judge = judge_llm_call
        self._models = models or []
        self._max_workers = max(1, min(max_workers, 10))
        self._timeout = max(10.0, timeout_seconds)

    def set_models(self, models: list[tuple[str, LlmCallable]]) -> None:
        self._models = models

    def add_model(self, name: str, llm_call: LlmCallable) -> None:
        self._models.append((name, llm_call))

    def solve(
        self,
        *,
        request: str,
        context: str = "",
        system_prompt: str = "You are a helpful AI assistant. Answer the user's request accurately and thoroughly.",
    ) -> EnsembleResult:
        """Run the ensemble. Returns result with final_answer."""
        started = time.time()
        if not self._models:
            return EnsembleResult(
                total_elapsed_ms=int((time.time() - started) * 1000),
            )
        # Phase 1: parallel model calls.
        responses = self._call_models_parallel(
            request=request,
            context=context,
            system_prompt=system_prompt,
        )
        # Phase 2: judge.
        if self._judge is None or not responses:
            # No judge — just return the first non-error response.
            ok = next((r for r in responses if not r.error), None)
            return EnsembleResult(
                responses=responses,
                decision="pick",
                chosen_model=ok.model_name if ok else "",
                final_answer=ok.response if ok else "",
                total_elapsed_ms=int((time.time() - started) * 1000),
                total_llm_calls=len(responses),
            )
        judge_result = self._judge_responses(
            request=request,
            responses=responses,
        )
        return EnsembleResult(
            responses=responses,
            decision=judge_result.get("decision", "pick"),
            chosen_model=judge_result.get("chosen_model", ""),
            final_answer=judge_result.get("final_answer", ""),
            disagreement=judge_result.get("disagreement", 0.0),
            judge_rationale=judge_result.get("judge_rationale", ""),
            total_elapsed_ms=int((time.time() - started) * 1000),
            total_llm_calls=len(responses) + 1,
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _call_models_parallel(
        self,
        *,
        request: str,
        context: str,
        system_prompt: str,
    ) -> list[ModelResponse]:
        user_prompt = (
            f"Context:\n{context or '(none)'}\n\n"
            f"Request:\n{request}\n\n"
            "Provide your best response."
        )
        results: list[ModelResponse] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._call_one, name, fn, system_prompt, user_prompt): name
                for name, fn in self._models
            }
            for future in as_completed(futures, timeout=self._timeout):
                name = futures[future]
                try:
                    result = future.result(timeout=self._timeout)
                    results.append(result)
                except Exception as exc:
                    results.append(ModelResponse(model_name=name, response="", error=repr(exc)))
        # Sort by model name for stable ordering.
        results.sort(key=lambda r: r.model_name)
        return results

    @staticmethod
    def _call_one(
        name: str,
        fn: LlmCallable,
        system: str,
        user: str,
    ) -> ModelResponse:
        started = time.time()
        try:
            raw = fn(system, user)
            return ModelResponse(
                model_name=name,
                response=raw.strip() if raw else "",
                elapsed_ms=int((time.time() - started) * 1000),
            )
        except Exception as exc:
            return ModelResponse(
                model_name=name,
                response="",
                error=repr(exc),
                elapsed_ms=int((time.time() - started) * 1000),
            )

    def _judge_responses(
        self,
        *,
        request: str,
        responses: list[ModelResponse],
    ) -> dict[str, Any]:
        try:
            parts = [f"Request:\n{request}\n"]
            for i, r in enumerate(responses, 1):
                if r.error:
                    parts.append(f"\n--- Model {i}: {r.model_name} (ERROR) ---\n{r.error}")
                else:
                    parts.append(f"\n--- Model {i}: {r.model_name} ---\n{r.response}")
            parts.append("\n\nJudge these responses. Return JSON now.")
            user = "\n".join(parts)
            raw = self._judge(_JUDGE_SYSTEM, user)
            data = self._parse_json(raw)
            if data is None:
                # Fallback: pick first non-error.
                ok = next((r for r in responses if not r.error), None)
                return {
                    "decision": "pick",
                    "chosen_model": ok.model_name if ok else "",
                    "final_answer": ok.response if ok else "",
                    "disagreement": 0.5,
                    "judge_rationale": "judge parse failed; picked first model",
                }
            return {
                "decision": str(data.get("decision", "pick")).strip().lower(),
                "chosen_model": str(data.get("chosen_model", "")).strip(),
                "final_answer": str(data.get("final_answer", "")).strip(),
                "disagreement": max(0.0, min(1.0, float(data.get("disagreement", 0.0)))),
                "judge_rationale": str(data.get("judge_rationale", "")).strip(),
            }
        except Exception as exc:
            ok = next((r for r in responses if not r.error), None)
            return {
                "decision": "pick",
                "chosen_model": ok.model_name if ok else "",
                "final_answer": ok.response if ok else "",
                "disagreement": 0.5,
                "judge_rationale": f"judge failed: {exc!r}; picked first model",
            }

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

_solver: EnsembleSolver | None = None


def get_ensemble_solver() -> EnsembleSolver | None:
    return _solver


def configure_ensemble_solver(
    *,
    judge_llm_call: LlmCallable | None = None,
    models: list[tuple[str, LlmCallable]] | None = None,
    max_workers: int = 5,
    timeout_seconds: float = 60.0,
) -> EnsembleSolver | None:
    global _solver
    _solver = EnsembleSolver(
        judge_llm_call=judge_llm_call,
        models=models,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
    )
    return _solver


def ensemble_solve(
    *,
    request: str,
    context: str = "",
    system_prompt: str = "You are a helpful AI assistant. Answer the user's request accurately and thoroughly.",
) -> EnsembleResult:
    """Public API: run ensemble solve. Returns result with final_answer."""
    if _solver is None:
        return EnsembleResult()
    return _solver.solve(request=request, context=context, system_prompt=system_prompt)


def register_ensemble_model(name: str, llm_call: LlmCallable) -> None:
    """Add a model to the ensemble."""
    if _solver is None:
        configure_ensemble_solver()
    if _solver is not None:
        _solver.add_model(name, llm_call)
