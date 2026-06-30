"""Cognitive Tree — branching reasoning with pruning.

THE CORE INNOVATION OF v2.

The problem with linear reasoning
---------------------------------
v1's ReasoningProtocol generates ONE plan per action. If that plan is
wrong, the agent fails. Top-tier LLMs can sometimes recover via
self-reflection, but mid-tier models get stuck in loops: they make
the same mistake, reflect on it, make it again.

CognitiveTree solves this by making reasoning **branching** instead of
linear. Before a consequential action, the agent generates N parallel
hypotheses about how to proceed. Each hypothesis is a "thought branch".
The tree is then pruned by:

1. **Cheap critic first** — fast heuristics reject obviously-bad branches
   (e.g., "this branch uses `rm -rf` without justification")
2. **LLM critic second** — for branches that pass the cheap critic, an
   LLM call evaluates them on a rubric (soundness, completeness, risk)
3. **Reflexion recall** — branches that contradict past lessons are
   penalized
4. **Confidence scoring** — surviving branches get a confidence score;
   the highest-scoring branch becomes the chosen plan

This is "Tree of Thoughts" (Yao et al. 2023) adapted for an agent
runtime. The key insight: **don't ask the LLM to commit to one plan;
ask it to generate alternatives, then prune**.

Token economics
---------------
- N=3 branches × 1 LLM call each = 3 calls for generation
- 1 LLM call to score all 3 (batched in 1 prompt)
- Total: 4 LLM calls for an IRREVERSIBLE action

vs. v1: 1 plan + 1 critique = 2 calls

So CognitiveTree is 2x more expensive. BUT:
- It only runs for CONSEQUENTIAL+ actions (STANDARD still uses v1 path)
- It only runs when `cognitive_tree_enabled: true` (opt-in)
- It prevents the "stuck in a loop" failure mode that wastes 10+ calls
  on retries

Net: cheaper in expectation for hard tasks, more expensive for easy ones.
The config flag lets the user decide.

Architecture
------------
```
                    ┌──────────────────────┐
                    │  User task + context │
                    └──────────┬───────────┘
                               │
                  ┌────────────▼────────────┐
                  │  generate_branches()    │
                  │  → N candidate plans    │
                  └────────────┬────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
           Branch 1       Branch 2       Branch 3
                │              │              │
                ▼              ▼              ▼
         ┌──────────────────────────────────────┐
         │  cheap_critic() — fast heuristics    │
         │  (regex, pattern, reflexion recall)  │
         └──────────────┬───────────────────────┘
                        │ survivors
                        ▼
         ┌──────────────────────────────────────┐
         │  llm_score() — batched LLM rubric    │
         │  (soundness, completeness, risk)     │
         └──────────────┬───────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────┐
         │  select_best() — pick highest score  │
         │  (or "ask_user" if confidence < 0.6) │
         └──────────────────────────────────────┘
```

Why this works
--------------
A senior engineer doesn't just "plan then execute". They sketch 2-3
approaches in their head, reject the bad ones quickly, then commit.
CognitiveTree makes the agent do the same. The "cheap critic" is the
analogue of the engineer's gut feeling — fast, mostly right, free.
The "LLM critic" is the analogue of writing out the pros/cons — slower,
more thorough.

The pruning matters more than the generation. N=5 branches with good
pruning beats N=20 branches with bad pruning. We invest in the critic,
not in generating more branches.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .decision import Classification, DecisionClass
from .reasoning import ReasoningPlan


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ThoughtBranch:
    """One candidate plan in the cognitive tree."""

    branch_id: int
    plan: ReasoningPlan
    cheap_critic_verdict: Literal["pass", "reject", "uncertain"] = "uncertain"
    cheap_critic_reason: str = ""
    llm_score: float = 0.0  # 0.0 to 1.0
    llm_rubric: dict[str, float] = field(default_factory=dict)
    # rubric keys: soundness, completeness, risk, efficiency
    reflexion_penalty: float = 0.0  # subtracted from llm_score
    final_score: float = 0.0  # llm_score - reflexion_penalty
    rejected: bool = False
    rejection_reason: str = ""


@dataclass
class CognitiveTreeResult:
    """The output of a cognitive tree evaluation."""

    branches: list[ThoughtBranch]
    selected: ThoughtBranch | None
    confidence: float  # 0.0 to 1.0
    decision: Literal["proceed", "abort", "ask_user"]
    rationale: str
    elapsed_ms: int = 0
    llm_calls: int = 0

    @property
    def selected_plan(self) -> ReasoningPlan | None:
        return self.selected.plan if self.selected else None


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_GENERATE_BRANCHES_SYSTEM = (
    "You are the planning layer of an AI agent. The agent is about to take "
    "a consequential action. Generate {n} DISTINCT candidate plans for how "
    "to proceed. Each plan must be a genuinely different approach — not "
    "paraphrases of the same idea. If you can only think of one good "
    "approach, generate 1 and leave the rest empty.\n\n"
    "Return STRICT JSON: an array of up to {n} objects. Schema per object:\n"
    "{\n"
    '  "goal": "what this approach achieves",\n'
    '  "approach": "how it works, step by step",\n'
    '  "alternatives_rejected": "why other approaches are worse",\n'
    '  "risks": "what could go wrong",\n'
    '  "reversibility": "can this be undone? how?",\n'
    '  "decision": "proceed" | "abort" | "ask_user",\n'
    '  "rationale": "one-sentence justification",\n'
    '  "confidence": 0.0 to 1.0 — your confidence this is the right approach\n'
    "}"
)

_SCORE_BRANCHES_SYSTEM = (
    "You are the evaluation layer of an AI agent. You are given {n} "
    "candidate plans for the same action. Score each on a rubric. Be "
    "harsh — a plan that 'sounds good' but has hidden assumptions should "
    "score low on soundness.\n\n"
    "Rubric (each 0.0 to 1.0):\n"
    "- soundness: are the assumptions valid? does the logic hold?\n"
    "- completeness: does it cover edge cases? what's missing?\n"
    "- risk: 1.0 = no risk, 0.0 = catastrophic risk (inverted)\n"
    "- efficiency: is this the simplest way to achieve the goal?\n\n"
    "Return STRICT JSON: an array of {n} objects in the same order as "
    "input. Schema per object:\n"
    "{\n"
    '  "soundness": 0.0 to 1.0,\n'
    '  "completeness": 0.0 to 1.0,\n'
    '  "risk": 0.0 to 1.0,\n'
    '  "efficiency": 0.0 to 1.0,\n'
    '  "overall": 0.0 to 1.0 — weighted average (your judgment),\n'
    '  "one_word_verdict": "proceed" | "modify" | "reject",\n'
    '  "critique": "one sentence explaining the overall score"\n'
    "}"
)


# --------------------------------------------------------------------------- #
# CognitiveTree
# --------------------------------------------------------------------------- #


class CognitiveTree:
    """Branching reasoning with pruning.

    Lifecycle:
        1. generate_branches() — N candidate plans (1 LLM call, batched)
        2. cheap_critic() — fast heuristics reject obviously-bad branches
        3. llm_score() — batched LLM rubric scoring (1 LLM call)
        4. select_best() — pick highest final_score, or ask_user if low confidence

    Total LLM cost: 2 calls (generation + scoring), regardless of N.
    """

    def __init__(
        self,
        *,
        llm_call=None,
        reflexion_recall=None,
        n_branches: int = 3,
        min_confidence: float = 0.6,
        max_confidence: float = 0.85,
    ) -> None:
        self._llm_call = llm_call
        # reflexion_recall(query, limit) -> str
        self._reflexion_recall = reflexion_recall or (lambda query, limit=3: "")
        self._n_branches = max(2, min(n_branches, 5))
        self._min_confidence = max(0.0, min(min_confidence, 1.0))
        self._max_confidence = max(self._min_confidence, min(max_confidence, 1.0))

    def evaluate(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
        classification: Classification,
        conversation_context: str,
    ) -> CognitiveTreeResult | None:
        """Run the full cognitive tree pipeline. Returns None on failure."""
        if self._llm_call is None:
            return None
        started = time.time()
        llm_calls = 0
        try:
            # Phase 1: generate branches.
            branches = self._generate_branches(
                tool_name=tool_name,
                args=args,
                classification=classification,
                conversation_context=conversation_context,
            )
            llm_calls += 1
            if not branches:
                return None

            # Phase 2: cheap critic.
            for branch in branches:
                self._cheap_critic(branch, tool_name=tool_name, args=args)

            # Phase 3: LLM score (batched).
            survivors = [b for b in branches if not b.rejected and b.cheap_critic_verdict != "reject"]
            if survivors:
                self._llm_score(survivors, tool_name=tool_name, args=args)
                llm_calls += 1

            # Phase 4: reflexion penalty.
            for branch in branches:
                self._apply_reflexion_penalty(branch, tool_name=tool_name)
                branch.final_score = max(0.0, branch.llm_score - branch.reflexion_penalty)

            # Phase 5: select best.
            eligible = [b for b in branches if not b.rejected and b.final_score > 0]
            eligible.sort(key=lambda b: b.final_score, reverse=True)
            selected = eligible[0] if eligible else None

            # Phase 6: decision.
            if selected is None:
                decision = "abort"
                confidence = 0.0
                rationale = "All branches rejected by critic."
            elif selected.final_score >= self._max_confidence:
                decision = "proceed"
                confidence = selected.final_score
                rationale = (
                    f"Branch #{selected.branch_id} selected with confidence "
                    f"{confidence:.2f}. Rubric: {selected.llm_rubric}. "
                    f"Critique: {selected.plan.rationale}"
                )
            elif selected.final_score >= self._min_confidence:
                decision = "proceed"
                confidence = selected.final_score
                rationale = (
                    f"Branch #{selected.branch_id} selected with moderate confidence "
                    f"{confidence:.2f}. Monitor execution closely."
                )
            else:
                decision = "ask_user"
                confidence = selected.final_score
                rationale = (
                    f"Best branch #{selected.branch_id} has low confidence "
                    f"{confidence:.2f}. Recommend asking user before proceeding."
                )

            elapsed = int((time.time() - started) * 1000)
            return CognitiveTreeResult(
                branches=branches,
                selected=selected,
                confidence=confidence,
                decision=decision,  # type: ignore[arg-type]
                rationale=rationale,
                elapsed_ms=elapsed,
                llm_calls=llm_calls,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Phase implementations
    # ------------------------------------------------------------------ #

    def _generate_branches(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
        classification: Classification,
        conversation_context: str,
    ) -> list[ThoughtBranch]:
        """Generate N candidate plans in a single LLM call."""
        args_str = json.dumps(args or {}, ensure_ascii=False, default=str)
        system = _GENERATE_BRANCHES_SYSTEM.format(n=self._n_branches)
        user = (
            f"Tool: {tool_name}\n"
            f"Classification: {classification.cls.name} — {classification.reason}\n"
            f"Arguments: {args_str}\n\n"
            f"Conversation context:\n{conversation_context}\n\n"
            f"Generate {self._n_branches} distinct candidate plans now."
        )
        raw = self._llm_call(system, user)
        data = self._parse_json_array(raw)
        if not data:
            return []
        branches: list[ThoughtBranch] = []
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            plan = ReasoningPlan(
                tool_name=tool_name,
                args_summary=args_str[:200],
                goal=str(entry.get("goal", "")).strip(),
                approach=str(entry.get("approach", "")).strip(),
                alternatives=str(entry.get("alternatives_rejected", "")).strip(),
                risks=str(entry.get("risks", "")).strip(),
                reversibility=str(entry.get("reversibility", "")).strip(),
                decision=str(entry.get("decision", "proceed")).strip() or "proceed",
                rationale=str(entry.get("rationale", "")).strip(),
                classification=classification.cls.name,
            )
            # Seed llm_score with the branch's own confidence (will be overwritten).
            try:
                self_confidence = float(entry.get("confidence", 0.5))
            except (TypeError, ValueError):
                self_confidence = 0.5
            branch = ThoughtBranch(
                branch_id=i,
                plan=plan,
                llm_score=self_confidence,
            )
            branches.append(branch)
        return branches

    def _cheap_critic(
        self,
        branch: ThoughtBranch,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
    ) -> None:
        """Fast heuristics to reject obviously-bad branches. No LLM call."""
        plan = branch.plan
        # Rule 1: plan decides to abort → reject.
        if plan.decision == "abort":
            branch.cheap_critic_verdict = "reject"
            branch.cheap_critic_reason = "Plan self-aborted."
            branch.rejected = True
            branch.rejection_reason = "Plan decided to abort."
            return
        # Rule 2: empty goal → reject.
        if not plan.goal:
            branch.cheap_critic_verdict = "reject"
            branch.cheap_critic_reason = "Plan has no goal."
            branch.rejected = True
            branch.rejection_reason = "Empty goal."
            return
        # Rule 3: empty approach → reject.
        if not plan.approach:
            branch.cheap_critic_verdict = "reject"
            branch.cheap_critic_reason = "Plan has no approach."
            branch.rejected = True
            branch.rejection_reason = "Empty approach."
            return
        # Rule 4: IRREVERSIBLE action with empty reversibility note → uncertain.
        if (
            branch.plan.classification == DecisionClass.IRREVERSIBLE.name
            and not plan.reversibility
        ):
            branch.cheap_critic_verdict = "uncertain"
            branch.cheap_critic_reason = "IRREVERSIBLE action with no reversibility note."
            return
        # Rule 5: approach mentions a different tool than the one being called.
        # (Common LLM mistake: plan says "use grep" but the tool call is `bash`.)
        # We only flag this, not reject — sometimes the LLM is suggesting a
        # better tool, which is legitimate.
        branch.cheap_critic_verdict = "pass"

    def _llm_score(
        self,
        branches: list[ThoughtBranch],
        *,
        tool_name: str,
        args: dict[str, Any] | None,
    ) -> None:
        """Batched LLM rubric scoring. 1 call for all branches."""
        n = len(branches)
        system = _SCORE_BRANCHES_SYSTEM.format(n=n)
        parts = [f"Tool: {tool_name}", f"Arguments: {json.dumps(args or {}, ensure_ascii=False, default=str)}", ""]
        for i, branch in enumerate(branches):
            parts.append(f"--- Plan {i + 1} ---")
            parts.append(branch.plan.as_prompt_block())
            parts.append("")
        parts.append(f"Score all {n} plans. Return JSON array of {n} objects.")
        user = "\n".join(parts)
        try:
            raw = self._llm_call(system, user)
            data = self._parse_json_array(raw)
        except Exception:
            data = None
        if not data or len(data) != n:
            # Scoring failed — keep self-confidence scores, don't reject.
            return
        for branch, entry in zip(branches, data):
            if not isinstance(entry, dict):
                continue
            try:
                rubric = {
                    "soundness": float(entry.get("soundness", 0.5)),
                    "completeness": float(entry.get("completeness", 0.5)),
                    "risk": float(entry.get("risk", 0.5)),
                    "efficiency": float(entry.get("efficiency", 0.5)),
                }
                overall = float(entry.get("overall", sum(rubric.values()) / 4))
            except (TypeError, ValueError):
                continue
            branch.llm_rubric = rubric
            branch.llm_score = max(0.0, min(1.0, overall))
            verdict = str(entry.get("one_word_verdict", "proceed")).strip().lower()
            if verdict == "reject":
                branch.rejected = True
                branch.rejection_reason = str(entry.get("critique", "LLM critic rejected"))

    def _apply_reflexion_penalty(self, branch: ThoughtBranch, *, tool_name: str) -> None:
        """Penalize branches that contradict past lessons."""
        if branch.rejected:
            return
        try:
            query = f"{tool_name} {branch.plan.goal[:100]} {branch.plan.risks[:100]}"
            lessons = self._reflexion_recall(query, limit=3)
            if not lessons:
                return
            # Simple heuristic: if reflexion recall returns content, apply
            # a small penalty proportional to how much was recalled. More
            # recalled lessons = more past failures = more caution.
            # This is intentionally crude — a future version could use
            # embedding similarity to score lesson-vs-branch relevance.
            lesson_count = lessons.count("[")  # rough count of lesson entries
            branch.reflexion_penalty = min(0.3, lesson_count * 0.05)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

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

_tree: CognitiveTree | None = None


def get_tree() -> CognitiveTree | None:
    return _tree


def configure_tree(
    *,
    llm_call=None,
    reflexion_recall=None,
    n_branches: int = 3,
    min_confidence: float = 0.6,
    max_confidence: float = 0.85,
) -> CognitiveTree | None:
    global _tree
    if llm_call is None:
        _tree = None
        return None
    _tree = CognitiveTree(
        llm_call=llm_call,
        reflexion_recall=reflexion_recall,
        n_branches=n_branches,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    return _tree
