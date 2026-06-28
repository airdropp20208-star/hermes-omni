"""Sentinel Agent — Task decomposition and progress tracking.

Inspired by OmniAgent's Sentinel, adapted for hermes-omni.

Features:
1. Task decomposition → structured milestone plan
2. Progress tracking → verify each milestone completion
3. Cross-session recovery → persist plans to workspace
4. Activation heuristics — complex multi-step tasks, repeated failures

Activation conditions (any one):
  1. Task contains multi-step keywords ("then"/"after that"/"next")
  2. Main agent has ≥N consecutive reflexion failures
  3. LLM complexity estimation judges the task needs planning
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Multi-step keywords (bilingual)
_MULTI_STEP_KEYWORDS = re.compile(
    r"\b(?:then|after\s+that|next|afterwards|finally|subsequently"
    r"|然后|接着|之后|最后|下一步|再|随后)\b",
    re.IGNORECASE | re.UNICODE,
)


@dataclass
class Milestone:
    """A single milestone in a task plan."""

    index: int
    description: str
    dependencies: List[int] = field(default_factory=list)
    success_criteria: str = ""
    status: str = "pending"  # "pending" | "in_progress" | "completed" | "failed"
    result_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Milestone":
        return cls(**data)


@dataclass
class TaskPlan:
    """A complete task decomposition plan."""

    task_hash: str
    task_description: str
    milestones: List[Milestone]
    created_at: str
    updated_at: str
    current_milestone_idx: int = 0
    status: str = "active"  # "active" | "completed" | "abandoned"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPlan":
        milestones = [Milestone.from_dict(m) for m in data["milestones"]]
        return cls(
            task_hash=data["task_hash"],
            task_description=data["task_description"],
            milestones=milestones,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            current_milestone_idx=data.get("current_milestone_idx", 0),
            status=data.get("status", "active"),
        )


class SentinelAgent:
    """Task decomposition and progress tracking agent.

    Usage:
        sentinel = SentinelAgent(work_dir=Path("./workspace"))

        # Check if task needs planning
        should_activate, reason = sentinel.should_activate("Build a full REST API...")

        if should_activate:
            plan = await sentinel.decompose(task, llm_call_fn)
            # ... execute plan ...
            sentinel.mark_milestone_completed(0, "API created and tested")
    """

    def __init__(
        self,
        work_dir: Path,
        activation_threshold: int = 2,
        llm_call_fn=None,
    ):
        """
        Args:
            work_dir: Working directory for plan persistence
            activation_threshold: Number of reflexion failures before auto-activation
            llm_call_fn: Optional async function for LLM-powered decomposition
        """
        self.work_dir = work_dir
        self._plans_dir = work_dir / ".hermes" / "sentinel"
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        self._activation_threshold = activation_threshold
        self._llm_call = llm_call_fn
        self._active_plan: Optional[TaskPlan] = None
        self._reflexion_failure_count = 0

    @property
    def is_active(self) -> bool:
        return self._active_plan is not None and self._active_plan.status == "active"

    def should_activate(
        self,
        task: str,
        reflexion_failure_count: int = 0,
    ) -> Tuple[bool, str]:
        """Determine if Sentinel should activate for this task.

        Returns:
            (should_activate, reason)
        """
        # Condition 1: Multi-step keywords
        keyword_matches = _MULTI_STEP_KEYWORDS.findall(task)
        if len(keyword_matches) >= 2:
            return True, f"Multi-step task detected ({len(keyword_matches)} keywords)"

        # Condition 2: Reflexion failures
        if reflexion_failure_count >= self._activation_threshold:
            return True, (
                f"Repeated failures ({reflexion_failure_count} >= "
                f"{self._activation_threshold})"
            )

        # Condition 3: Task length heuristic
        if len(task) > 500:
            return True, "Long task description suggests complexity"

        return False, "Task does not meet activation criteria"

    async def decompose(
        self,
        task: str,
    ) -> Optional[TaskPlan]:
        """Decompose task into milestones.

        Uses LLM if available, otherwise falls back to heuristic decomposition.
        """
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:16]
        now = datetime.now().isoformat()

        # Try LLM decomposition
        if self._llm_call:
            milestones = await self._llm_decompose(task)
            if milestones:
                plan = TaskPlan(
                    task_hash=task_hash,
                    task_description=task[:500],
                    milestones=milestones,
                    created_at=now,
                    updated_at=now,
                )
                self._active_plan = plan
                self._save_plan(plan)
                logger.info(
                    "sentinel_decomposed task_hash=%s milestones=%d",
                    task_hash, len(milestones),
                )
                return plan

        # Fallback: heuristic decomposition
        milestones = self._heuristic_decompose(task)
        plan = TaskPlan(
            task_hash=task_hash,
            task_description=task[:500],
            milestones=milestones,
            created_at=now,
            updated_at=now,
        )
        self._active_plan = plan
        self._save_plan(plan)
        return plan

    async def _llm_decompose(self, task: str) -> Optional[List[Milestone]]:
        """Use LLM to decompose task into milestones."""
        if not self._llm_call:
            return None

        prompt = f"""Decompose this task into clear, sequential milestones:

Task: {task}

Respond in JSON only:
{{
  "milestones": [
    {{"index": 0, "description": "...", "success_criteria": "...", "dependencies": []}},
    {{"index": 1, "description": "...", "success_criteria": "...", "dependencies": [0]}},
    ...
  ]
}}

Requirements:
- Each milestone should be a single, verifiable step
- Dependencies reference prerequisite milestone indices
- Maximum 10 milestones
- Success criteria should be specific and testable"""

        try:
            response = await self._llm_call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            content = (getattr(response, "content", None) or "").strip()
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                milestones = []
                for m in data.get("milestones", []):
                    milestones.append(Milestone(
                        index=m.get("index", len(milestones)),
                        description=m.get("description", ""),
                        success_criteria=m.get("success_criteria", ""),
                        dependencies=m.get("dependencies", []),
                    ))
                return milestones if milestones else None
        except Exception as e:
            logger.warning("sentinel_llm_decompose_failed: %s", e)

        return None

    @staticmethod
    def _heuristic_decompose(task: str) -> List[Milestone]:
        """Fallback heuristic decomposition based on task structure."""
        milestones = []

        # Split by common delimiters
        steps = re.split(
            r"(?:\d+[\.\)]\s*|(?:^|\n)[-•]\s*|(?:^|\n)\*\s*)",
            task,
        )
        steps = [s.strip() for s in steps if s.strip() and len(s.strip()) > 10]

        if not steps:
            # Single step task
            milestones.append(Milestone(
                index=0,
                description=task[:200],
                success_criteria="Task completed successfully",
            ))
        else:
            for i, step in enumerate(steps[:10]):
                milestones.append(Milestone(
                    index=i,
                    description=step[:200],
                    success_criteria=f"Step {i+1} completed",
                    dependencies=[i - 1] if i > 0 else [],
                ))

        return milestones

    def get_current_milestone(self) -> Optional[Milestone]:
        """Get the current in-progress milestone."""
        if not self._active_plan:
            return None
        for m in self._active_plan.milestones:
            if m.status in ("pending", "in_progress"):
                return m
        return None

    def mark_milestone_completed(
        self,
        milestone_idx: int,
        result_summary: str = "",
    ):
        """Mark a milestone as completed."""
        if not self._active_plan:
            return

        for m in self._active_plan.milestones:
            if m.index == milestone_idx:
                m.status = "completed"
                m.result_summary = result_summary[:500]
                break

        # Check if all milestones are completed
        all_done = all(
            m.status == "completed" for m in self._active_plan.milestones
        )
        if all_done:
            self._active_plan.status = "completed"
            logger.info("sentinel_plan_completed task_hash=%s", self._active_plan.task_hash)

        self._active_plan.updated_at = datetime.now().isoformat()
        self._save_plan(self._active_plan)

    def mark_milestone_failed(self, milestone_idx: int, reason: str = ""):
        """Mark a milestone as failed."""
        if not self._active_plan:
            return

        for m in self._active_plan.milestones:
            if m.index == milestone_idx:
                m.status = "failed"
                m.result_summary = reason[:500]
                break

        self._active_plan.updated_at = datetime.now().isoformat()
        self._save_plan(self._active_plan)

    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary."""
        if not self._active_plan:
            return ""

        lines = [f"Task: {self._active_plan.task_description[:100]}"]
        lines.append(f"Status: {self._active_plan.status}")
        lines.append("")

        for m in self._active_plan.milestones:
            status_icon = {
                "pending": "⬜",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌",
            }.get(m.status, "❓")
            lines.append(f"{status_icon} [{m.index}] {m.description[:80]}")
            if m.result_summary:
                lines.append(f"   → {m.result_summary[:100]}")

        return "\n".join(lines)

    def load_plan(self, task: str) -> Optional[TaskPlan]:
        """Try to load an existing plan for this task."""
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:16]
        plan_file = self._plans_dir / f"{task_hash}.json"

        if plan_file.exists():
            try:
                data = json.loads(plan_file.read_text())
                plan = TaskPlan.from_dict(data)
                if plan.status == "active":
                    self._active_plan = plan
                    logger.info("sentinel_plan_loaded task_hash=%s", task_hash)
                    return plan
            except Exception as e:
                logger.warning("sentinel_plan_load_failed: %s", e)

        return None

    def _save_plan(self, plan: TaskPlan):
        """Persist plan to disk."""
        plan_file = self._plans_dir / f"{plan.task_hash}.json"
        try:
            plan_file.write_text(json.dumps(plan.to_dict(), indent=2))
        except Exception as e:
            logger.warning("sentinel_plan_save_failed: %s", e)

    def abandon_plan(self):
        """Abandon the current plan."""
        if self._active_plan:
            self._active_plan.status = "abandoned"
            self._active_plan.updated_at = datetime.now().isoformat()
            self._save_plan(self._active_plan)
            self._active_plan = None
