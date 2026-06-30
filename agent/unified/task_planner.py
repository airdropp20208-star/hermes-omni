"""Task Planner — task-level planning with decomposition, tracking, and replanning.

THE MISSING PIECE
-----------------
CognitiveTree (v2) reasons per-ACTION: "should I run `git push` now?"
But many user requests are multi-step: "deploy this app to production"
needs ~8 subtasks (run tests, build, push, ssh to server, restart, verify...).

Without task-level planning, the agent:
1. Loses the big picture — does step 3 without remembering step 1's result
2. Can't recover gracefully — step 5 fails, agent retries step 5 forever
   instead of replanning steps 5-8
3. Can't show progress — user has no idea "where" the agent is

TaskPlanner fixes this:
1. **Decompose** — break user request into ordered subtasks (1 LLM call)
2. **Track** — mark each subtask pending → in_progress → done/failed/skipped
3. **Adapt** — when a subtask fails, replan remaining subtasks (1 LLM call)
4. **Integrate** — subtask dependencies feed into CausalGraph; per-subtask
   reasoning uses CognitiveTree
5. **Persist** — plan survives across turns; agent can resume after interrupt

This is the "executive function" — the agent's prefrontal cortex that
holds the plan in working memory while CognitiveTree handles tactics.

WHEN IT RUNS
------------
- Explicitly: agent calls `task_plan_create` tool when user request is complex
- Heuristically: conversation loop may auto-trigger for requests containing
  "and then", "after that", multiple sentences with action verbs

The planner is OPT-IN. Simple requests ("what's 2+2?") don't need a plan.
The agent decides when to invoke it.

TOKEN ECONOMICS
---------------
- 1 LLM call to decompose (initial plan)
- 1 LLM call per replan (only when subtask fails)
- 0 LLM calls for tracking (pure data structure)
- Progress block injected into system prompt (~200 tokens) replaces the
  agent having to re-derive "where am I?" each turn

Net: 1-2 LLM calls per task, large benefit for multi-step work.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"  # dependency failed


class PlanStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    REPLANNED = "replanned"  # was replanned, now active again


@dataclass
class Subtask:
    """One step in a task plan."""

    subtask_id: str
    description: str  # what to do
    depends_on: list[str] = field(default_factory=list)  # other subtask_ids
    status: str = SubtaskStatus.PENDING.value
    result_summary: str = ""
    attempts: int = 0
    estimated_difficulty: float = 0.5  # 0.0 (trivial) to 1.0 (hard)
    tool_hints: list[str] = field(default_factory=list)  # suggested tools
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    failure_reason: str = ""

    def is_terminal(self) -> bool:
        return self.status in (
            SubtaskStatus.DONE.value,
            SubtaskStatus.FAILED.value,
            SubtaskStatus.SKIPPED.value,
        )

    def is_blocked(self) -> bool:
        return self.status == SubtaskStatus.BLOCKED.value


@dataclass
class TaskPlan:
    """A decomposed plan for a complex user request."""

    plan_id: str
    original_request: str
    subtasks: list[Subtask] = field(default_factory=list)
    status: str = PlanStatus.ACTIVE.value
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    replan_count: int = 0
    context_summary: str = ""  # brief context for replanning
    completed_count: int = 0
    failed_count: int = 0

    def next_actionable(self) -> Subtask | None:
        """Return the next subtask that can be started (pending, deps met)."""
        completed_deps = {
            s.subtask_id for s in self.subtasks
            if s.status in (SubtaskStatus.DONE.value, SubtaskStatus.SKIPPED.value)
        }
        failed_deps = {
            s.subtask_id for s in self.subtasks
            if s.status == SubtaskStatus.FAILED.value
        }
        for subtask in self.subtasks:
            if subtask.status != SubtaskStatus.PENDING.value:
                continue
            # Check if any dependency failed → mark blocked.
            if any(dep in failed_deps for dep in subtask.depends_on):
                subtask.status = SubtaskStatus.BLOCKED.value
                subtask.failure_reason = "Dependency failed"
                continue
            # Check if all deps are done.
            if all(dep in completed_deps for dep in subtask.depends_on):
                return subtask
        return None

    def update_counts(self) -> None:
        self.completed_count = sum(
            1 for s in self.subtasks if s.status == SubtaskStatus.DONE.value
        )
        self.failed_count = sum(
            1 for s in self.subtasks if s.status == SubtaskStatus.FAILED.value
        )

    def is_complete(self) -> bool:
        self.update_counts()
        return all(s.is_terminal() for s in self.subtasks) and bool(self.subtasks)

    def progress_fraction(self) -> float:
        if not self.subtasks:
            return 0.0
        done = sum(1 for s in self.subtasks if s.status == SubtaskStatus.DONE.value)
        return done / len(self.subtasks)

    def to_progress_block(self) -> str:
        """Render as a markdown block for system prompt injection."""
        if not self.subtasks:
            return ""
        self.update_counts()
        total = len(self.subtasks)
        pct = int(self.progress_fraction() * 100)
        lines = [
            "<task-plan>",
            f"Request: {self.original_request[:200]}",
            f"Progress: {self.completed_count}/{total} done ({pct}%), {self.failed_count} failed",
            "",
        ]
        for i, s in enumerate(self.subtasks, 1):
            marker = {
                SubtaskStatus.DONE.value: "✓",
                SubtaskStatus.FAILED.value: "✗",
                SubtaskStatus.IN_PROGRESS.value: "⟳",
                SubtaskStatus.SKIPPED.value: "⊘",
                SubtaskStatus.BLOCKED.value: "⛔",
                SubtaskStatus.PENDING.value: "○",
            }.get(s.status, "○")
            deps = f" (needs: {', '.join(s.depends_on)})" if s.depends_on else ""
            result = f" → {s.result_summary[:80]}" if s.result_summary else ""
            lines.append(f"  {i}. {marker} {s.description[:100]}{deps}{result}")
        next_task = self.next_actionable()
        if next_task is not None:
            lines.append("")
            lines.append(f"Next: {next_task.description[:120]}")
        elif not self.is_complete():
            lines.append("")
            lines.append("Next: replan needed (subtasks blocked or failed)")
        lines.append("</task-plan>")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "original_request": self.original_request,
            "status": self.status,
            "subtasks": [asdict(s) for s in self.subtasks],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "replan_count": self.replan_count,
            "context_summary": self.context_summary,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
        }


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_DECOMPOSE_SYSTEM = (
    "You are the planning layer of an AI agent. The user has made a complex "
    "request that requires multiple steps. Decompose it into ordered subtasks.\n\n"
    "Rules:\n"
    "- Each subtask should be ONE discrete action (not 'do X and Y')\n"
    "- Order subtasks in execution order\n"
    "- Specify dependencies (a subtask that needs another's output)\n"
    "- Rate difficulty 0.0 (trivial) to 1.0 (hard)\n"
    "- Suggest tools (read_file, bash, edit_file, web_search, etc.)\n"
    "- Be concrete — 'run pytest' not 'test the code'\n"
    "- Skip subtasks the user said are already done\n"
    "- 3-10 subtasks is ideal; fewer is too coarse, more is too granular\n\n"
    "Return STRICT JSON:\n"
    "{\n"
    '  "subtasks": [\n'
    "    {\n"
    '      "description": "concrete action",\n'
    '      "depends_on": ["subtask_id_1"] (empty if none),\n'
    '      "estimated_difficulty": 0.0 to 1.0,\n'
    '      "tool_hints": ["tool_name", ...]\n'
    "    }\n"
    "  ],\n"
    '  "context_summary": "brief context for replanning later"\n'
    "}"
)

_REPLAN_SYSTEM = (
    "You are the planning layer of an AI agent. A subtask has FAILED and "
    "you need to replan the remaining work. You see the original plan, "
    "which subtasks succeeded, which failed, and the failure reason.\n\n"
    "Decide:\n"
    "1. Should the failed subtask be retried with a different approach?\n"
    "2. Should it be skipped (and downstream subtasks adjusted)?\n"
    "3. Should new subtasks be added to work around the failure?\n"
    "4. Should downstream subtasks be marked as blocked?\n\n"
    "Return STRICT JSON with the FULL updated plan:\n"
    "{\n"
    '  "action": "retry" | "skip" | "workaround" | "abort",\n'
    '  "rationale": "one sentence",\n'
    '  "updated_subtasks": [\n'
    "    {\n"
    '      "subtask_id": "original or new id",\n'
    '      "description": "may be modified for retry/workaround",\n'
    '      "depends_on": [...],\n'
    '      "status": "pending" | "skipped" | "blocked",\n'
    '      "estimated_difficulty": 0.0 to 1.0,\n'
    '      "tool_hints": [...]\n'
    "    }\n"
    "  ]\n"
    "}"
)


# --------------------------------------------------------------------------- #
# TaskPlanner
# --------------------------------------------------------------------------- #


class TaskPlanner:
    """Decompose, track, and replan complex tasks.

    Maintains an in-memory plan registry. The active plan is injected
    into the system prompt via `get_progress_block()`.
    """

    def __init__(
        self,
        *,
        llm_call=None,
        max_subtasks: int = 15,
        max_replans: int = 3,
        persist_path: str | Path | None = None,
    ) -> None:
        self._llm_call = llm_call
        self._max_subtasks = max(2, min(max_subtasks, 30))
        self._max_replans = max(1, min(max_replans, 10))
        self._plans: dict[str, TaskPlan] = {}
        self._active_plan_id: str | None = None
        self._persist_path = Path(persist_path).expanduser() if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def create_plan(
        self,
        *,
        request: str,
        context: str = "",
    ) -> TaskPlan | None:
        """Decompose a request into a plan. Returns the plan, or None on failure."""
        if self._llm_call is None:
            return None
        try:
            user = (
                f"User request:\n{request}\n\n"
                f"Additional context:\n{context or '(none)'}\n\n"
                "Decompose into subtasks now."
            )
            raw = self._llm_call(_DECOMPOSE_SYSTEM, user)
            data = self._parse_json(raw)
            if data is None:
                return None
            subtasks_data = data.get("subtasks", [])
            if not subtasks_data:
                return None
            subtasks: list[Subtask] = []
            for i, st_data in enumerate(subtasks_data[: self._max_subtasks]):
                if not isinstance(st_data, dict):
                    continue
                subtask_id = f"st_{i + 1:02d}"
                subtasks.append(
                    Subtask(
                        subtask_id=subtask_id,
                        description=str(st_data.get("description", "")).strip(),
                        depends_on=[
                            str(d).strip() for d in st_data.get("depends_on", []) if str(d).strip()
                        ],
                        estimated_difficulty=max(
                            0.0, min(1.0, float(st_data.get("estimated_difficulty", 0.5)))
                        ),
                        tool_hints=[
                            str(t).strip() for t in st_data.get("tool_hints", []) if str(t).strip()
                        ],
                    )
                )
            if not subtasks:
                return None
            plan = TaskPlan(
                plan_id=f"plan_{uuid.uuid4().hex[:12]}",
                original_request=request,
                subtasks=subtasks,
                context_summary=str(data.get("context_summary", "")).strip() or context,
            )
            self._plans[plan.plan_id] = plan
            self._active_plan_id = plan.plan_id
            self._persist()
            return plan
        except Exception:
            return None

    def get_active_plan(self) -> TaskPlan | None:
        if self._active_plan_id is None:
            return None
        return self._plans.get(self._active_plan_id)

    def set_active_plan(self, plan_id: str) -> bool:
        if plan_id not in self._plans:
            return False
        self._active_plan_id = plan_id
        self._persist()
        return True

    def start_subtask(self, subtask_id: str) -> bool:
        """Mark a subtask as in_progress. Returns False if not actionable."""
        plan = self.get_active_plan()
        if plan is None:
            return False
        subtask = next((s for s in plan.subtasks if s.subtask_id == subtask_id), None)
        if subtask is None:
            return False
        if subtask.status != SubtaskStatus.PENDING.value:
            return False
        # Check deps.
        completed = {
            s.subtask_id for s in plan.subtasks
            if s.status in (SubtaskStatus.DONE.value, SubtaskStatus.SKIPPED.value)
        }
        if not all(dep in completed for dep in subtask.depends_on):
            return False
        subtask.status = SubtaskStatus.IN_PROGRESS.value
        subtask.started_at = time.time()
        subtask.attempts += 1
        plan.updated_at = time.time()
        self._persist()
        return True

    def complete_subtask(
        self,
        subtask_id: str,
        *,
        result_summary: str = "",
    ) -> bool:
        """Mark a subtask as done."""
        plan = self.get_active_plan()
        if plan is None:
            return False
        subtask = next((s for s in plan.subtasks if s.subtask_id == subtask_id), None)
        if subtask is None:
            return False
        subtask.status = SubtaskStatus.DONE.value
        subtask.result_summary = result_summary
        subtask.completed_at = time.time()
        plan.updated_at = time.time()
        plan.update_counts()
        # Check if plan is complete.
        if plan.is_complete():
            plan.status = PlanStatus.COMPLETED.value
        self._persist()
        return True

    def fail_subtask(
        self,
        subtask_id: str,
        *,
        failure_reason: str = "",
    ) -> bool:
        """Mark a subtask as failed. Triggers replanning if LLM available."""
        plan = self.get_active_plan()
        if plan is None:
            return False
        subtask = next((s for s in plan.subtasks if s.subtask_id == subtask_id), None)
        if subtask is None:
            return False
        subtask.status = SubtaskStatus.FAILED.value
        subtask.failure_reason = failure_reason
        subtask.completed_at = time.time()
        plan.updated_at = time.time()
        plan.update_counts()
        # Mark downstream subtasks as blocked.
        self._mark_blocked_downstream(plan, subtask_id)
        self._persist()
        # Try to replan if LLM available and replans remaining.
        if self._llm_call is not None and plan.replan_count < self._max_replans:
            self._replan(plan, failed_subtask_id=subtask_id, failure_reason=failure_reason)
        return True

    def skip_subtask(self, subtask_id: str, *, reason: str = "") -> bool:
        """Skip a subtask. Downstream subtasks proceed (dependency considered met)."""
        plan = self.get_active_plan()
        if plan is None:
            return False
        subtask = next((s for s in plan.subtasks if s.subtask_id == subtask_id), None)
        if subtask is None:
            return False
        subtask.status = SubtaskStatus.SKIPPED.value
        subtask.failure_reason = reason
        subtask.completed_at = time.time()
        plan.updated_at = time.time()
        plan.update_counts()
        if plan.is_complete():
            plan.status = PlanStatus.COMPLETED.value
        self._persist()
        return True

    def abandon_plan(self, *, reason: str = "") -> bool:
        """Abandon the active plan."""
        plan = self.get_active_plan()
        if plan is None:
            return False
        plan.status = PlanStatus.ABANDONED.value
        plan.updated_at = time.time()
        self._active_plan_id = None
        self._persist()
        return True

    def get_progress_block(self) -> str:
        """Return the active plan as a prompt block. Empty if no active plan."""
        plan = self.get_active_plan()
        if plan is None or plan.status != PlanStatus.ACTIVE.value:
            return ""
        return plan.to_progress_block()

    def list_plans(self) -> list[dict[str, Any]]:
        return [
            {
                "plan_id": p.plan_id,
                "request": p.original_request[:200],
                "status": p.status,
                "progress": f"{p.completed_count}/{len(p.subtasks)}",
                "replan_count": p.replan_count,
                "created_at": p.created_at,
            }
            for p in self._plans.values()
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "total_plans": len(self._plans),
            "active_plan": self._active_plan_id,
            "plans_by_status": {
                status: sum(1 for p in self._plans.values() if p.status == status)
                for status in (PlanStatus.ACTIVE.value, PlanStatus.COMPLETED.value, PlanStatus.ABANDONED.value)
            },
        }

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _mark_blocked_downstream(self, plan: TaskPlan, failed_id: str) -> None:
        """Mark subtasks that depend (transitively) on failed_id as blocked."""
        blocked_ids = {failed_id}
        changed = True
        while changed:
            changed = False
            for subtask in plan.subtasks:
                if subtask.status != SubtaskStatus.PENDING.value:
                    continue
                if any(dep in blocked_ids for dep in subtask.depends_on):
                    subtask.status = SubtaskStatus.BLOCKED.value
                    subtask.failure_reason = f"Upstream subtask {failed_id} failed"
                    blocked_ids.add(subtask.subtask_id)
                    changed = True

    def _replan(self, plan: TaskPlan, *, failed_subtask_id: str, failure_reason: str) -> bool:
        """Replan after a subtask failure. Returns True if replan succeeded."""
        try:
            user = (
                f"Original request: {plan.original_request}\n\n"
                f"Plan so far:\n{self._format_plan_for_replan(plan)}\n\n"
                f"Failed subtask: {failed_subtask_id}\n"
                f"Failure reason: {failure_reason}\n\n"
                "Replan the remaining work."
            )
            raw = self._llm_call(_REPLAN_SYSTEM, user)
            data = self._parse_json(raw)
            if data is None:
                return False
            action = str(data.get("action", "retry")).strip().lower()
            updated = data.get("updated_subtasks", [])
            if not isinstance(updated, list):
                return False
            # Apply updates.
            for st_update in updated:
                if not isinstance(st_update, dict):
                    continue
                st_id = str(st_update.get("subtask_id", "")).strip()
                if not st_id:
                    continue
                # Find existing subtask or create new.
                subtask = next((s for s in plan.subtasks if s.subtask_id == st_id), None)
                if subtask is None:
                    # New subtask added by replan.
                    subtask = Subtask(
                        subtask_id=st_id,
                        description=str(st_update.get("description", "")).strip(),
                        depends_on=[
                            str(d).strip() for d in st_update.get("depends_on", []) if str(d).strip()
                        ],
                    )
                    plan.subtasks.append(subtask)
                else:
                    # Update existing.
                    new_desc = str(st_update.get("description", "")).strip()
                    if new_desc:
                        subtask.description = new_desc
                    new_deps = [str(d).strip() for d in st_update.get("depends_on", []) if str(d).strip()]
                    if new_deps:
                        subtask.depends_on = new_deps
                    new_status = str(st_update.get("status", "")).strip().lower()
                    if new_status in (
                        SubtaskStatus.PENDING.value,
                        SubtaskStatus.SKIPPED.value,
                        SubtaskStatus.BLOCKED.value,
                    ):
                        subtask.status = new_status
                # Update difficulty/tool hints if provided.
                try:
                    subtask.estimated_difficulty = max(
                        0.0, min(1.0, float(st_update.get("estimated_difficulty", subtask.estimated_difficulty)))
                    )
                except (TypeError, ValueError):
                    pass
                new_tools = [str(t).strip() for t in st_update.get("tool_hints", []) if str(t).strip()]
                if new_tools:
                    subtask.tool_hints = new_tools
            plan.replan_count += 1
            plan.status = PlanStatus.ACTIVE.value  # re-activate
            plan.update_counts()
            self._persist()
            return True
        except Exception:
            return False

    @staticmethod
    def _format_plan_for_replan(plan: TaskPlan) -> str:
        lines = []
        for s in plan.subtasks:
            deps = f" (deps: {', '.join(s.depends_on)})" if s.depends_on else ""
            result = f" → {s.result_summary[:60]}" if s.result_summary else ""
            fail = f" [FAILED: {s.failure_reason[:60]}]" if s.failure_reason else ""
            lines.append(f"  {s.subtask_id} [{s.status}] {s.description[:80]}{deps}{result}{fail}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active_plan_id": self._active_plan_id,
                "plans": {pid: p.to_dict() for pid, p in self._plans.items()},
            }
            self._persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._active_plan_id = data.get("active_plan_id")
            for pid, pdata in data.get("plans", {}).items():
                subtasks = [
                    Subtask(**{k: v for k, v in st.items() if k in Subtask.__dataclass_fields__})
                    for st in pdata.get("subtasks", [])
                ]
                plan = TaskPlan(
                    plan_id=pid,
                    original_request=pdata.get("original_request", ""),
                    subtasks=subtasks,
                    status=pdata.get("status", PlanStatus.ACTIVE.value),
                    created_at=pdata.get("created_at", time.time()),
                    updated_at=pdata.get("updated_at", time.time()),
                    replan_count=pdata.get("replan_count", 0),
                    context_summary=pdata.get("context_summary", ""),
                )
                plan.update_counts()
                self._plans[pid] = plan
        except Exception:
            pass

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


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_planner: TaskPlanner | None = None


def get_planner() -> TaskPlanner | None:
    return _planner


def configure_planner(
    *,
    llm_call=None,
    max_subtasks: int = 15,
    max_replans: int = 3,
    persist_path: str | Path | None = None,
) -> TaskPlanner | None:
    global _planner
    if llm_call is None:
        _planner = None
        return None
    if persist_path is None:
        from hermes_constants import get_hermes_home

        persist_path = get_hermes_home() / "unified" / "task_plans.json"
    _planner = TaskPlanner(
        llm_call=llm_call,
        max_subtasks=max_subtasks,
        max_replans=max_replans,
        persist_path=persist_path,
    )
    return _planner
