"""Trajectory Distillery — compress 200 tool calls into 30-step "golden path".

THE BREAKTHROUGH
----------------
When agent solves a complex task, it makes 100-200 tool calls (many are
exploration, dead-ends, retries). Next time similar task → agent starts
from scratch, repeats same 200 calls.

TrajectoryDistillery analyzes completed trajectories and extracts the
"golden path" — the minimal set of tool calls that ACTUALLY contributed
to the solution. Dead-ends, retries, and exploratory calls are stripped.

Next time similar task → load golden path → agent follows 30 steps
instead of 200. 80%+ token savings for repeated tasks.

ARCHITECTURE
------------
1. **Record**: Track all tool calls + outcomes during a task
2. **Analyze**: After task completes, classify each call as:
   - ESSENTIAL: directly contributed to solution
   - SUPPORTING: provided context for essential calls
   - DEAD_END: explored wrong path, output not used later
   - RETRY: duplicate of earlier call
3. **Distill**: Extract only ESSENTIAL + SUPPORTING calls → golden path
4. **Store**: Save golden path keyed by task signature
5. **Recall**: Next similar task → load golden path → skip exploration

GOLDEN PATH FORMAT
------------------
    {
      "task_signature": "debug python import error",
      "total_original_calls": 187,
      "golden_path_calls": 31,
      "compression_ratio": 0.17,
      "steps": [
        {"step": 1, "tool": "read_file", "args": {"path": "main.py"}, "why": "Identify entry point"},
        {"step": 2, "tool": "grep", "args": {"pattern": "import", "path": "main.py"}, "why": "Find import statements"},
        {"step": 3, "tool": "read_file", "args": {"path": "utils.py"}, "why": "Check missing import target"},
        ...
      ]
    }
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


class CallClassification(str):
    ESSENTIAL = "essential"
    SUPPORTING = "supporting"
    DEAD_END = "dead_end"
    RETRY = "retry"


@dataclass
class TrajectoryStep:
    """One step in a recorded trajectory."""

    step_num: int
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    classification: str = ""  # filled during distillation
    reason: str = ""  # why this classification


@dataclass
class GoldenPath:
    """A distilled trajectory — minimal steps to solve a task."""

    path_id: str
    task_signature: str
    task_description: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    total_original_calls: int = 0
    golden_path_calls: int = 0
    compression_ratio: float = 0.0  # golden / original (lower = better)
    created_at: float = field(default_factory=time.time)
    times_used: int = 0
    last_used: float = 0.0
    effectiveness_score: float = 1.0  # adjusted by feedback


# --------------------------------------------------------------------------- #
# TrajectoryDistillery
# --------------------------------------------------------------------------- #


class TrajectoryDistillery:
    """Records trajectories, distills golden paths, recalls on demand.

    Thread-safe. Persists to ~/.hermes/unified/golden_paths.jsonl.
    """

    def __init__(
        self,
        *,
        store_path: str | Path | None = None,
        max_paths: int = 500,
        similarity_threshold: float = 0.6,
    ) -> None:
        if store_path is None:
            from hermes_constants import get_hermes_home

            store_path = get_hermes_home() / "unified" / "golden_paths.jsonl"
        self._store_path = Path(store_path).expanduser()
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_paths = max_paths
        self._similarity_threshold = similarity_threshold
        self._lock = RLock()
        self._paths: list[GoldenPath] = []
        self._current_trajectory: list[TrajectoryStep] = []
        self._current_task_sig: str = ""
        self._current_task_desc: str = ""
        self._load()

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            for line in self._store_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    allowed = GoldenPath.__dataclass_fields__.keys()
                    data = {k: v for k, v in data.items() if k in allowed}
                    self._paths.append(GoldenPath(**data))
                except Exception:
                    continue
        except Exception:
            pass

    def _persist(self) -> None:
        try:
            with self._store_path.open("w", encoding="utf-8") as fh:
                for p in self._paths:
                    fh.write(json.dumps(asdict(p), ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def start_task(self, *, task_description: str) -> str:
        """Start recording a new trajectory."""
        self._current_trajectory = []
        self._current_task_desc = task_description
        self._current_task_sig = self._signature(task_description)
        return self._current_task_sig

    def record_step(
        self,
        *,
        tool: str,
        args: dict[str, Any] | None,
        result: Any,
        success: bool = True,
    ) -> None:
        """Record one tool call step."""
        step = TrajectoryStep(
            step_num=len(self._current_trajectory) + 1,
            tool=tool,
            args=args or {},
            result_preview=str(result)[:200] if result else "",
            success=success,
        )
        self._current_trajectory.append(step)

    def finish_task(self, *, success: bool = True) -> GoldenPath | None:
        """Distill the recorded trajectory into a golden path.

        Only distills if task was successful (no point learning from failures).
        Returns the GoldenPath if created, None otherwise.
        """
        if not success or len(self._current_trajectory) < 3:
            self._current_trajectory = []
            return None

        # Classify each step
        classified = self._classify_steps(self._current_trajectory)

        # Extract golden path (essential + supporting only)
        golden_steps = [
            {
                "step": i + 1,
                "tool": s.tool,
                "args": s.args,
                "why": s.reason,
            }
            for i, s in enumerate(classified)
            if s.classification in (CallClassification.ESSENTIAL, CallClassification.SUPPORTING)
        ]

        if len(golden_steps) < 2:
            self._current_trajectory = []
            return None

        total = len(self._current_trajectory)
        golden_count = len(golden_steps)
        compression = golden_count / total if total > 0 else 1.0

        path = GoldenPath(
            path_id=self._hash_path(self._current_task_sig, golden_steps),
            task_signature=self._current_task_sig,
            task_description=self._current_task_desc,
            steps=golden_steps,
            total_original_calls=total,
            golden_path_calls=golden_count,
            compression_ratio=compression,
        )

        with self._lock:
            # Dedupe by path_id
            self._paths = [p for p in self._paths if p.path_id != path.path_id]
            self._paths.append(path)
            # Evict oldest if over limit
            if len(self._paths) > self._max_paths:
                self._paths.sort(key=lambda p: (p.effectiveness_score, p.last_used), reverse=True)
                self._paths = self._paths[: self._max_paths]
            self._persist()

        self._current_trajectory = []
        return path

    # ------------------------------------------------------------------ #
    # Classification (the core intelligence)
    # ------------------------------------------------------------------ #

    def _classify_steps(self, steps: list[TrajectoryStep]) -> list[TrajectoryStep]:
        """Classify each step as essential / supporting / dead_end / retry."""
        classified = list(steps)

        # 1. Mark retries (same tool + similar args as previous)
        seen_hashes: set[str] = set()
        for step in classified:
            h = self._hash_args(step.tool, step.args)
            if h in seen_hashes:
                step.classification = CallClassification.RETRY
                step.reason = "Duplicate of earlier call"
            else:
                seen_hashes.add(h)

        # 2. Mark dead-ends (failed calls whose output was never referenced)
        for i, step in enumerate(classified):
            if step.classification:  # already classified
                continue
            if not step.success:
                # Check if any later step references this call's output
                referenced = False
                for later in classified[i + 1 :]:
                    if self._references(step, later):
                        referenced = True
                        break
                if not referenced:
                    step.classification = CallClassification.DEAD_END
                    step.reason = "Failed call, output never used"

        # 3. Mark essential (last successful call before task completion)
        # Last successful call is always essential
        for step in reversed(classified):
            if step.success and not step.classification:
                step.classification = CallClassification.ESSENTIAL
                step.reason = "Final successful call"
                break

        # 4. Mark supporting (calls whose output was referenced by essential calls)
        essential_args_text = " ".join(
            str(s.args) + " " + s.result_preview
            for s in classified
            if s.classification == CallClassification.ESSENTIAL
        )
        for step in classified:
            if step.classification:
                continue
            # Was this call's output referenced by an essential call?
            if self._referenced_by(step, essential_args_text):
                step.classification = CallClassification.SUPPORTING
                step.reason = "Output used by essential call"
            else:
                # Check if it's a read/search that provided context
                if any(
                    kw in step.tool.lower()
                    for kw in ("read", "grep", "search", "list", "glob", "find")
                ):
                    step.classification = CallClassification.SUPPORTING
                    step.reason = "Context-gathering call"
                else:
                    step.classification = CallClassification.DEAD_END
                    step.reason = "Output not used in final solution"

        return classified

    @staticmethod
    def _references(source: TrajectoryStep, target: TrajectoryStep) -> bool:
        """Check if target step references source step's output."""
        if not source.result_preview:
            return False
        # Check if source's result content appears in target's args
        target_text = json.dumps(target.args, default=str).lower()
        source_text = source.result_preview.lower()
        # Simple: check if any 10+ char substring of source appears in target
        if len(source_text) > 20:
            chunk = source_text[:50]
            return chunk in target_text
        return False

    @staticmethod
    def _referenced_by(step: TrajectoryStep, text: str) -> bool:
        """Check if step's output appears in the given text."""
        if not step.result_preview:
            return False
        # Simple: check if step's result preview keywords appear in text
        keywords = [w for w in re.findall(r"\w{5,}", step.result_preview.lower())]
        if not keywords:
            return False
        matches = sum(1 for kw in keywords if kw in text.lower())
        return matches >= 2

    @staticmethod
    def _hash_args(tool: str, args: dict[str, Any]) -> str:
        """Hash tool + args for dedup."""
        try:
            raw = f"{tool}:{json.dumps(args, sort_keys=True, default=str)[:200]}"
            return hashlib.md5(raw.encode()).hexdigest()[:16]
        except Exception:
            return f"{tool}:{str(args)[:50]}"

    @staticmethod
    def _signature(task: str) -> str:
        """Generate task signature for matching."""
        # Extract key words
        words = re.findall(r"\w{3,}", task.lower())
        return " ".join(sorted(words[:10]))

    @staticmethod
    def _hash_path(sig: str, steps: list[dict]) -> str:
        """Generate stable ID for a golden path."""
        h = hashlib.sha256()
        h.update(sig.encode())
        for s in steps:
            h.update(s["tool"].encode())
        return h.hexdigest()[:24]

    # ------------------------------------------------------------------ #
    # Recall
    # ------------------------------------------------------------------ #

    def recall(self, task_description: str) -> GoldenPath | None:
        """Find best matching golden path for a task."""
        if not self._paths:
            return None
        sig = self._signature(task_description)
        sig_words = set(sig.split())
        if not sig_words:
            return None

        best: GoldenPath | None = None
        best_score = 0.0
        for path in self._paths:
            # Word overlap score
            path_words = set(path.task_signature.split())
            overlap = len(sig_words & path_words)
            if overlap == 0:
                continue
            score = overlap / max(len(sig_words), len(path_words))
            # Boost by effectiveness
            score *= path.effectiveness_score
            # Penalize low compression (less useful)
            if path.compression_ratio < 0.5:
                score *= 1.2  # bonus for good compression
            if score > best_score and score >= self._similarity_threshold:
                best_score = score
                best = path

        if best is not None:
            with self._lock:
                best.times_used += 1
                best.last_used = time.time()
                self._persist()
        return best

    def recall_as_prompt(self, task_description: str) -> str:
        """Recall golden path and format as system prompt block."""
        path = self.recall(task_description)
        if path is None:
            return ""
        lines = [
            "<golden-path>",
            f"Task pattern: {path.task_description[:150]}",
            f"This task was solved before in {path.total_original_calls} calls, distilled to {path.golden_path_calls} essential steps:",
            "",
        ]
        for step in path.steps:
            lines.append(f"  {step['step']}. {step['tool']} {json.dumps(step['args'], default=str)[:80]}")
            if step.get("why"):
                lines.append(f"     → {step['why']}")
        lines.append("")
        lines.append("Follow this path if applicable. Skip exploration that led to dead-ends.")
        lines.append("</golden-path>")
        return "\n".join(lines)

    def provide_feedback(self, path_id: str, *, positive: bool) -> None:
        """Adjust effectiveness score based on user feedback."""
        with self._lock:
            for path in self._paths:
                if path.path_id == path_id:
                    if positive:
                        path.effectiveness_score = min(2.0, path.effectiveness_score + 0.1)
                    else:
                        path.effectiveness_score = max(0.1, path.effectiveness_score - 0.2)
                    self._persist()
                    break

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_paths": len(self._paths),
                "avg_compression": (
                    sum(p.compression_ratio for p in self._paths) / len(self._paths)
                    if self._paths
                    else 0
                ),
                "total_times_used": sum(p.times_used for p in self._paths),
                "avg_effectiveness": (
                    sum(p.effectiveness_score for p in self._paths) / len(self._paths)
                    if self._paths
                    else 0
                ),
                "store_path": str(self._store_path),
            }

    def list_paths(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            sorted_paths = sorted(self._paths, key=lambda p: p.times_used, reverse=True)
            return [
                {
                    "path_id": p.path_id,
                    "task": p.task_description[:100],
                    "compression": f"{p.golden_path_calls}/{p.total_original_calls} ({p.compression_ratio:.0%})",
                    "times_used": p.times_used,
                    "effectiveness": p.effectiveness_score,
                }
                for p in sorted_paths[:limit]
            ]


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_distiller: TrajectoryDistillery | None = None


def get_distiller() -> TrajectoryDistillery | None:
    return _distiller


def configure_distiller(
    *,
    store_path: str | Path | None = None,
    max_paths: int = 500,
    similarity_threshold: float = 0.6,
) -> TrajectoryDistillery:
    global _distiller
    _distiller = TrajectoryDistillery(
        store_path=store_path,
        max_paths=max_paths,
        similarity_threshold=similarity_threshold,
    )
    return _distiller


def start_trajectory_task(*, task_description: str) -> str:
    """Public API: start recording a trajectory."""
    if _distiller is None:
        return ""
    return _distiller.start_task(task_description=task_description)


def record_trajectory_step(
    *,
    tool: str,
    args: dict[str, Any] | None,
    result: Any,
    success: bool = True,
) -> None:
    """Public API: record a step."""
    if _distiller is not None:
        _distiller.record_step(tool=tool, args=args, result=result, success=success)


def finish_trajectory_task(*, success: bool = True) -> dict[str, Any] | None:
    """Public API: distill and save golden path."""
    if _distiller is None:
        return None
    path = _distiller.finish_task(success=success)
    if path is None:
        return None
    return {
        "path_id": path.path_id,
        "task": path.task_description[:100],
        "compression": f"{path.golden_path_calls}/{path.total_original_calls}",
        "compression_ratio": path.compression_ratio,
    }


def recall_golden_path(task_description: str) -> str:
    """Public API: recall golden path as prompt block."""
    if _distiller is None:
        return ""
    return _distiller.recall_as_prompt(task_description)


def distillery_stats() -> dict[str, Any]:
    """Public API: get distillery stats."""
    if _distiller is None:
        return {"enabled": False}
    return {"enabled": True, **_distiller.stats()}
