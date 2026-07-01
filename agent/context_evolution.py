"""Context Evolution — Auto-learning from failures and user feedback.

Inspired by OmniAgent's Context Evolution, adapted for hermes-omni.

Features:
1. Lesson extraction from failed executions and reflections
2. User feedback detection (corrections, preferences)
3. Automatic promotion of validated lessons to AGENTS.md
4. Evidence-based validation (lessons confirmed multiple times)

Architecture:
  LessonRecorder → .hermes/learnings/lessons.jsonl (append-only)
  LessonAnalyzer → checks evidence count, promotes to AGENTS.md
  ContextEvolutionManager → orchestrator
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Lesson:
    """A single recorded lesson."""

    timestamp: str
    lesson_hash: str
    source: str  # "failure", "reflection", "user_feedback", "error_recovery"
    category: str  # "approach", "preference", "constraint", "workflow"
    lesson: str
    context: str
    evidence: int = 1
    promoted: bool = False
    promoted_at: Optional[str] = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> Optional["Lesson"]:
        try:
            data = json.loads(line)
            return cls(**data)
        except Exception:
            return None


class LessonRecorder:
    """Records lessons from agent execution outcomes to append-only JSONL."""

    def __init__(self, learnings_dir: Path, max_learnings: int = 100):
        self.learnings_dir = learnings_dir
        self.max_learnings = max_learnings
        self.learnings_dir.mkdir(parents=True, exist_ok=True)
        self._lessons_file = self.learnings_dir / "lessons.jsonl"
        self._index: Dict[str, Lesson] = {}
        self._load_index()

    def _load_index(self):
        """Load existing lessons into memory index."""
        if not self._lessons_file.exists():
            return
        try:
            with open(self._lessons_file, "r", encoding="utf-8") as f:
                for line in f:
                    lesson = Lesson.from_jsonl(line.strip())
                    if lesson:
                        self._index[lesson.lesson_hash] = lesson
        except Exception as e:
            logger.warning("lesson_load_failed: %s", e)

    def record(
        self,
        source: str,
        category: str,
        lesson: str,
        context: str,
    ) -> Optional[Lesson]:
        """Record a lesson. If it already exists, increment evidence count."""
        lesson_hash = hashlib.sha256(
            f"{category}:{lesson}".encode()
        ).hexdigest()[:16]

        if lesson_hash in self._index:
            # Already exists — increment evidence
            existing = self._index[lesson_hash]
            existing.evidence += 1
            self._rewrite_file()
            logger.info(
                "lesson_evidence_incremented hash=%s evidence=%d",
                lesson_hash, existing.evidence,
            )
            return existing

        # New lesson
        new_lesson = Lesson(
            timestamp=datetime.now().isoformat(),
            lesson_hash=lesson_hash,
            source=source,
            category=category,
            lesson=lesson,
            context=context[:500],
        )
        self._index[lesson_hash] = new_lesson

        # Append to file
        try:
            with open(self._lessons_file, "a", encoding="utf-8") as f:
                f.write(new_lesson.to_jsonl() + "\n")
        except Exception as e:
            logger.warning("lesson_write_failed: %s", e)

        # Enforce max limit
        self._enforce_limit()

        logger.info(
            "lesson_recorded hash=%s source=%s category=%s",
            lesson_hash, source, category,
        )
        return new_lesson

    def _rewrite_file(self):
        """Rewrite the entire lessons file from index."""
        try:
            with open(self._lessons_file, "w", encoding="utf-8") as f:
                for lesson in self._index.values():
                    f.write(lesson.to_jsonl() + "\n")
        except Exception as e:
            logger.warning("lesson_rewrite_failed: %s", e)

    def _enforce_limit(self):
        """Remove oldest lessons if over limit."""
        if len(self._index) <= self.max_learnings:
            return
        sorted_lessons = sorted(
            self._index.values(), key=lambda l: l.timestamp
        )
        to_remove = sorted_lessons[: len(self._index) - self.max_learnings]
        for lesson in to_remove:
            del self._index[lesson.lesson_hash]
        self._rewrite_file()

    def get_unpromoted(self, min_evidence: int = 2) -> List[Lesson]:
        """Get lessons that haven't been promoted yet and have enough evidence."""
        return [
            l for l in self._index.values()
            if not l.promoted and l.evidence >= min_evidence
        ]

    def get_all(self) -> List[Lesson]:
        return list(self._index.values())

    def mark_promoted(self, lesson_hash: str):
        """Mark a lesson as promoted to AGENTS.md."""
        if lesson_hash in self._index:
            self._index[lesson_hash].promoted = True
            self._index[lesson_hash].promoted_at = datetime.now().isoformat()
            self._rewrite_file()


class LessonExtractor:
    """Extracts lessons from agent execution outcomes."""

    @staticmethod
    def from_failure(
        task: str,
        error: str,
        conversation_history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, str]]:
        """Extract a lesson from a failed execution."""
        if not error:
            return None

        # Analyze the error pattern
        error_lower = error.lower()

        if "timeout" in error_lower or "timed out" in error_lower:
            return {
                "source": "failure",
                "category": "approach",
                "lesson": "Use longer timeouts or async operations for slow tasks",
                "context": f"Task '{task[:100]}' failed with: {error[:200]}",
            }

        if "permission" in error_lower or "access denied" in error_lower:
            return {
                "source": "failure",
                "category": "constraint",
                "lesson": "Check file permissions before attempting write operations",
                "context": f"Task '{task[:100]}' failed with: {error[:200]}",
            }

        if "not found" in error_lower or "no such file" in error_lower:
            return {
                "source": "failure",
                "category": "approach",
                "lesson": "Verify file/path existence before operations",
                "context": f"Task '{task[:100]}' failed with: {error[:200]}",
            }

        if "rate limit" in error_lower or "429" in error:
            return {
                "source": "failure",
                "category": "constraint",
                "lesson": "Implement rate limiting and backoff for API calls",
                "context": f"Task '{task[:100]}' failed with: {error[:200]}",
            }

        # Generic failure lesson
        return {
            "source": "failure",
            "category": "approach",
            "lesson": f"When encountering '{error[:100]}', try alternative approach",
            "context": f"Task '{task[:100]}'",
        }

    @staticmethod
    def from_reflection(reflection: str) -> Optional[Dict[str, str]]:
        """Extract a lesson from a self-reflection."""
        if not reflection or len(reflection) < 20:
            return None

        return {
            "source": "reflection",
            "category": "approach",
            "lesson": reflection[:300],
            "context": "Self-reflection after failed attempt",
        }

    @staticmethod
    def from_user_feedback(feedback: str) -> Optional[Dict[str, str]]:
        """Extract a lesson from user feedback/correction."""
        if not feedback or len(feedback) < 5:
            return None

        # Classify feedback type
        feedback_lower = feedback.lower()

        if any(w in feedback_lower for w in ["don't", "stop", "never", "avoid", "不要", "别"]):
            category = "constraint"
        elif any(w in feedback_lower for w in ["prefer", "want", "like", "better", "喜欢", "更"]):
            category = "preference"
        else:
            category = "workflow"

        return {
            "source": "user_feedback",
            "category": category,
            "lesson": feedback[:300],
            "context": "User correction/preference",
        }


class ContextEvolutionManager:
    """Orchestrates context evolution — learning from experience.

    Usage:
        evolution = ContextEvolutionManager(work_dir=Path("./workspace"))

        # After a failed execution
        evolution.on_failure(task, error, history)

        # After user feedback
        evolution.on_user_feedback("Don't use that approach, do this instead")

        # Check for promotable lessons
        to_promote = evolution.get_promotable_lessons()
        for lesson in to_promote:
            evolution.promote_to_agents_md(lesson)
    """

    def __init__(
        self,
        work_dir: Path,
        promotion_threshold: int = 2,
        auto_promote: bool = False,
    ):
        """
        Args:
            work_dir: Working directory
            promotion_threshold: Minimum evidence count before promotion
            auto_promote: Automatically promote lessons to AGENTS.md
        """
        self.work_dir = work_dir
        self.promotion_threshold = promotion_threshold
        self.auto_promote = auto_promote

        learnings_dir = work_dir / ".hermes" / "learnings"
        self.recorder = LessonRecorder(learnings_dir)
        self.extractor = LessonExtractor()

    def on_failure(
        self,
        task: str,
        error: str,
        conversation_history: List[Dict[str, Any]] = None,
    ):
        """Record lessons from a failed execution."""
        lesson_data = self.extractor.from_failure(task, error, conversation_history or [])
        if lesson_data:
            self.recorder.record(**lesson_data)

    def on_reflection(self, reflection: str):
        """Record lessons from a self-reflection."""
        lesson_data = self.extractor.from_reflection(reflection)
        if lesson_data:
            self.recorder.record(**lesson_data)

    def on_user_feedback(self, feedback: str):
        """Record lessons from user feedback."""
        lesson_data = self.extractor.from_user_feedback(feedback)
        if lesson_data:
            self.recorder.record(**lesson_data)

    def get_promotable_lessons(self) -> List[Lesson]:
        """Get lessons ready for promotion to AGENTS.md."""
        return self.recorder.get_unpromoted(min_evidence=self.promotion_threshold)

    def promote_to_agents_md(self, lesson: Lesson) -> bool:
        """Promote a lesson to AGENTS.md as a learned rule."""
        agents_md_path = self.work_dir / "AGENTS.md"

        try:
            # Read existing content
            content = ""
            if agents_md_path.exists():
                content = agents_md_path.read_text(encoding="utf-8")

            # Check if already promoted
            if lesson.lesson_hash in content:
                self.recorder.mark_promoted(lesson.lesson_hash)
                return True

            # Find or create the "Learned Rules" section
            section_header = "## Learned Rules (Auto-generated)"
            if section_header not in content:
                content += f"\n\n{section_header}\n\n"

            # Add the lesson
            rule_entry = (
                f"- **[{lesson.category}]** {lesson.lesson}\n"
                f"  <!-- hash: {lesson.lesson_hash} | "
                f"evidence: {lesson.evidence} | "
                f"source: {lesson.source} -->\n"
            )

            # Insert before the last newline
            insert_pos = content.rfind(section_header) + len(section_header)
            content = content[:insert_pos] + "\n" + rule_entry + content[insert_pos:]

            # Write back
            agents_md_path.write_text(content, encoding="utf-8")
            self.recorder.mark_promoted(lesson.lesson_hash)

            logger.info(
                "lesson_promoted hash=%s category=%s",
                lesson.lesson_hash, lesson.category,
            )
            return True

        except Exception as e:
            logger.warning("lesson_promotion_failed: %s", e)
            return False

    def auto_promote_lessons(self) -> int:
        """Auto-promote all eligible lessons. Returns count promoted."""
        if not self.auto_promote:
            return 0

        promotable = self.get_promotable_lessons()
        promoted = 0
        for lesson in promotable:
            if self.promote_to_agents_md(lesson):
                promoted += 1

        return promoted

    def get_stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        all_lessons = self.recorder.get_all()
        return {
            "total_lessons": len(all_lessons),
            "promoted": sum(1 for l in all_lessons if l.promoted),
            "unpromoted": sum(1 for l in all_lessons if not l.promoted),
            "by_source": self._count_by(all_lessons, "source"),
            "by_category": self._count_by(all_lessons, "category"),
        }

    @staticmethod
    def _count_by(lessons: List[Lesson], attr: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for l in lessons:
            key = getattr(l, attr, "unknown")
            counts[key] = counts.get(key, 0) + 1
        return counts
