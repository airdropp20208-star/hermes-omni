"""Tests for OmniAgent-inspired modules: Deep Reflexion, Guardian, Sentinel, Context Evolution."""

import json
import tempfile
from pathlib import Path

import pytest


# ── Deep Reflexion Tests ──────────────────────────────────────────


class TestToolLoopDetector:
    def test_no_loop(self):
        from agent.deep_reflexion import ToolLoopDetector

        detector = ToolLoopDetector(window=5, threshold=3)
        assert detector.check("bash", {"command": "ls"}) is False
        assert detector.check("bash", {"command": "pwd"}) is False
        assert detector.check("read_file", {"path": "test.py"}) is False

    def test_loop_detected(self):
        from agent.deep_reflexion import ToolLoopDetector

        detector = ToolLoopDetector(window=5, threshold=3)
        params = {"command": "ls -la"}
        assert detector.check("bash", params) is False
        assert detector.check("bash", params) is False
        assert detector.check("bash", params) is True  # 3rd consecutive same call

    def test_different_params_no_loop(self):
        from agent.deep_reflexion import ToolLoopDetector

        detector = ToolLoopDetector(window=5, threshold=3)
        assert detector.check("bash", {"command": "ls"}) is False
        assert detector.check("bash", {"command": "pwd"}) is False
        assert detector.check("bash", {"command": "ls"}) is False  # Different from previous

    def test_reset(self):
        from agent.deep_reflexion import ToolLoopDetector

        detector = ToolLoopDetector(window=5, threshold=3)
        params = {"command": "ls"}
        detector.check("bash", params)
        detector.check("bash", params)
        detector.reset()
        assert detector.check("bash", params) is False  # Reset cleared history


class TestErrorRepeatDetector:
    def test_no_repeat(self):
        from agent.deep_reflexion import ErrorRepeatDetector

        detector = ErrorRepeatDetector()
        assert detector.check("Error: file not found") is False
        assert detector.check("Error: permission denied") is False

    def test_repeat_detected(self):
        from agent.deep_reflexion import ErrorRepeatDetector

        detector = ErrorRepeatDetector()
        assert detector.check("Error: file not found") is False
        assert detector.check("Error: file not found") is True  # Same error again

    def test_different_errors(self):
        from agent.deep_reflexion import ErrorRepeatDetector

        detector = ErrorRepeatDetector()
        assert detector.check("Error: A") is False
        assert detector.check("Error: B") is False
        assert detector.check("Error: A") is True  # A repeated


class TestNoProgressDetector:
    def test_tool_overuse(self):
        from agent.deep_reflexion import NoProgressDetector

        detector = NoProgressDetector(tool_overuse_threshold=4, window=10)
        for _ in range(5):
            detector.record_tool_call("bash")
        warning = detector.check()
        assert warning is not None
        assert "bash" in warning

    def test_result_similarity(self):
        from agent.deep_reflexion import NoProgressDetector

        detector = NoProgressDetector()
        detector.record_result("same output")
        detector.record_result("same output")
        detector.record_result("same output")
        warning = detector.check()
        assert warning is not None
        assert "identical" in warning

    def test_no_progress_ok(self):
        from agent.deep_reflexion import NoProgressDetector

        detector = NoProgressDetector()
        detector.record_tool_call("bash")
        detector.record_tool_call("read_file")
        detector.record_result("output A")
        detector.record_result("output B")
        assert detector.check() is None


class TestDiscoveryExtractor:
    def test_extract_read_files(self):
        from agent.deep_reflexion import DiscoveryExtractor

        history = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "test.py"}'}}
            ]},
            {"role": "tool", "content": "file content"},
        ]
        discoveries = DiscoveryExtractor.extract(history)
        assert "read_file: test.py" in discoveries

    def test_extract_empty(self):
        from agent.deep_reflexion import DiscoveryExtractor

        assert DiscoveryExtractor.extract([]) == ""
        assert DiscoveryExtractor.extract([{"role": "user", "content": "hi"}]) == ""


class TestDeepReflexion:
    def test_on_tool_call_loop(self):
        from agent.deep_reflexion import DeepReflexion

        reflexion = DeepReflexion(loop_threshold=2)
        params = {"command": "ls"}
        assert reflexion.on_tool_call("bash", params) is None
        warning = reflexion.on_tool_call("bash", params)
        assert warning is not None
        assert "LOOP" in warning

    def test_on_tool_result_error_repeat(self):
        from agent.deep_reflexion import DeepReflexion

        reflexion = DeepReflexion()
        assert reflexion.on_tool_result("Error: not found") is None
        warning = reflexion.on_tool_result("Error: not found")
        assert warning is not None
        assert "ERROR REPEAT" in warning

    def test_should_retry(self):
        from agent.deep_reflexion import DeepReflexion

        reflexion = DeepReflexion(max_retries=3)
        assert reflexion.should_retry() is True
        reflexion.attempt_count = 3
        assert reflexion.should_retry() is False

    def test_get_retry_context(self):
        from agent.deep_reflexion import DeepReflexion

        reflexion = DeepReflexion()
        assert reflexion.get_retry_context() == ""
        reflexion.reflections = ["Reflection 1"]
        reflexion.discoveries = "- read_file: test.py"
        context = reflexion.get_retry_context()
        assert "Previous Attempt Context" in context
        assert "Reflection 1" in context
        assert "test.py" in context

    def test_get_stats(self):
        from agent.deep_reflexion import DeepReflexion

        reflexion = DeepReflexion()
        stats = reflexion.get_stats()
        assert stats["attempts"] == 0
        assert stats["reflections"] == 0


# ── Guardian Tests ────────────────────────────────────────────────


class TestPatternScanner:
    def test_safe_command(self):
        from agent.guardian import PatternScanner

        scanner = PatternScanner()
        risk, findings = scanner.scan_bash("ls -la /tmp")
        assert risk == "low"
        assert len(findings) == 0

    def test_rm_root(self):
        from agent.guardian import PatternScanner

        scanner = PatternScanner()
        risk, findings = scanner.scan_bash("rm -rf /")
        assert risk == "critical"
        assert len(findings) > 0

    def test_sudo(self):
        from agent.guardian import PatternScanner

        scanner = PatternScanner()
        risk, findings = scanner.scan_bash("sudo apt install something")
        assert risk == "high"

    def test_write_sensitive_path(self):
        from agent.guardian import PatternScanner

        scanner = PatternScanner()
        risk, findings = scanner.scan_write_operation(
            "write_file", {"path": "/etc/passwd", "content": "test"}
        )
        assert risk == "high"
        assert len(findings) > 0


class TestGuardianAgent:
    @pytest.mark.asyncio
    async def test_review_safe(self):
        from agent.guardian import GuardianAgent

        guardian = GuardianAgent()
        result = await guardian.review("read_file", {"path": "test.py"})
        assert result.passed is True
        assert result.risk_level == "low"

    @pytest.mark.asyncio
    async def test_review_dangerous(self):
        from agent.guardian import GuardianAgent

        guardian = GuardianAgent(auto_block_critical=True)
        result = await guardian.review("bash", {"command": "rm -rf /"})
        assert result.passed is False
        assert result.risk_level == "critical"
        assert len(result.findings) > 0

    @pytest.mark.asyncio
    async def test_review_sudo(self):
        from agent.guardian import GuardianAgent

        guardian = GuardianAgent()
        result = await guardian.review("bash", {"command": "sudo apt install x"})
        assert result.passed is True  # High but not critical
        assert result.risk_level == "high"

    def test_session_summary(self):
        from agent.guardian import GuardianAgent

        guardian = GuardianAgent()
        # Manually add some operations
        guardian._session_operations = [
            {"tool": "bash", "risk_level": "low", "passed": True},
            {"tool": "bash", "risk_level": "high", "passed": True},
            {"tool": "bash", "risk_level": "critical", "passed": False},
        ]
        summary = guardian.get_session_summary()
        assert summary["total_operations"] == 3
        assert summary["blocked_count"] == 1


# ── Sentinel Tests ───────────────────────────────────────────────


class TestSentinelAgent:
    def test_should_activate_keywords(self):
        from agent.sentinel import SentinelAgent

        sentinel = SentinelAgent(work_dir=Path("/tmp"))
        should, reason = sentinel.should_activate("Do A then B then C")
        assert should is True
        assert "Multi-step" in reason

    def test_should_activate_reflexion_failures(self):
        from agent.sentinel import SentinelAgent

        sentinel = SentinelAgent(work_dir=Path("/tmp"), activation_threshold=2)
        should, reason = sentinel.should_activate("Simple task", reflexion_failure_count=3)
        assert should is True
        assert "failures" in reason

    def test_should_not_activate_simple(self):
        from agent.sentinel import SentinelAgent

        sentinel = SentinelAgent(work_dir=Path("/tmp"))
        should, reason = sentinel.should_activate("Read this file")
        assert should is False

    def test_heuristic_decompose(self):
        from agent.sentinel import SentinelAgent

        sentinel = SentinelAgent(work_dir=Path("/tmp"))
        milestones = sentinel._heuristic_decompose("1. First step 2. Second step 3. Third step")
        assert len(milestones) >= 1

    def test_progress_summary(self):
        from agent.sentinel import SentinelAgent, TaskPlan, Milestone

        sentinel = SentinelAgent(work_dir=Path("/tmp"))
        plan = TaskPlan(
            task_hash="test",
            task_description="Test task",
            milestones=[
                Milestone(index=0, description="Step 1", status="completed"),
                Milestone(index=1, description="Step 2", status="in_progress"),
                Milestone(index=2, description="Step 3", status="pending"),
            ],
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        sentinel._active_plan = plan
        summary = sentinel.get_progress_summary()
        assert "✅" in summary
        assert "🔄" in summary
        assert "⬜" in summary

    def test_mark_completed(self):
        from agent.sentinel import SentinelAgent, TaskPlan, Milestone

        with tempfile.TemporaryDirectory() as tmp:
            sentinel = SentinelAgent(work_dir=Path(tmp))
            plan = TaskPlan(
                task_hash="test",
                task_description="Test",
                milestones=[
                    Milestone(index=0, description="Step 1"),
                    Milestone(index=1, description="Step 2"),
                ],
                created_at="2026-01-01",
                updated_at="2026-01-01",
            )
            sentinel._active_plan = plan
            sentinel.mark_milestone_completed(0, "Done")
            assert plan.milestones[0].status == "completed"
            sentinel.mark_milestone_completed(1, "Done")
            assert plan.status == "completed"


# ── Context Evolution Tests ──────────────────────────────────────


class TestLessonRecorder:
    def test_record_new(self):
        from agent.context_evolution import LessonRecorder

        with tempfile.TemporaryDirectory() as tmp:
            recorder = LessonRecorder(Path(tmp))
            lesson = recorder.record("failure", "approach", "Test lesson", "Test context")
            assert lesson is not None
            assert lesson.evidence == 1

    def test_record_duplicate_increments(self):
        from agent.context_evolution import LessonRecorder

        with tempfile.TemporaryDirectory() as tmp:
            recorder = LessonRecorder(Path(tmp))
            recorder.record("failure", "approach", "Test lesson", "Context 1")
            lesson2 = recorder.record("failure", "approach", "Test lesson", "Context 2")
            assert lesson2.evidence == 2

    def test_get_unpromoted(self):
        from agent.context_evolution import LessonRecorder

        with tempfile.TemporaryDirectory() as tmp:
            recorder = LessonRecorder(Path(tmp))
            recorder.record("failure", "approach", "Lesson A", "ctx")
            recorder.record("failure", "approach", "Lesson A", "ctx")  # evidence 2
            recorder.record("failure", "approach", "Lesson B", "ctx")  # evidence 1
            unpromoted = recorder.get_unpromoted(min_evidence=2)
            assert len(unpromoted) == 1
            assert unpromoted[0].lesson == "Lesson A"


class TestLessonExtractor:
    def test_from_failure_timeout(self):
        from agent.context_evolution import LessonExtractor

        lesson = LessonExtractor.from_failure("task", "Connection timed out", [])
        assert lesson is not None
        assert lesson["category"] == "approach"

    def test_from_failure_permission(self):
        from agent.context_evolution import LessonExtractor

        lesson = LessonExtractor.from_failure("task", "Permission denied", [])
        assert lesson is not None
        assert "permission" in lesson["lesson"].lower()

    def test_from_user_feedback_constraint(self):
        from agent.context_evolution import LessonExtractor

        lesson = LessonExtractor.from_user_feedback("Don't use that method anymore")
        assert lesson is not None
        assert lesson["category"] == "constraint"

    def test_from_user_feedback_preference(self):
        from agent.context_evolution import LessonExtractor

        lesson = LessonExtractor.from_user_feedback("I prefer using pytest over unittest")
        assert lesson is not None
        assert lesson["category"] == "preference"


class TestContextEvolutionManager:
    def test_on_failure(self):
        from agent.context_evolution import ContextEvolutionManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextEvolutionManager(Path(tmp))
            manager.on_failure("task", "Connection timed out")
            stats = manager.get_stats()
            assert stats["total_lessons"] == 1

    def test_on_user_feedback(self):
        from agent.context_evolution import ContextEvolutionManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextEvolutionManager(Path(tmp))
            manager.on_user_feedback("Don't use that approach")
            stats = manager.get_stats()
            assert stats["total_lessons"] == 1
            assert stats["by_source"]["user_feedback"] == 1

    def test_auto_promote(self):
        from agent.context_evolution import ContextEvolutionManager

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            # Create AGENTS.md
            (work_dir / "AGENTS.md").write_text("# AGENTS.md\n\nExisting content\n")

            manager = ContextEvolutionManager(work_dir, promotion_threshold=1, auto_promote=True)
            manager.on_failure("task", "timeout error")
            promoted = manager.auto_promote_lessons()
            assert promoted == 1

            # Verify AGENTS.md was updated
            content = (work_dir / "AGENTS.md").read_text()
            assert "Learned Rules" in content
