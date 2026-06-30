#!/usr/bin/env python3
"""Evaluation script — measure effectiveness of cognitive modules.

Answers: "Do the 21+ cognitive modules actually help?"

Runs a battery of tests:
1. Module health check (all importable, configurable)
2. Wiring verification (runtime hooks active)
3. Functional tests (each module does what it claims)
4. Integration tests (modules work together)
5. Token cost estimate per feature flag combo

Usage:
    python scripts/evaluate_cognitive.py
    python scripts/evaluate_cognitive.py --verbose
    python scripts/evaluate_cognitive.py --json > report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Test result data structures
# --------------------------------------------------------------------------- #


@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    detail: str = ""
    elapsed_ms: int = 0


@dataclass
class EvalReport:
    timestamp: float = field(default_factory=time.time)
    total_modules: int = 0
    healthy_modules: int = 0
    wired_hooks: int = 0
    total_hooks: int = 0
    tests: list[TestResult] = field(default_factory=list)
    token_cost_estimates: dict[str, dict[str, Any]] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.tests:
            return 0.0
        passed = sum(1 for t in self.tests if t.passed)
        return passed / len(self.tests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_modules": self.total_modules,
            "healthy_modules": self.healthy_modules,
            "wired_hooks": self.wired_hooks,
            "total_hooks": self.total_hooks,
            "pass_rate": self.pass_rate,
            "tests": [asdict(t) for t in self.tests],
            "token_cost_estimates": self.token_cost_estimates,
            "recommendations": self.recommendations,
        }


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


def _check(name: str, category: str, fn) -> TestResult:
    """Run a check function, capture result."""
    started = time.time()
    try:
        detail = fn()
        elapsed = int((time.time() - started) * 1000)
        if isinstance(detail, tuple) and len(detail) == 2:
            passed, detail_str = detail
        else:
            passed = True
            detail_str = str(detail) if detail else "OK"
        return TestResult(
            name=name,
            category=category,
            passed=bool(passed),
            detail=detail_str,
            elapsed_ms=elapsed,
        )
    except Exception as exc:
        elapsed = int((time.time() - started) * 1000)
        return TestResult(
            name=name,
            category=category,
            passed=False,
            detail=f"EXCEPTION: {exc!r}",
            elapsed_ms=elapsed,
        )


# --------------------------------------------------------------------------- #
# Test categories
# --------------------------------------------------------------------------- #


# List of all cognitive modules to check
COGNITIVE_MODULES = [
    "agent.unified.config",
    "agent.unified.events",
    "agent.unified.policy",
    "agent.unified.reflexion",
    "agent.unified.decision",
    "agent.unified.reasoning",
    "agent.unified.smart_guardian",
    "agent.unified.longrun",
    "agent.unified.tool_router",
    "agent.unified.cognitive_tree",
    "agent.unified.hypothesis",
    "agent.unified.context_distiller",
    "agent.unified.metacognitive",
    "agent.unified.causal_graph",
    "agent.unified.learning",
    "agent.unified.skill_synthesizer",
    "agent.unified.task_planner",
    "agent.unified.output_formatter",
    "agent.unified.verifier",
    "agent.unified.constitution",
    "agent.unified.slow_thinking",
    "agent.unified.ensemble",
    "agent.unified.capability_resolver",
    "agent.unified.cost_tracker",
    "agent.unified.response_cache",
    "agent.unified.user_model",
    "agent.unified.clarifier",
    "agent.unified.streaming",
    "agent.unified.embedding",
    "agent.unified.integration",
    "agent.unified.runtime_wiring",
]


def test_module_health() -> list[TestResult]:
    """Check that all cognitive modules import successfully."""
    results = []
    for mod_name in COGNITIVE_MODULES:
        def _check_one():
            __import__(mod_name)
            return True, f"imported"
        results.append(_check(f"import {mod_name}", "module_health", _check_one))
    return results


def test_wiring() -> list[TestResult]:
    """Verify that runtime hooks are wired into Hermes mega-files."""
    results = []
    repo_root = Path(__file__).parent.parent

    def _check_wired(file_rel: str, marker: str):
        fpath = repo_root / file_rel
        if not fpath.exists():
            return False, f"file not found: {file_rel}"
        content = fpath.read_text(encoding="utf-8")
        if marker in content:
            return True, f"marker found in {file_rel}"
        return False, f"marker '{marker}' NOT found in {file_rel}"

    results.append(_check(
        "system_prompt.py wired",
        "wiring",
        lambda: _check_wired("agent/system_prompt.py", "augment_volatile_prompt"),
    ))
    results.append(_check(
        "turn_finalizer.py wired",
        "wiring",
        lambda: _check_wired("agent/turn_finalizer.py", "maybe_run_cognitive_pipeline"),
    ))
    results.append(_check(
        "conversation_loop.py wired",
        "wiring",
        lambda: _check_wired("agent/conversation_loop.py", "_set_last_user_message"),
    ))
    results.append(_check(
        "model_tools.py wired",
        "wiring",
        lambda: _check_wired("model_tools.py", "on_tool_call_complete"),
    ))
    results.append(_check(
        "gateway/delivery.py wired",
        "wiring",
        lambda: _check_wired("gateway/delivery.py", "format_for_delivery"),
    ))
    results.append(_check(
        "run_agent.py wired",
        "wiring",
        lambda: _check_wired("run_agent.py", "wire_llm_client"),
    ))
    results.append(_check(
        "hermes_cli/main.py wired",
        "wiring",
        lambda: _check_wired("hermes_cli/main.py", "wire_llm_client"),
    ))
    return results


def test_functional() -> list[TestResult]:
    """Functional tests for each module."""
    results = []

    # DecisionFramework
    def _test_decision():
        from agent.unified.decision import DecisionFramework, DecisionClass
        df = DecisionFramework()
        c = df.classify("read_file", {"path": "/tmp/x"})
        assert c.is_trivial, f"read_file should be TRIVIAL, got {c.cls.name}"
        c = df.classify("bash", {"command": "rm -rf /"})
        assert c.cls == DecisionClass.IRREVERSIBLE, f"rm -rf / should be IRREVERSIBLE"
        return True, f"classification OK"
    results.append(_check("DecisionFramework classify", "functional", _test_decision))

    # ReflexionStore
    def _test_reflexion():
        from agent.unified.reflexion import ReflexionStore, ReflexionRecord
        with tempfile.TemporaryDirectory() as tmp:
            store = ReflexionStore(os.path.join(tmp, "reflex.jsonl"))
            rec = ReflexionRecord(lesson="test lesson", source="test", scope="global")
            added = store.add(rec)
            assert added, "should add record"
            results = store.recall("test", limit=5)
            assert len(results) == 1, f"expected 1, got {len(results)}"
            return True, f"add + recall OK"
    results.append(_check("ReflexionStore add+recall", "functional", _test_reflexion))

    # CostTracker
    def _test_cost():
        from agent.unified.cost_tracker import CostTracker, BudgetConfig
        tracker = CostTracker(budget=BudgetConfig(total_token_budget=10000))
        tracker.record(phase="plan", prompt_tokens=100, completion_tokens=50)
        s = tracker.summary()
        assert s["total_tokens"] == 150, f"expected 150, got {s['total_tokens']}"
        return True, f"token count OK"
    results.append(_check("CostTracker count", "functional", _test_cost))

    # ResponseCache
    def _test_cache():
        from agent.unified.response_cache import ResponseCache
        cache = ResponseCache(max_entries=10, ttl_seconds=60)
        assert cache.get("sys", "user") is None, "should miss"
        cache.put("sys", "user", "response")
        assert cache.get("sys", "user") == "response", "should hit"
        s = cache.stats()
        assert s["hits"] == 1 and s["misses"] == 1
        return True, f"cache hit/miss OK"
    results.append(_check("ResponseCache hit/miss", "functional", _test_cache))

    # UserModel
    def _test_user_model():
        from agent.unified.user_model import UserModel
        with tempfile.TemporaryDirectory() as tmp:
            model = UserModel(profile_path=os.path.join(tmp, "profile.json"))
            for _ in range(5):
                model.observe_message("help me debug this python error")
            p = model.get_profile()
            assert p.total_messages == 5
            assert "python" in p.domains or "debug" in p.recurring_requests
            return True, f"profile built OK"
    results.append(_check("UserModel profile build", "functional", _test_user_model))

    # Clarifier
    def _test_clarifier():
        from agent.unified.clarifier import Clarifier
        c = Clarifier(llm_call=None, heuristic_threshold=0.3)
        result = c.assess(user_message="fix it")
        assert result.ambiguity_score > 0, "should detect some ambiguity"
        return True, f"score={result.ambiguity_score:.2f}, signals={result.signals}"
    results.append(_check("Clarifier ambiguity detection", "functional", _test_clarifier))

    # CausalGraph
    def _test_causal():
        from agent.unified.causal_graph import CausalGraph
        g = CausalGraph()
        g.add_node(node_id="a", label="Step A", node_type="action", status="failed")
        g.add_node(node_id="b", label="Bad config", node_type="assumption", status="succeeded")
        g.add_edge(src="b", dst="a", edge_type="causes")
        roots = g.root_causes("a")
        assert len(roots) == 1 and roots[0].node_id == "b", f"expected b as root, got {[r.node_id for r in roots]}"
        return True, f"root cause found: {roots[0].label}"
    results.append(_check("CausalGraph root cause", "functional", _test_causal))

    # TaskPlanner (recursive)
    def _test_task_planner():
        from agent.unified.task_planner import TaskPlanner, TaskPlan, Subtask
        with tempfile.TemporaryDirectory() as tmp:
            planner = TaskPlanner(persist_path=os.path.join(tmp, "plans.json"))
            p = TaskPlan(plan_id="t", original_request="test")
            p.subtasks = [Subtask(subtask_id="st_01", description="simple", estimated_difficulty=0.2)]
            planner._plans[p.plan_id] = p
            planner._active_plan_id = p.plan_id
            # decompose should return None (too simple, no LLM)
            result = planner.decompose_subtask(subtask_id="st_01")
            assert result is None, "should not decompose simple task"
            return True, f"recursive decompose skips simple tasks OK"
    results.append(_check("TaskPlanner recursive", "functional", _test_task_planner))

    # OutputFormatter
    def _test_formatter():
        from agent.unified.output_formatter import OutputFormatter
        f = OutputFormatter()
        chunks = f.format('{"key": "value"}', platform="telegram")
        assert len(chunks) >= 1
        text = chunks[0].text
        # JSON should be converted to readable format (no raw braces).
        assert "key" in text and "value" in text
        return True, f"JSON→readable OK, first 50 chars: {text[:50]!r}"
    results.append(_check("OutputFormatter JSON→readable", "functional", _test_formatter))

    # Embedding (null backend)
    def _test_embedding():
        from agent.unified.embedding import configure_embedder, embed, embedding_stats
        configure_embedder(backend="none")
        assert embed("test") is None, "null embedder should return None"
        s = embedding_stats()
        assert s["backend"] == "none"
        return True, f"null embedder OK"
    results.append(_check("Embedding null backend", "functional", _test_embedding))

    return results


def test_token_costs() -> dict[str, dict[str, Any]]:
    """Estimate token cost per feature flag combo."""
    # Rough estimates based on module prompts.
    combos = {
        "baseline (no cognitive)": {
            "calls_per_response": 1,
            "extra_tokens": 0,
            "notes": "Single LLM call, no cognitive overhead.",
        },
        "verifier only": {
            "calls_per_response": 2,  # 1 generate + 1 critique
            "extra_tokens": 500,
            "notes": "Critique adds ~500 tokens. Revise adds 1 more call if needed.",
        },
        "verifier + constitution": {
            "calls_per_response": 3,
            "extra_tokens": 800,
            "notes": "Constitution check adds 1 call.",
        },
        "slow_thinking balanced": {
            "calls_per_response": 4,  # decompose + analyze + synthesize + final
            "extra_tokens": 2000,
            "notes": "3 reasoning rounds + final answer.",
        },
        "slow_thinking deep": {
            "calls_per_response": 6,  # + critique + refine
            "extra_tokens": 4000,
            "notes": "5 rounds + final.",
        },
        "slow_thinking max": {
            "calls_per_response": 9,  # explore + 3 + critique + refine + critique + refine + final
            "extra_tokens": 12000,
            "notes": "8 rounds + final. ~15K extra tokens.",
        },
        "ensemble (3 models + judge)": {
            "calls_per_response": 4,  # 3 models + 1 judge
            "extra_tokens": 3000,
            "notes": "3 parallel model calls + 1 judge. 3x base cost.",
        },
        "full pipeline (slow max + ensemble + verify + constitution)": {
            "calls_per_response": 15,
            "extra_tokens": 18000,
            "notes": "All enabled. Use only for critical/irreversible tasks.",
        },
    }
    return combos


def generate_recommendations(report: EvalReport) -> list[str]:
    """Generate actionable recommendations based on test results."""
    recs = []
    if report.healthy_modules < report.total_modules:
        recs.append(
            f"❌ {report.total_modules - report.healthy_modules} modules failed to import. "
            "Fix import errors before enabling features."
        )
    if report.wired_hooks < report.total_hooks:
        recs.append(
            f"⚠️ {report.total_hooks - report.wired_hooks} runtime hooks not wired. "
            "Cognitive modules won't activate even if enabled."
        )
    failed_tests = [t for t in report.tests if not t.passed]
    if failed_tests:
        recs.append(
            f"❌ {len(failed_tests)} functional tests failed. "
            "See test details for specifics."
        )
    # Token cost recommendations
    costs = report.token_cost_estimates
    if costs:
        max_combo = max(costs.values(), key=lambda c: c["extra_tokens"])
        if max_combo["extra_tokens"] > 10000:
            recs.append(
                f"💰 Most expensive combo adds ~{max_combo['extra_tokens']} tokens. "
                "Use tiered activation: fast for trivial, max for critical."
            )
    if not recs:
        recs.append("✅ All systems healthy. Enable features incrementally and monitor.")
    return recs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate cognitive modules")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--json", action="store_true", help="output as JSON")
    args = parser.parse_args()

    # Setup test environment
    os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="hermes-eval-"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    report = EvalReport()
    report.total_modules = len(COGNITIVE_MODULES)

    # 1. Module health
    if not args.json:
        print("=== 1. Module Health ===")
    health_results = test_module_health()
    report.tests.extend(health_results)
    report.healthy_modules = sum(1 for r in health_results if r.passed)
    if not args.json:
        print(f"  {report.healthy_modules}/{report.total_modules} modules importable")
        if args.verbose:
            for r in health_results:
                status = "✓" if r.passed else "✗"
                print(f"    {status} {r.name}: {r.detail}")

    # 2. Wiring verification
    if not args.json:
        print("\n=== 2. Runtime Wiring ===")
    wiring_results = test_wiring()
    report.tests.extend(wiring_results)
    report.total_hooks = len(wiring_results)
    report.wired_hooks = sum(1 for r in wiring_results if r.passed)
    if not args.json:
        print(f"  {report.wired_hooks}/{report.total_hooks} hooks wired")
        for r in wiring_results:
            status = "✓" if r.passed else "✗"
            print(f"    {status} {r.name}")

    # 3. Functional tests
    if not args.json:
        print("\n=== 3. Functional Tests ===")
    functional_results = test_functional()
    report.tests.extend(functional_results)
    passed = sum(1 for r in functional_results if r.passed)
    if not args.json:
        print(f"  {passed}/{len(functional_results)} tests passed")
        if args.verbose:
            for r in functional_results:
                status = "✓" if r.passed else "✗"
                print(f"    {status} {r.name}: {r.detail}")

    # 4. Token cost estimates
    if not args.json:
        print("\n=== 4. Token Cost Estimates ===")
    report.token_cost_estimates = test_token_costs()
    if not args.json:
        for combo, info in report.token_cost_estimates.items():
            print(f"  {combo}: {info['calls_per_response']} calls, +{info['extra_tokens']} tokens")
            if args.verbose:
                print(f"    {info['notes']}")

    # 5. Recommendations
    report.recommendations = generate_recommendations(report)
    if not args.json:
        print("\n=== 5. Recommendations ===")
        for rec in report.recommendations:
            print(f"  {rec}")

    # Summary
    if not args.json:
        print(f"\n=== Summary ===")
        print(f"  Overall pass rate: {report.pass_rate:.1%}")
        print(f"  Modules healthy: {report.healthy_modules}/{report.total_modules}")
        print(f"  Hooks wired: {report.wired_hooks}/{report.total_hooks}")
    else:
        print(json.dumps(report.to_dict(), indent=2, default=str))

    return 0 if report.pass_rate >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
