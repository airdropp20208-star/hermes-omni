"""Configuration helpers for Hermes unified integration.

Reads Hermes config.yaml when available and keeps environment variables as
high-priority overrides for deployment/testing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


def _cfg_get(*path: str, default: Any = None) -> Any:
    try:
        from hermes_cli.config import cfg_get, load_config

        return cfg_get(load_config(), *path, default=default)
    except Exception:
        return default


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _falsey_env(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disabled"}


@dataclass(frozen=True)
class UnifiedConfig:
    enabled: bool = True
    reflexion_enabled: bool = True
    guardian_enabled: bool = True
    auto_prefetch_enabled: bool = True
    store_path: Path = field(default_factory=lambda: get_hermes_home() / "unified" / "reflexions.jsonl")
    block_tools: tuple[str, ...] = ()
    max_records: int = 2000
    scope_by_cwd: bool = True
    # --- Reasoning-first protocol (v1) ---
    # When True, the agent generates structured plans before any
    # CONSEQUENTIAL+ action, and reflects after. Off by default to
    # preserve legacy behavior; enable explicitly to opt into the
    # reasoning-first mode.
    reasoning_enabled: bool = False
    # When True, the Smart Guardian (LLM-as-judge) runs for
    # CONSEQUENTIAL+ actions that pass the pattern layer. Off by default.
    smart_guardian_enabled: bool = False
    # When True, IRREVERSIBLE actions are auto-blocked at the framework
    # layer even if the LLM judge says ALLOW. This is the hard floor.
    # Set to False to trust the LLM judge entirely (Codex-style).
    hard_block_irreversible: bool = True
    # When True, the protocol persists reasoning reflections into the
    # reflexion store, so lessons compound across sessions.
    persist_reflections: bool = True
    # Cache size & TTL for the Smart Guardian verdict cache.
    guardian_cache_size: int = 512
    guardian_cache_ttl_seconds: int = 3600
    # --- Long-run engine (v1.1) ---
    # When True, the engine runs a background work queue + checkpoint store
    # + reflection worker. Reflections move off the synchronous path.
    longrun_enabled: bool = False
    # Heartbeat interval for the long-run worker (seconds).
    longrun_heartbeat_seconds: float = 30.0
    # Debounce: how long to wait after the last reflection request before
    # processing a batch (seconds).
    longrun_reflection_debounce_seconds: float = 5.0
    # Max items per reflection batch (single LLM call).
    longrun_reflection_batch_size: int = 10
    # --- Tool router (v1.1) ---
    # When True, the router injects "Relevant tools for this task:" into
    # the system prompt before each turn. Helps mid-tier LLMs pick the
    # right tool.
    tool_router_enabled: bool = False
    # Number of tools to suggest in the system prompt (0 = no limit).
    tool_router_top_n: int = 5
    # When True, the router learns from usage (records which tools are
    # actually called for which queries) and boosts future suggestions.
    tool_router_learn: bool = True
    # --- v2 cognitive extensions ---
    # CognitiveTree: branching reasoning with pruning. Only for
    # CONSEQUENTIAL+ actions. 2 LLM calls per evaluation.
    cognitive_tree_enabled: bool = False
    cognitive_tree_n_branches: int = 3
    cognitive_tree_min_confidence: float = 0.6
    cognitive_tree_max_confidence: float = 0.85
    # HypothesisEngine: hypothesis-test-revise for diagnostic tasks.
    # Triggered explicitly via tools or by "why/debug" keywords.
    hypothesis_enabled: bool = False
    hypothesis_n_hypotheses: int = 3
    hypothesis_max_iterations: int = 5
    hypothesis_confidence_threshold: float = 0.8
    # ContextDistiller: extract structured insights from conversation.
    # Runs every N turns.
    context_distiller_enabled: bool = False
    context_distill_every_n_turns: int = 10
    context_distiller_max_items: int = 30
    context_distiller_merge_threshold: int = 50
    # MetacognitiveMonitor: self-doubt and calibration.
    metacognitive_enabled: bool = False
    metacognitive_self_doubt_threshold: float = 0.5
    metacognitive_repeated_failure_count: int = 3
    metacognitive_min_samples: int = 5
    # CausalGraph: cause-effect model per task. Agent uses explicitly.
    causal_graph_enabled: bool = False
    # --- v2.1 learning + memory + skill synthesis ---
    # LearningEngine: extract learnings from EVERY interaction (success,
    # correction, pattern, fact, preference, timing). Separate from
    # reflexion (which is failure-only).
    learning_enabled: bool = False
    learning_max_records: int = 5000
    learning_extract_every_n_turns: int = 8
    # SkillSynthesizer: detect repeated tool-call patterns and auto-
    # generate reusable skills in ~/.hermes/skills/auto-synthesized/.
    skill_synthesis_enabled: bool = False
    skill_synthesis_min_occurrences: int = 3
    skill_synthesis_max_skills: int = 100
    # --- v2.2 task planning ---
    # TaskPlanner: decompose complex requests into subtasks, track
    # progress, replan on failure. Persisted to
    # ~/.hermes/unified/task_plans.json.
    task_planner_enabled: bool = False
    task_planner_max_subtasks: int = 15
    task_planner_max_replans: int = 3
    # --- v2.3 output formatting ---
    # OutputFormatter: transform agent output for messaging platforms
    # (Telegram MarkdownV2 escaping, JSON pretty-print, chunking, table
    # conversion, control char stripping).
    output_formatter_enabled: bool = False
    output_formatter_summarize_long: bool = False
    output_formatter_summarize_threshold: int = 8000
    # --- v3 cognitive extensions (phases 1-5) ---
    # Verifier: self-verification loop (critique → revise → re-critique).
    # Catches hallucinations, logical errors, factual mistakes.
    verifier_enabled: bool = False
    verifier_max_iterations: int = 3
    verifier_pass_threshold: float = 0.7
    # Constitution: value alignment layer. User defines principles in
    # ~/.hermes/constitution.md. Verifier checks alignment too.
    constitution_enabled: bool = False
    # SlowThinking: multi-level deep reasoning (fast/balanced/deep/max).
    # The biggest differentiator for math/code/analysis quality.
    slow_thinking_enabled: bool = False
    slow_thinking_default_level: str = "fast"  # fast/balanced/deep/max
    slow_thinking_store_traces: bool = False
    # Ensemble: multi-model reasoning + LLM judge. Runs N models in
    # parallel for hard tasks, judge picks/synthesizes.
    ensemble_enabled: bool = False
    ensemble_max_workers: int = 5
    ensemble_timeout_seconds: float = 60.0
    # CapabilityResolver: auto-install/create missing tools+MCPs.
    # Agent never gets stuck on "missing tool" — installs or generates.
    capability_resolver_enabled: bool = False
    capability_resolver_auto_approve: bool = False  # DANGER: if true, no user prompt
    capability_resolver_allow_network: bool = True
    # --- v3.1 Tier 1-2 upgrades ---
    # CostTracker: token accounting + budget caps per phase/session.
    cost_tracker_enabled: bool = False
    cost_tracker_total_budget: int = 1_000_000
    cost_tracker_hard_stop: bool = True
    # ResponseCache: content-hash cache for LLM calls (saves 30-50% for
    # repeated prompts in ensemble/slow_thinking).
    response_cache_enabled: bool = False
    response_cache_max_entries: int = 512
    response_cache_ttl_seconds: int = 3600
    # UserModel: build user profile (expertise, style, domains) for
    # personalization. Injected into system prompt.
    user_model_enabled: bool = False
    # Clarifier: detect ambiguity + ask clarifying questions before acting.
    clarifier_enabled: bool = False
    clarifier_heuristic_threshold: float = 0.5
    clarifier_always_use_llm: bool = False
    # Streaming: stream LLM tokens instead of batch (UX: user sees progress).
    streaming_enabled: bool = False
    # Embedding: semantic recall (replace BM25). +40% recall quality.
    embedding_enabled: bool = False
    embedding_backend: str = "local"  # "local" | "openai" | "none"
    embedding_model: str = "all-MiniLM-L6-v2"


_CONFIG_CACHE: UnifiedConfig | None = None


def load_unified_config(*, refresh: bool = False) -> UnifiedConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not refresh:
        return _CONFIG_CACHE

    enabled = _truthy(_cfg_get("unified", "enabled", default=True), default=True)
    env_enabled = _falsey_env("HERMES_UNIFIED_ENABLED")
    if env_enabled is not None:
        enabled = env_enabled

    reflexion_enabled = _truthy(_cfg_get("unified", "reflexion", "enabled", default=True), default=True)
    guardian_enabled = _truthy(_cfg_get("unified", "guardian", "enabled", default=True), default=True)
    auto_prefetch_enabled = _truthy(_cfg_get("unified", "reflexion", "auto_prefetch", default=True), default=True)

    store_raw = os.getenv("HERMES_UNIFIED_REFLEXION_STORE") or _cfg_get(
        "unified", "reflexion", "store", default=""
    )
    store_path = Path(str(store_raw)).expanduser() if store_raw else get_hermes_home() / "unified" / "reflexions.jsonl"

    raw_block_tools = os.getenv("HERMES_UNIFIED_BLOCK_TOOLS")
    if raw_block_tools is None:
        raw_cfg = _cfg_get("unified", "guardian", "block_tools", default=[])
        if isinstance(raw_cfg, str):
            block_tools = tuple(part.strip() for part in raw_cfg.split(",") if part.strip())
        elif isinstance(raw_cfg, list):
            block_tools = tuple(str(part).strip() for part in raw_cfg if str(part).strip())
        else:
            block_tools = ()
    else:
        block_tools = tuple(part.strip() for part in raw_block_tools.split(",") if part.strip())

    try:
        max_records = int(_cfg_get("unified", "reflexion", "max_records", default=2000) or 2000)
    except Exception:
        max_records = 2000

    scope_by_cwd = _truthy(_cfg_get("unified", "reflexion", "scope_by_cwd", default=True), default=True)

    # --- Reasoning-first protocol config ---
    reasoning_enabled = _truthy(
        _cfg_get("unified", "reasoning", "enabled", default=False), default=False
    )
    env_reasoning = _falsey_env("HERMES_UNIFIED_REASONING")
    if env_reasoning is not None:
        reasoning_enabled = env_reasoning

    smart_guardian_enabled = _truthy(
        _cfg_get("unified", "smart_guardian", "enabled", default=False), default=False
    )
    env_smart_guardian = _falsey_env("HERMES_UNIFIED_SMART_GUARDIAN")
    if env_smart_guardian is not None:
        smart_guardian_enabled = env_smart_guardian

    hard_block_irreversible = _truthy(
        _cfg_get("unified", "smart_guardian", "hard_block_irreversible", default=True),
        default=True,
    )
    persist_reflections = _truthy(
        _cfg_get("unified", "reasoning", "persist_reflections", default=True), default=True
    )
    try:
        guardian_cache_size = int(
            _cfg_get("unified", "smart_guardian", "cache_size", default=512) or 512
        )
    except Exception:
        guardian_cache_size = 512
    try:
        guardian_cache_ttl_seconds = int(
            _cfg_get("unified", "smart_guardian", "cache_ttl_seconds", default=3600) or 3600
        )
    except Exception:
        guardian_cache_ttl_seconds = 3600

    # --- Long-run engine config ---
    longrun_enabled = _truthy(
        _cfg_get("unified", "longrun", "enabled", default=False), default=False
    )
    env_longrun = _falsey_env("HERMES_UNIFIED_LONGRUN")
    if env_longrun is not None:
        longrun_enabled = env_longrun
    try:
        longrun_heartbeat_seconds = float(
            _cfg_get("unified", "longrun", "heartbeat_seconds", default=30.0) or 30.0
        )
    except Exception:
        longrun_heartbeat_seconds = 30.0
    try:
        longrun_reflection_debounce_seconds = float(
            _cfg_get("unified", "longrun", "reflection_debounce_seconds", default=5.0) or 5.0
        )
    except Exception:
        longrun_reflection_debounce_seconds = 5.0
    try:
        longrun_reflection_batch_size = int(
            _cfg_get("unified", "longrun", "reflection_batch_size", default=10) or 10
        )
    except Exception:
        longrun_reflection_batch_size = 10

    # --- Tool router config ---
    tool_router_enabled = _truthy(
        _cfg_get("unified", "tool_router", "enabled", default=False), default=False
    )
    env_tool_router = _falsey_env("HERMES_UNIFIED_TOOL_ROUTER")
    if env_tool_router is not None:
        tool_router_enabled = env_tool_router
    try:
        tool_router_top_n = int(
            _cfg_get("unified", "tool_router", "top_n", default=5) or 5
        )
    except Exception:
        tool_router_top_n = 5
    tool_router_learn = _truthy(
        _cfg_get("unified", "tool_router", "learn", default=True), default=True
    )

    # --- v2 cognitive extensions config ---
    cognitive_tree_enabled = _truthy(
        _cfg_get("unified", "cognitive_tree", "enabled", default=False), default=False
    )
    env_cog = _falsey_env("HERMES_UNIFIED_COGNITIVE_TREE")
    if env_cog is not None:
        cognitive_tree_enabled = env_cog
    try:
        cognitive_tree_n_branches = int(
            _cfg_get("unified", "cognitive_tree", "n_branches", default=3) or 3
        )
    except Exception:
        cognitive_tree_n_branches = 3
    try:
        cognitive_tree_min_confidence = float(
            _cfg_get("unified", "cognitive_tree", "min_confidence", default=0.6) or 0.6
        )
    except Exception:
        cognitive_tree_min_confidence = 0.6
    try:
        cognitive_tree_max_confidence = float(
            _cfg_get("unified", "cognitive_tree", "max_confidence", default=0.85) or 0.85
        )
    except Exception:
        cognitive_tree_max_confidence = 0.85

    hypothesis_enabled = _truthy(
        _cfg_get("unified", "hypothesis", "enabled", default=False), default=False
    )
    env_hyp = _falsey_env("HERMES_UNIFIED_HYPOTHESIS")
    if env_hyp is not None:
        hypothesis_enabled = env_hyp
    try:
        hypothesis_n_hypotheses = int(
            _cfg_get("unified", "hypothesis", "n_hypotheses", default=3) or 3
        )
    except Exception:
        hypothesis_n_hypotheses = 3
    try:
        hypothesis_max_iterations = int(
            _cfg_get("unified", "hypothesis", "max_iterations", default=5) or 5
        )
    except Exception:
        hypothesis_max_iterations = 5
    try:
        hypothesis_confidence_threshold = float(
            _cfg_get("unified", "hypothesis", "confidence_threshold", default=0.8) or 0.8
        )
    except Exception:
        hypothesis_confidence_threshold = 0.8

    context_distiller_enabled = _truthy(
        _cfg_get("unified", "context_distiller", "enabled", default=False), default=False
    )
    env_cd = _falsey_env("HERMES_UNIFIED_CONTEXT_DISTILLER")
    if env_cd is not None:
        context_distiller_enabled = env_cd
    try:
        context_distill_every_n_turns = int(
            _cfg_get("unified", "context_distiller", "every_n_turns", default=10) or 10
        )
    except Exception:
        context_distill_every_n_turns = 10
    try:
        context_distiller_max_items = int(
            _cfg_get("unified", "context_distiller", "max_items", default=30) or 30
        )
    except Exception:
        context_distiller_max_items = 30
    try:
        context_distiller_merge_threshold = int(
            _cfg_get("unified", "context_distiller", "merge_threshold", default=50) or 50
        )
    except Exception:
        context_distiller_merge_threshold = 50

    metacognitive_enabled = _truthy(
        _cfg_get("unified", "metacognitive", "enabled", default=False), default=False
    )
    env_meta = _falsey_env("HERMES_UNIFIED_METACOGNITIVE")
    if env_meta is not None:
        metacognitive_enabled = env_meta
    try:
        metacognitive_self_doubt_threshold = float(
            _cfg_get("unified", "metacognitive", "self_doubt_threshold", default=0.5) or 0.5
        )
    except Exception:
        metacognitive_self_doubt_threshold = 0.5
    try:
        metacognitive_repeated_failure_count = int(
            _cfg_get("unified", "metacognitive", "repeated_failure_count", default=3) or 3
        )
    except Exception:
        metacognitive_repeated_failure_count = 3
    try:
        metacognitive_min_samples = int(
            _cfg_get("unified", "metacognitive", "min_samples", default=5) or 5
        )
    except Exception:
        metacognitive_min_samples = 5

    causal_graph_enabled = _truthy(
        _cfg_get("unified", "causal_graph", "enabled", default=False), default=False
    )
    env_cg = _falsey_env("HERMES_UNIFIED_CAUSAL_GRAPH")
    if env_cg is not None:
        causal_graph_enabled = env_cg

    # --- v2.1 learning + skill synthesis ---
    learning_enabled = _truthy(
        _cfg_get("unified", "learning", "enabled", default=False), default=False
    )
    env_learn = _falsey_env("HERMES_UNIFIED_LEARNING")
    if env_learn is not None:
        learning_enabled = env_learn
    try:
        learning_max_records = int(
            _cfg_get("unified", "learning", "max_records", default=5000) or 5000
        )
    except Exception:
        learning_max_records = 5000
    try:
        learning_extract_every_n_turns = int(
            _cfg_get("unified", "learning", "extract_every_n_turns", default=8) or 8
        )
    except Exception:
        learning_extract_every_n_turns = 8

    skill_synthesis_enabled = _truthy(
        _cfg_get("unified", "skill_synthesis", "enabled", default=False), default=False
    )
    env_skill = _falsey_env("HERMES_UNIFIED_SKILL_SYNTHESIS")
    if env_skill is not None:
        skill_synthesis_enabled = env_skill
    try:
        skill_synthesis_min_occurrences = int(
            _cfg_get("unified", "skill_synthesis", "min_occurrences", default=3) or 3
        )
    except Exception:
        skill_synthesis_min_occurrences = 3
    try:
        skill_synthesis_max_skills = int(
            _cfg_get("unified", "skill_synthesis", "max_skills", default=100) or 100
        )
    except Exception:
        skill_synthesis_max_skills = 100

    # --- v2.2 task planning config ---
    task_planner_enabled = _truthy(
        _cfg_get("unified", "task_planner", "enabled", default=False), default=False
    )
    env_tp = _falsey_env("HERMES_UNIFIED_TASK_PLANNER")
    if env_tp is not None:
        task_planner_enabled = env_tp
    try:
        task_planner_max_subtasks = int(
            _cfg_get("unified", "task_planner", "max_subtasks", default=15) or 15
        )
    except Exception:
        task_planner_max_subtasks = 15
    try:
        task_planner_max_replans = int(
            _cfg_get("unified", "task_planner", "max_replans", default=3) or 3
        )
    except Exception:
        task_planner_max_replans = 3

    # --- v2.3 output formatter config ---
    output_formatter_enabled = _truthy(
        _cfg_get("unified", "output_formatter", "enabled", default=False), default=False
    )
    env_of = _falsey_env("HERMES_UNIFIED_OUTPUT_FORMATTER")
    if env_of is not None:
        output_formatter_enabled = env_of
    output_formatter_summarize_long = _truthy(
        _cfg_get("unified", "output_formatter", "summarize_long", default=False), default=False
    )
    try:
        output_formatter_summarize_threshold = int(
            _cfg_get("unified", "output_formatter", "summarize_threshold", default=8000) or 8000
        )
    except Exception:
        output_formatter_summarize_threshold = 8000

    # --- v3 cognitive extensions config ---
    verifier_enabled = _truthy(
        _cfg_get("unified", "verifier", "enabled", default=False), default=False
    )
    env_ver = _falsey_env("HERMES_UNIFIED_VERIFIER")
    if env_ver is not None:
        verifier_enabled = env_ver
    try:
        verifier_max_iterations = int(
            _cfg_get("unified", "verifier", "max_iterations", default=3) or 3
        )
    except Exception:
        verifier_max_iterations = 3
    try:
        verifier_pass_threshold = float(
            _cfg_get("unified", "verifier", "pass_threshold", default=0.7) or 0.7
        )
    except Exception:
        verifier_pass_threshold = 0.7

    constitution_enabled = _truthy(
        _cfg_get("unified", "constitution", "enabled", default=False), default=False
    )
    env_const = _falsey_env("HERMES_UNIFIED_CONSTITUTION")
    if env_const is not None:
        constitution_enabled = env_const

    slow_thinking_enabled = _truthy(
        _cfg_get("unified", "slow_thinking", "enabled", default=False), default=False
    )
    env_st = _falsey_env("HERMES_UNIFIED_SLOW_THINKING")
    if env_st is not None:
        slow_thinking_enabled = env_st
    slow_thinking_default_level = str(
        _cfg_get("unified", "slow_thinking", "default_level", default="fast") or "fast"
    )
    slow_thinking_store_traces = _truthy(
        _cfg_get("unified", "slow_thinking", "store_traces", default=False), default=False
    )

    ensemble_enabled = _truthy(
        _cfg_get("unified", "ensemble", "enabled", default=False), default=False
    )
    env_ens = _falsey_env("HERMES_UNIFIED_ENSEMBLE")
    if env_ens is not None:
        ensemble_enabled = env_ens
    try:
        ensemble_max_workers = int(
            _cfg_get("unified", "ensemble", "max_workers", default=5) or 5
        )
    except Exception:
        ensemble_max_workers = 5
    try:
        ensemble_timeout_seconds = float(
            _cfg_get("unified", "ensemble", "timeout_seconds", default=60.0) or 60.0
        )
    except Exception:
        ensemble_timeout_seconds = 60.0

    capability_resolver_enabled = _truthy(
        _cfg_get("unified", "capability_resolver", "enabled", default=False), default=False
    )
    env_cr = _falsey_env("HERMES_UNIFIED_CAPABILITY_RESOLVER")
    if env_cr is not None:
        capability_resolver_enabled = env_cr
    capability_resolver_auto_approve = _truthy(
        _cfg_get("unified", "capability_resolver", "auto_approve", default=False), default=False
    )
    capability_resolver_allow_network = _truthy(
        _cfg_get("unified", "capability_resolver", "allow_network", default=True), default=True
    )

    # --- v3.1 Tier 1-2 config ---
    cost_tracker_enabled = _truthy(
        _cfg_get("unified", "cost_tracker", "enabled", default=False), default=False
    )
    try:
        cost_tracker_total_budget = int(
            _cfg_get("unified", "cost_tracker", "total_budget", default=1_000_000) or 1_000_000
        )
    except Exception:
        cost_tracker_total_budget = 1_000_000
    cost_tracker_hard_stop = _truthy(
        _cfg_get("unified", "cost_tracker", "hard_stop", default=True), default=True
    )

    response_cache_enabled = _truthy(
        _cfg_get("unified", "response_cache", "enabled", default=False), default=False
    )
    try:
        response_cache_max_entries = int(
            _cfg_get("unified", "response_cache", "max_entries", default=512) or 512
        )
    except Exception:
        response_cache_max_entries = 512
    try:
        response_cache_ttl_seconds = int(
            _cfg_get("unified", "response_cache", "ttl_seconds", default=3600) or 3600
        )
    except Exception:
        response_cache_ttl_seconds = 3600

    user_model_enabled = _truthy(
        _cfg_get("unified", "user_model", "enabled", default=False), default=False
    )

    clarifier_enabled = _truthy(
        _cfg_get("unified", "clarifier", "enabled", default=False), default=False
    )
    try:
        clarifier_heuristic_threshold = float(
            _cfg_get("unified", "clarifier", "heuristic_threshold", default=0.5) or 0.5
        )
    except Exception:
        clarifier_heuristic_threshold = 0.5
    clarifier_always_use_llm = _truthy(
        _cfg_get("unified", "clarifier", "always_use_llm", default=False), default=False
    )

    streaming_enabled = _truthy(
        _cfg_get("unified", "streaming", "enabled", default=False), default=False
    )

    embedding_enabled = _truthy(
        _cfg_get("unified", "embedding", "enabled", default=False), default=False
    )
    embedding_backend = str(
        _cfg_get("unified", "embedding", "backend", default="local") or "local"
    )
    embedding_model = str(
        _cfg_get("unified", "embedding", "model", default="all-MiniLM-L6-v2") or "all-MiniLM-L6-v2"
    )

    _CONFIG_CACHE = UnifiedConfig(
        enabled=enabled,
        reflexion_enabled=reflexion_enabled,
        guardian_enabled=guardian_enabled,
        auto_prefetch_enabled=auto_prefetch_enabled,
        store_path=store_path,
        block_tools=block_tools,
        max_records=max_records,
        scope_by_cwd=scope_by_cwd,
        reasoning_enabled=reasoning_enabled,
        smart_guardian_enabled=smart_guardian_enabled,
        hard_block_irreversible=hard_block_irreversible,
        persist_reflections=persist_reflections,
        guardian_cache_size=guardian_cache_size,
        guardian_cache_ttl_seconds=guardian_cache_ttl_seconds,
        longrun_enabled=longrun_enabled,
        longrun_heartbeat_seconds=longrun_heartbeat_seconds,
        longrun_reflection_debounce_seconds=longrun_reflection_debounce_seconds,
        longrun_reflection_batch_size=longrun_reflection_batch_size,
        tool_router_enabled=tool_router_enabled,
        tool_router_top_n=tool_router_top_n,
        tool_router_learn=tool_router_learn,
        cognitive_tree_enabled=cognitive_tree_enabled,
        cognitive_tree_n_branches=cognitive_tree_n_branches,
        cognitive_tree_min_confidence=cognitive_tree_min_confidence,
        cognitive_tree_max_confidence=cognitive_tree_max_confidence,
        hypothesis_enabled=hypothesis_enabled,
        hypothesis_n_hypotheses=hypothesis_n_hypotheses,
        hypothesis_max_iterations=hypothesis_max_iterations,
        hypothesis_confidence_threshold=hypothesis_confidence_threshold,
        context_distiller_enabled=context_distiller_enabled,
        context_distill_every_n_turns=context_distill_every_n_turns,
        context_distiller_max_items=context_distiller_max_items,
        context_distiller_merge_threshold=context_distiller_merge_threshold,
        metacognitive_enabled=metacognitive_enabled,
        metacognitive_self_doubt_threshold=metacognitive_self_doubt_threshold,
        metacognitive_repeated_failure_count=metacognitive_repeated_failure_count,
        metacognitive_min_samples=metacognitive_min_samples,
        causal_graph_enabled=causal_graph_enabled,
        learning_enabled=learning_enabled,
        learning_max_records=learning_max_records,
        learning_extract_every_n_turns=learning_extract_every_n_turns,
        skill_synthesis_enabled=skill_synthesis_enabled,
        skill_synthesis_min_occurrences=skill_synthesis_min_occurrences,
        skill_synthesis_max_skills=skill_synthesis_max_skills,
        task_planner_enabled=task_planner_enabled,
        task_planner_max_subtasks=task_planner_max_subtasks,
        task_planner_max_replans=task_planner_max_replans,
        output_formatter_enabled=output_formatter_enabled,
        output_formatter_summarize_long=output_formatter_summarize_long,
        output_formatter_summarize_threshold=output_formatter_summarize_threshold,
        verifier_enabled=verifier_enabled,
        verifier_max_iterations=verifier_max_iterations,
        verifier_pass_threshold=verifier_pass_threshold,
        constitution_enabled=constitution_enabled,
        slow_thinking_enabled=slow_thinking_enabled,
        slow_thinking_default_level=slow_thinking_default_level,
        slow_thinking_store_traces=slow_thinking_store_traces,
        ensemble_enabled=ensemble_enabled,
        ensemble_max_workers=ensemble_max_workers,
        ensemble_timeout_seconds=ensemble_timeout_seconds,
        capability_resolver_enabled=capability_resolver_enabled,
        capability_resolver_auto_approve=capability_resolver_auto_approve,
        capability_resolver_allow_network=capability_resolver_allow_network,
        cost_tracker_enabled=cost_tracker_enabled,
        cost_tracker_total_budget=cost_tracker_total_budget,
        cost_tracker_hard_stop=cost_tracker_hard_stop,
        response_cache_enabled=response_cache_enabled,
        response_cache_max_entries=response_cache_max_entries,
        response_cache_ttl_seconds=response_cache_ttl_seconds,
        user_model_enabled=user_model_enabled,
        clarifier_enabled=clarifier_enabled,
        clarifier_heuristic_threshold=clarifier_heuristic_threshold,
        clarifier_always_use_llm=clarifier_always_use_llm,
        streaming_enabled=streaming_enabled,
        embedding_enabled=embedding_enabled,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )
    return _CONFIG_CACHE
