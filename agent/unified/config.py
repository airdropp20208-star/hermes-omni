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
    )
    return _CONFIG_CACHE
