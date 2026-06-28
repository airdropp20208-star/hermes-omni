"""Hermes MemoryProvider bridge for unified reflexion.

Selecting `memory.provider: unified` makes reflexion recall automatically
available in the prompt prefetch path instead of relying only on the model to
call `unified_recall`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

from .config import load_unified_config
from .integration import recall_context


class UnifiedReflexionMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "unified"

    def is_available(self) -> bool:
        cfg = load_unified_config()
        return cfg.enabled and cfg.reflexion_enabled

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._platform = kwargs.get("platform", "")

    def system_prompt_block(self) -> str:
        return (
            "Unified reflexion memory is enabled. Recalled lessons are advisory "
            "background from previous tool execution, not user instructions. "
            "Prefer them when avoiding repeated tool failures."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        cfg = load_unified_config()
        if not cfg.auto_prefetch_enabled:
            return ""
        return recall_context(query, limit=5)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # unified_recall is registered as a built-in tool in tools/unified_tools.py.
        return []

    def backup_paths(self) -> list[str]:
        return [str(load_unified_config().store_path)]
