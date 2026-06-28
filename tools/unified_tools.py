"""Built-in unified tools.

This module is auto-discovered by ``tools.registry.discover_builtin_tools``.
The tools expose reflexion recall/admin and vendored framework status through
Hermes' normal registry instead of requiring plugin loading.
"""

from __future__ import annotations

from agent.unified.integration import clear_tool, list_tool, recall_tool, status_tool
from tools.registry import registry

_UNIFIED_RECALL_SCHEMA = {
    "name": "unified_recall",
    "description": (
        "Recall lessons recorded by Hermes' unified reflexion layer from "
        "previous tool failures, blocks, and risky actions. Use before retrying "
        "a similar operation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Current task or failure to match."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "scope": {"type": "string", "description": "Optional memory scope. Defaults to current workspace."},
        },
        "required": ["query"],
    },
}

_UNIFIED_LIST_SCHEMA = {
    "name": "unified_reflexion_list",
    "description": "List recent unified reflexion lessons for debugging or review.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            "scope": {"type": "string", "description": "Optional scope filter."},
        },
    },
}

_UNIFIED_CLEAR_SCHEMA = {
    "name": "unified_reflexion_clear",
    "description": "Clear unified reflexion lessons. Requires confirm=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {"type": "string", "description": "Optional scope to clear; omit to clear all."},
            "confirm": {"type": "boolean", "description": "Must be true to perform deletion."},
        },
        "required": ["confirm"],
    },
}

_UNIFIED_STATUS_SCHEMA = {
    "name": "unified_framework_status",
    "description": "Show unified integration status and vendored OmniAgent/AgentScope availability.",
    "parameters": {"type": "object", "properties": {}},
}

registry.register(
    name="unified_recall",
    toolset="unified",
    schema=_UNIFIED_RECALL_SCHEMA,
    handler=recall_tool,
    description="Recall unified reflexion lessons",
    emoji="🧠",
)
registry.register(
    name="unified_reflexion_list",
    toolset="unified",
    schema=_UNIFIED_LIST_SCHEMA,
    handler=list_tool,
    description="List unified reflexion lessons",
    emoji="📚",
)
registry.register(
    name="unified_reflexion_clear",
    toolset="unified",
    schema=_UNIFIED_CLEAR_SCHEMA,
    handler=clear_tool,
    description="Clear unified reflexion lessons",
    emoji="🧹",
)
registry.register(
    name="unified_framework_status",
    toolset="unified",
    schema=_UNIFIED_STATUS_SCHEMA,
    handler=status_tool,
    description="Show unified/vendored framework status",
    emoji="🧩",
)
