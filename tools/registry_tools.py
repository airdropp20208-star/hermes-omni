"""Skill + API Registry tools — let agent use marketplace + public APIs.

These tools let the agent:
1. Search/install/load skills from curated marketplace (18 GitHub repos)
2. Search/call 1500+ public APIs (no manual URL memorization)
3. Auto-install skill when needed (like CapabilityResolver, but for skills)

Tools registered:
- skill_search      — search marketplace for skills
- skill_list        — list available/installed skills
- skill_install     — download + cache a skill
- skill_load        — load skill content (SKILL.md)
- skill_uninstall   — remove cached skill
- api_search        — search 1500+ public APIs
- api_call          — call a public API (handles auth, params, URL)
- api_categories    — list API categories
"""

from __future__ import annotations

import json
from typing import Any

from agent.unified.api_registry import (
    api_registry_stats,
    call_api,
    fetch_full_api_catalog,
    get_api_info,
    list_api_categories,
    list_apis_by_category,
    search_apis,
)
from agent.unified.skill_registry import (
    install_skill,
    list_available_skills,
    list_installed_skills,
    load_skill,
    search_skills,
    skill_registry_stats,
    uninstall_skill,
)
from tools.registry import registry


# ─── Skill Registry Tools ────────────────────────────────────────────────────


def skill_search_tool(args: dict[str, Any], **_: Any) -> str:
    """Search skill marketplace by query."""
    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)
    results = search_skills(query)
    return json.dumps(
        {
            "query": query,
            "count": len(results),
            "skills": [
                {
                    "id": s["id"],
                    "desc": s["desc"],
                    "category": s["category"],
                    "stars": s["stars"],
                    "installed": s["installed"],
                    "repo": s["repo"],
                }
                for s in results
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def skill_list_tool(args: dict[str, Any], **_: Any) -> str:
    """List skills. Set installed=true to show only installed."""
    installed_only = bool(args.get("installed", False))
    if installed_only:
        skills = list_installed_skills()
    else:
        skills = list_available_skills()
    return json.dumps(
        {
            "count": len(skills),
            "installed_only": installed_only,
            "skills": [
                {
                    "id": s["id"],
                    "desc": s["desc"],
                    "category": s["category"],
                    "installed": s["installed"],
                }
                for s in skills
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def skill_install_tool(args: dict[str, Any], **_: Any) -> str:
    """Install (download + cache) a skill from marketplace."""
    skill_id = str(args.get("skill_id", "")).strip()
    if not skill_id:
        return json.dumps({"error": "skill_id is required"}, ensure_ascii=False)
    force = bool(args.get("force", False))
    result = install_skill(skill_id, force=force)
    return json.dumps(result, ensure_ascii=False, indent=2)


def skill_load_tool(args: dict[str, Any], **_: Any) -> str:
    """Load skill content (SKILL.md). Install if not yet installed."""
    skill_id = str(args.get("skill_id", "")).strip()
    if not skill_id:
        return json.dumps({"error": "skill_id is required"}, ensure_ascii=False)
    auto_install = bool(args.get("auto_install", True))
    # Check if installed.
    installed = {s["id"]: s for s in list_installed_skills()}
    if skill_id not in installed:
        if not auto_install:
            return json.dumps(
                {"success": False, "error": f"Skill '{skill_id}' not installed. Set auto_install=true to auto-install."},
                ensure_ascii=False,
            )
        # Auto-install.
        inst_result = install_skill(skill_id)
        if not inst_result["success"]:
            return json.dumps(inst_result, ensure_ascii=False)
    result = load_skill(skill_id)
    if not result["success"]:
        return json.dumps(result, ensure_ascii=False)
    # Return content (truncated for large skills).
    content = result["content"]
    max_chars = 10000
    truncated = len(content) > max_chars
    return json.dumps(
        {
            "success": True,
            "skill_id": skill_id,
            "content": content[:max_chars],
            "truncated": truncated,
            "full_length": len(content),
        },
        ensure_ascii=False,
        indent=2,
    )


def skill_uninstall_tool(args: dict[str, Any], **_: Any) -> str:
    """Remove a cached skill."""
    skill_id = str(args.get("skill_id", "")).strip()
    if not skill_id:
        return json.dumps({"error": "skill_id is required"}, ensure_ascii=False)
    result = uninstall_skill(skill_id)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ─── API Registry Tools ───────────────────────────────────────────────────────


def api_search_tool(args: dict[str, Any], **_: Any) -> str:
    """Search 1500+ public APIs by query."""
    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)
    limit = int(args.get("limit", 20))
    results = search_apis(query, limit=limit)
    return json.dumps(
        {
            "query": query,
            "count": len(results),
            "apis": [
                {
                    "name": a["name"],
                    "description": a["description"],
                    "auth": a["auth"],
                    "https": a["https"],
                    "cors": a["cors"],
                    "category": a["category"],
                    "link": a["link"],
                }
                for a in results
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def api_call_tool(args: dict[str, Any], **_: Any) -> str:
    """Call a public API. Handles auth, params, URL building."""
    name = str(args.get("name", "")).strip()
    if not name:
        return json.dumps({"error": "name is required (API name)"}, ensure_ascii=False)
    endpoint = str(args.get("endpoint", "")).strip()
    params = args.get("params", {})
    if not isinstance(params, dict):
        params = {}
    api_key = str(args.get("api_key", "")).strip()
    method = str(args.get("method", "GET")).strip().upper()
    result = call_api(
        name,
        endpoint=endpoint,
        params=params,
        api_key=api_key,
        method=method,
    )
    # Truncate data if too large.
    if result.get("success") and isinstance(result.get("data"), str):
        if len(result["data"]) > 5000:
            result["data"] = result["data"][:5000]
            result["truncated"] = True
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


def api_categories_tool(args: dict[str, Any], **_: Any) -> str:
    """List all API categories."""
    categories = list_api_categories()
    return json.dumps(
        {"count": len(categories), "categories": categories},
        ensure_ascii=False,
        indent=2,
    )


def api_list_by_category_tool(args: dict[str, Any], **_: Any) -> str:
    """List APIs in a specific category."""
    category = str(args.get("category", "")).strip()
    if not category:
        return json.dumps({"error": "category is required"}, ensure_ascii=False)
    apis = list_apis_by_category(category)
    return json.dumps(
        {
            "category": category,
            "count": len(apis),
            "apis": [a.to_dict() for a in apis] if apis else [],
        },
        ensure_ascii=False,
        indent=2,
    )


def api_info_tool(args: dict[str, Any], **_: Any) -> str:
    """Get detailed info about a specific API."""
    name = str(args.get("name", "")).strip()
    if not name:
        return json.dumps({"error": "name is required"}, ensure_ascii=False)
    info = get_api_info(name)
    if info is None:
        return json.dumps({"error": f"API '{name}' not found"}, ensure_ascii=False)
    return json.dumps(info, ensure_ascii=False, indent=2)


def api_fetch_catalog_tool(args: dict[str, Any], **_: Any) -> str:
    """Download full 1500+ API catalog from public-apis/public-apis."""
    result = fetch_full_api_catalog()
    return json.dumps(result, ensure_ascii=False, indent=2)


# ─── Schemas + Registration ───────────────────────────────────────────────────


_SKILL_SEARCH_SCHEMA = {
    "name": "skill_search",
    "description": (
        "Search the curated skill marketplace (18 GitHub repos, 25+ skills). "
        "Returns matching skills with id, description, category, stars, install status. "
        "Use when you need a skill for a specific task (e.g., UI/UX, code review, debugging)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (e.g., 'code review', 'ui ux', 'debugging')"},
        },
        "required": ["query"],
    },
}

_SKILL_LIST_SCHEMA = {
    "name": "skill_list",
    "description": "List all available skills in the marketplace, or only installed skills.",
    "parameters": {
        "type": "object",
        "properties": {
            "installed": {"type": "boolean", "description": "If true, list only installed skills. Default: false (all)."},
        },
    },
}

_SKILL_INSTALL_SCHEMA = {
    "name": "skill_install",
    "description": (
        "Download and cache a skill from the marketplace. "
        "After install, use skill_load to get the skill content. "
        "Skills are cached at ~/.hermes/skills/cached/<skill_id>/."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill_id": {"type": "string", "description": "Skill ID from skill_search/skill_list"},
            "force": {"type": "boolean", "description": "Re-download even if already installed. Default: false."},
        },
        "required": ["skill_id"],
    },
}

_SKILL_LOAD_SCHEMA = {
    "name": "skill_load",
    "description": (
        "Load a skill's content (SKILL.md). Auto-installs if not yet cached "
        "(set auto_install=false to disable). Returns the skill content for "
        "you to follow its instructions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill_id": {"type": "string", "description": "Skill ID to load"},
            "auto_install": {"type": "boolean", "description": "Auto-install if not cached. Default: true."},
        },
        "required": ["skill_id"],
    },
}

_SKILL_UNINSTALL_SCHEMA = {
    "name": "skill_uninstall",
    "description": "Remove a cached skill.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill_id": {"type": "string", "description": "Skill ID to remove"},
        },
        "required": ["skill_id"],
    },
}

_API_SEARCH_SCHEMA = {
    "name": "api_search",
    "description": (
        "Search 1500+ public APIs (from public-apis/public-apis). "
        "Returns matching APIs with name, description, auth requirement, HTTPS, CORS. "
        "Use when you need external data (weather, crypto, books, news, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (e.g., 'weather', 'crypto', 'books')"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        },
        "required": ["query"],
    },
}

_API_CALL_SCHEMA = {
    "name": "api_call",
    "description": (
        "Call a public API by name. Handles URL building, auth, params. "
        "First use api_search to find the API name, then call. "
        "For APIs requiring apiKey, pass api_key parameter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "API name (from api_search)"},
            "endpoint": {"type": "string", "description": "API endpoint path (e.g., '/weather'). Default: ''."},
            "params": {"type": "object", "description": "Query parameters as key-value pairs"},
            "api_key": {"type": "string", "description": "API key (for APIs requiring auth). Default: ''."},
            "method": {"type": "string", "enum": ["GET", "POST"], "default": "GET"},
        },
        "required": ["name"],
    },
}

_API_CATEGORIES_SCHEMA = {
    "name": "api_categories",
    "description": "List all API categories (Animals, Books, Finance, Weather, etc.).",
    "parameters": {"type": "object", "properties": {}},
}

_API_LIST_BY_CATEGORY_SCHEMA = {
    "name": "api_list_by_category",
    "description": "List all APIs in a specific category.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Category name (from api_categories)"},
        },
        "required": ["category"],
    },
}

_API_INFO_SCHEMA = {
    "name": "api_info",
    "description": "Get detailed info about a specific API (base URL, auth type, etc.).",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "API name"},
        },
        "required": ["name"],
    },
}

_API_FETCH_CATALOG_SCHEMA = {
    "name": "api_fetch_catalog",
    "description": "Download the full 1500+ API catalog from public-apis/public-apis. Run once to get all APIs.",
    "parameters": {"type": "object", "properties": {}},
}


# ─── Register all tools ───────────────────────────────────────────────────────


def _register_all() -> None:
    """Register all skill + API registry tools."""
    tools = [
        ("skill_search", _SKILL_SEARCH_SCHEMA, skill_search_tool, "🔎", "Search skill marketplace"),
        ("skill_list", _SKILL_LIST_SCHEMA, skill_list_tool, "📋", "List skills"),
        ("skill_install", _SKILL_INSTALL_SCHEMA, skill_install_tool, "📥", "Install a skill"),
        ("skill_load", _SKILL_LOAD_SCHEMA, skill_load_tool, "📂", "Load skill content"),
        ("skill_uninstall", _SKILL_UNINSTALL_SCHEMA, skill_uninstall_tool, "🗑️", "Remove a skill"),
        ("api_search", _API_SEARCH_SCHEMA, api_search_tool, "🔍", "Search public APIs"),
        ("api_call", _API_CALL_SCHEMA, api_call_tool, "🌐", "Call a public API"),
        ("api_categories", _API_CATEGORIES_SCHEMA, api_categories_tool, "📂", "List API categories"),
        ("api_list_by_category", _API_LIST_BY_CATEGORY_SCHEMA, api_list_by_category_tool, "📂", "List APIs by category"),
        ("api_info", _API_INFO_SCHEMA, api_info_tool, "ℹ️", "Get API info"),
        ("api_fetch_catalog", _API_FETCH_CATALOG_SCHEMA, api_fetch_catalog_tool, "📥", "Download full API catalog"),
    ]
    for name, schema, handler, emoji, desc in tools:
        registry.register(
            name=name,
            toolset="registry",
            schema=schema,
            handler=handler,
            description=desc,
            emoji=emoji,
        )


# Auto-register on import.
try:
    _register_all()
except Exception:
    pass
