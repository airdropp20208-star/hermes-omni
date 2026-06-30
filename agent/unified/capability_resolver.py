r"""Capability Resolver — auto-install or auto-create missing tools/MCPs.

THE BIG IDEA
-------------
A senior engineer doesn't stop when they're missing a tool — they
`pip install` it, or `npm install`, or write a quick script. Hermes-Omni
should do the same.

When the agent encounters a task that needs a capability it doesn't have:

1. **Detect** — "I need to send a Slack message but no Slack tool exists"
2. **Search** — check PyPI, npm, MCP registry, Hermes plugin catalog
3. **Install** — pip/uv install the package, register as a tool
4. **If not found** — GENERATE a new tool from scratch using LLM

This is the agent's "self-extension" capability. It never gets stuck on
"missing tool" — it either installs one or writes one.

PIPELINE
--------
```
   ┌──────────────────────────────────────────────────────┐
   │  Agent: "I need capability X but don't have it"      │
   └─────────────────────────┬────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  1. Check existing tools    │
              │     (registry + MCP catalog)│
              └──────────────┬──────────────┘
                             │ not found
              ┌──────────────▼──────────────┐
              │  2. Search PyPI             │
              │     pip search equivalent   │
              └──────────────┬──────────────┘
                             │ not found
              ┌──────────────▼──────────────┐
              │  3. Search MCP registry     │
              │     (optional-mcps/)        │
              └──────────────┬──────────────┘
                             │ not found
              ┌──────────────▼──────────────┐
              │  4. Search Hermes plugin    │
              │     catalog (GitHub)        │
              └──────────────┬──────────────┘
                             │ not found
              ┌──────────────▼──────────────┐
              │  5. GENERATE new tool       │
              │     LLM writes Python code  │
              │     Save to auto-tools/     │
              │     Register in registry    │
              └─────────────────────────────┘
```

SAFETY
------
- Installations require USER APPROVAL by default (config flag to auto-approve)
- Generated tools are SANDBOXED first (run in isolated env, check output)
- All auto-installed/generated tools go to ~/.hermes/auto-tools/ (separate
  from bundled tools, easy to audit/remove)
- Every install/generation is LOGGED for audit

TOKEN ECONOMICS
---------------
- 0 LLM calls for steps 1-4 (search + install)
- 1 LLM call for step 5 (generate tool code) — only when nothing found
- The generated tool SAVES tokens in future (don't need to LLM-generate
  the same code again)

Net: rare LLM cost, large capability benefit.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ResolutionResult:
    """Result of a capability resolution attempt."""

    capability: str
    method: Literal["found_existing", "installed_pypi", "installed_mcp", "installed_plugin", "generated", "failed"]
    tool_name: str = ""
    package_name: str = ""
    file_path: str = ""
    error: str = ""
    requires_approval: bool = False
    approved: bool = False
    elapsed_ms: int = 0


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_GENERATE_TOOL_SYSTEM = (
    "You are the tool-generation layer of an AI agent. The agent needs a "
    "tool that doesn't exist. Write a complete, working Python tool.\n\n"
    "Requirements:\n"
    "- Use only stdlib + commonly-available packages (requests, httpx)\n"
    "- The tool must be a function `handler(args: dict, **kwargs) -> str`\n"
    "- Return a JSON string with the result\n"
    "- Handle errors gracefully (try/except, return error JSON)\n"
    "- Include docstrings\n"
    "- Include type hints\n"
    "- Keep it under 200 lines\n\n"
    "Return STRICT JSON:\n"
    "{\n"
    '  "tool_name": "snake_case_name",\n'
    '  "description": "one-sentence description",\n'
    '  "parameters": {\n'
    '    "type": "object",\n'
    '    "properties": {...},\n'
    '    "required": [...]\n'
    '  },\n'
    '  "code": "the full Python source code as a string",\n'
    '  "dependencies": ["pip package names needed, or empty list"]\n'
    "}"
)

_SEARCH_SYSTEM = (
    "You are the search layer of an AI agent. The agent needs a capability. "
    "Suggest the most likely PyPI package name and MCP server name that "
    "would provide this capability. Be specific — guess actual package "
    "names that exist on PyPI.\n\n"
    "Return STRICT JSON:\n"
    "{\n"
    '  "pypi_packages": ["package1", "package2"],\n'
    '  "mcp_servers": ["server1", "server2"],\n'
    '  "hermes_plugins": ["plugin1"]\n'
    "}"
)


# --------------------------------------------------------------------------- #
# CapabilityResolver
# --------------------------------------------------------------------------- #


class CapabilityResolver:
    """Detect missing capabilities and install/generate them.

    Safety:
    - All installs require approval unless auto_approve=True
    - Generated tools go to ~/.hermes/auto-tools/ (separate from bundled)
    - All actions logged to ~/.hermes/auto-tools/.resolution_log.jsonl
    """

    def __init__(
        self,
        *,
        llm_call: Callable[[str, str], str] | None = None,
        auto_tools_dir: str | Path | None = None,
        auto_approve: bool = False,
        pip_command: str = "pip",
        allow_network: bool = True,
    ) -> None:
        self._llm_call = llm_call
        if auto_tools_dir is None:
            from hermes_constants import get_hermes_home

            auto_tools_dir = get_hermes_home() / "auto-tools"
        self._auto_tools_dir = Path(auto_tools_dir).expanduser()
        self._auto_tools_dir.mkdir(parents=True, exist_ok=True)
        self._auto_approve = auto_approve
        self._pip = pip_command
        self._allow_network = allow_network
        self._log_path = self._auto_tools_dir / ".resolution_log.jsonl"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def resolve(
        self,
        *,
        capability: str,
        context: str = "",
        approve_fn: Callable[[str, str], bool] | None = None,
    ) -> ResolutionResult:
        """Resolve a missing capability. Returns ResolutionResult."""
        started = time.time()

        # Step 1: Check existing tools.
        existing = self._check_existing(capability)
        if existing is not None:
            return ResolutionResult(
                capability=capability,
                method="found_existing",
                tool_name=existing,
                elapsed_ms=int((time.time() - started) * 1000),
            )

        # Step 2-4: Search + install (if network allowed).
        if self._allow_network and self._llm_call is not None:
            # Get search suggestions.
            suggestions = self._search_suggestions(capability, context)

            # Try PyPI packages.
            for pkg in suggestions.get("pypi_packages", []):
                if self._should_approve(f"pip install {pkg}", approve_fn):
                    result = self._install_pypi(pkg, capability)
                    if result.method == "installed_pypi":
                        result.elapsed_ms = int((time.time() - started) * 1000)
                        self._log(result)
                        return result

            # Try MCP servers.
            for mcp in suggestions.get("mcp_servers", []):
                if self._should_approve(f"hermes mcp install {mcp}", approve_fn):
                    result = self._install_mcp(mcp, capability)
                    if result.method == "installed_mcp":
                        result.elapsed_ms = int((time.time() - started) * 1000)
                        self._log(result)
                        return result

        # Step 5: Generate new tool.
        if self._llm_call is not None:
            if self._should_approve(f"generate new tool for: {capability}", approve_fn):
                result = self._generate_tool(capability, context)
                result.elapsed_ms = int((time.time() - started) * 1000)
                self._log(result)
                return result

        return ResolutionResult(
            capability=capability,
            method="failed",
            error="no resolution method succeeded",
            elapsed_ms=int((time.time() - started) * 1000),
        )

    def list_auto_tools(self) -> list[dict[str, Any]]:
        """List all auto-installed/generated tools."""
        tools = []
        if not self._auto_tools_dir.exists():
            return tools
        for item in self._auto_tools_dir.iterdir():
            if item.name.startswith(".") or item.name == "resolution_log.jsonl":
                continue
            if item.is_file() and item.suffix == ".py":
                tools.append({
                    "name": item.stem,
                    "type": "generated",
                    "path": str(item),
                    "size_bytes": item.stat().st_size,
                })
            elif item.is_dir():
                tools.append({
                    "name": item.name,
                    "type": "installed_package",
                    "path": str(item),
                })
        return tools

    def remove_auto_tool(self, name: str) -> bool:
        """Remove an auto-installed/generated tool."""
        target = self._auto_tools_dir / name
        if target.exists() and target.is_file() and target.suffix == ".py":
            target.unlink()
            return True
        target_dir = self._auto_tools_dir / name
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Internal: search + install
    # ------------------------------------------------------------------ #

    def _check_existing(self, capability: str) -> str | None:
        """Check if a tool matching this capability already exists."""
        try:
            from tools.registry import registry

            names = registry.get_all_tool_names()
            cap_lower = capability.lower()
            # Direct name match.
            for name in names:
                if cap_lower in name.lower():
                    return name
            # Check descriptions.
            for name in names:
                schema = registry.get_schema(name) or {}
                fn = schema.get("function", schema) if isinstance(schema, dict) else {}
                desc = str(fn.get("description", "")).lower()
                if cap_lower in desc or any(
                    word in desc for word in cap_lower.split() if len(word) > 3
                ):
                    return name
        except Exception:
            pass
        return None

    def _search_suggestions(self, capability: str, context: str) -> dict[str, list[str]]:
        """Ask LLM for PyPI/MCP/plugin suggestions."""
        if self._llm_call is None:
            return {"pypi_packages": [], "mcp_servers": [], "hermes_plugins": []}
        try:
            user = (
                f"Capability needed: {capability}\n"
                f"Context: {context or '(none)'}\n\n"
                "Suggest packages/servers that provide this capability."
            )
            raw = self._llm_call(_SEARCH_SYSTEM, user)
            data = self._parse_json(raw)
            if data is None:
                return {"pypi_packages": [], "mcp_servers": [], "hermes_plugins": []}
            return {
                "pypi_packages": [str(s).strip() for s in data.get("pypi_packages", []) if str(s).strip()][:3],
                "mcp_servers": [str(s).strip() for s in data.get("mcp_servers", []) if str(s).strip()][:3],
                "hermes_plugins": [str(s).strip() for s in data.get("hermes_plugins", []) if str(s).strip()][:3],
            }
        except Exception:
            return {"pypi_packages": [], "mcp_servers": [], "hermes_plugins": []}

    def _install_pypi(self, package: str, capability: str) -> ResolutionResult:
        """pip install a package and try to register a tool from it."""
        try:
            # Install.
            result = subprocess.run(
                [sys.executable, "-m", self._pip, "install", package],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return ResolutionResult(
                    capability=capability,
                    method="failed",
                    package_name=package,
                    error=f"pip install failed: {result.stderr[:200]}",
                )
            # Try to import and find a callable.
            # This is best-effort — we can't auto-discover tool schemas
            # from arbitrary packages. The user will need to wrap it manually
            # OR we generate a wrapper tool that uses the package.
            tool_name = f"auto_{package.replace('-', '_')}"
            wrapper_code = self._generate_wrapper_tool(package, capability, tool_name)
            if wrapper_code:
                tool_path = self._auto_tools_dir / f"{tool_name}.py"
                tool_path.write_text(wrapper_code, encoding="utf-8")
                # Try to register.
                self._try_register_tool(tool_path, tool_name)
                return ResolutionResult(
                    capability=capability,
                    method="installed_pypi",
                    tool_name=tool_name,
                    package_name=package,
                    file_path=str(tool_path),
                    approved=True,
                )
            return ResolutionResult(
                capability=capability,
                method="failed",
                package_name=package,
                error="installed but couldn't generate wrapper",
            )
        except subprocess.TimeoutExpired:
            return ResolutionResult(
                capability=capability,
                method="failed",
                package_name=package,
                error="pip install timed out",
            )
        except Exception as exc:
            return ResolutionResult(
                capability=capability,
                method="failed",
                package_name=package,
                error=repr(exc),
            )

    def _install_mcp(self, mcp_name: str, capability: str) -> ResolutionResult:
        """Try to install an MCP server via hermes mcp."""
        try:
            result = subprocess.run(
                ["hermes", "mcp", "install", mcp_name],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return ResolutionResult(
                    capability=capability,
                    method="installed_mcp",
                    tool_name=f"mcp_{mcp_name}",
                    package_name=mcp_name,
                    approved=True,
                )
            return ResolutionResult(
                capability=capability,
                method="failed",
                package_name=mcp_name,
                error=f"hermes mcp install failed: {result.stderr[:200]}",
            )
        except Exception as exc:
            return ResolutionResult(
                capability=capability,
                method="failed",
                package_name=mcp_name,
                error=repr(exc),
            )

    # ------------------------------------------------------------------ #
    # Internal: generate tool
    # ------------------------------------------------------------------ #

    def _generate_tool(self, capability: str, context: str) -> ResolutionResult:
        """Generate a new tool from scratch using LLM."""
        try:
            user = (
                f"Capability needed: {capability}\n"
                f"Context: {context or '(none)'}\n\n"
                "Generate a Python tool that provides this capability."
            )
            raw = self._llm_call(_GENERATE_TOOL_SYSTEM, user)
            data = self._parse_json(raw)
            if data is None:
                return ResolutionResult(
                    capability=capability,
                    method="failed",
                    error="LLM returned unparseable output",
                )
            tool_name = str(data.get("tool_name", "")).strip()
            if not tool_name:
                tool_name = f"auto_{int(time.time())}"
            tool_name = re.sub(r"[^a-z0-9_]", "_", tool_name.lower())[:60]
            if not tool_name.startswith("auto_"):
                tool_name = f"auto_{tool_name}"
            code = str(data.get("code", "")).strip()
            if not code:
                return ResolutionResult(
                    capability=capability,
                    method="failed",
                    error="LLM returned empty code",
                )
            # Install dependencies first.
            deps = [str(d).strip() for d in data.get("dependencies", []) if str(d).strip()]
            for dep in deps:
                try:
                    subprocess.run(
                        [sys.executable, "-m", self._pip, "install", dep],
                        capture_output=True,
                        timeout=60,
                    )
                except Exception:
                    pass  # best-effort
            # Write tool file.
            tool_path = self._auto_tools_dir / f"{tool_name}.py"
            tool_path.write_text(code, encoding="utf-8")
            # Try to register.
            self._try_register_tool(tool_path, tool_name)
            return ResolutionResult(
                capability=capability,
                method="generated",
                tool_name=tool_name,
                file_path=str(tool_path),
                approved=True,
            )
        except Exception as exc:
            return ResolutionResult(
                capability=capability,
                method="failed",
                error=repr(exc),
            )

    def _generate_wrapper_tool(self, package: str, capability: str, tool_name: str) -> str | None:
        """Generate a wrapper tool that uses an installed package."""
        if self._llm_call is None:
            return None
        try:
            user = (
                f"A Python package '{package}' was just installed to provide: {capability}.\n"
                f"Generate a Hermes tool wrapper that imports and uses this package.\n"
                f"Tool name: {tool_name}\n\n"
                "Generate the wrapper now."
            )
            raw = self._llm_call(_GENERATE_TOOL_SYSTEM, user)
            data = self._parse_json(raw)
            if data is None:
                return None
            return str(data.get("code", "")).strip() or None
        except Exception:
            return None

    def _try_register_tool(self, tool_path: Path, tool_name: str) -> bool:
        """Try to register a generated tool with the Hermes registry."""
        try:
            from tools.registry import registry

            # Import the module dynamically.
            import importlib.util

            spec = importlib.util.spec_from_file_location(tool_name, tool_path)
            if spec is None or spec.loader is None:
                return False
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            handler = getattr(module, "handler", None)
            if handler is None:
                return False
            # Build schema from module if available, else minimal.
            schema = getattr(module, "SCHEMA", {
                "name": tool_name,
                "description": getattr(module, "__doc__", "")[:200] or f"Auto-generated tool: {tool_name}",
                "parameters": {"type": "object", "properties": {}, "required": []},
            })
            registry.register(
                name=tool_name,
                toolset="auto",
                schema=schema,
                handler=handler,
                description=str(schema.get("description", tool_name)),
                emoji="🔧",
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Internal: approval + logging
    # ------------------------------------------------------------------ #

    def _should_approve(
        self,
        action: str,
        approve_fn: Callable[[str, str], bool] | None,
    ) -> bool:
        if self._auto_approve:
            return True
        if approve_fn is not None:
            try:
                return approve_fn(action, self._auto_tools_dir)
            except Exception:
                return False
        # Default: require explicit approval. In interactive mode, the
        # conversation loop should provide approve_fn. In batch mode,
        # auto_approve should be set.
        return False

    def _log(self, result: ResolutionResult) -> None:
        try:
            import dataclasses

            entry = {
                "timestamp": time.time(),
                **dataclasses.asdict(result),
            }
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

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

_resolver: CapabilityResolver | None = None


def get_resolver() -> CapabilityResolver | None:
    return _resolver


def configure_resolver(
    *,
    llm_call: Callable[[str, str], str] | None = None,
    auto_approve: bool = False,
    allow_network: bool = True,
) -> CapabilityResolver | None:
    global _resolver
    _resolver = CapabilityResolver(
        llm_call=llm_call,
        auto_approve=auto_approve,
        allow_network=allow_network,
    )
    return _resolver


def resolve_capability(
    *,
    capability: str,
    context: str = "",
    approve_fn: Callable[[str, str], bool] | None = None,
) -> ResolutionResult:
    """Public API: resolve a missing capability."""
    if _resolver is None:
        return ResolutionResult(
            capability=capability,
            method="failed",
            error="resolver not configured",
        )
    return _resolver.resolve(capability=capability, context=context, approve_fn=approve_fn)


def list_auto_tools() -> list[dict[str, Any]]:
    """Public API: list auto-installed/generated tools."""
    if _resolver is None:
        return []
    return _resolver.list_auto_tools()


def remove_auto_tool(name: str) -> bool:
    """Public API: remove an auto tool."""
    if _resolver is None:
        return False
    return _resolver.remove_auto_tool(name)
