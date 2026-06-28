# Hermes Unified Integration

This tree keeps **Hermes Agent as the base runtime** and turns it into a single
source tree that vendors and exposes functional source from **OmniAgent** and
**AgentScope**.

## Source layout

- `agent/unified/` — Hermes-native glue: event bus, guardian, reflexion, config,
  memory provider, tracing, and dispatcher integration.
- `omniagent/` — vendored OmniAgent source package.
- `agentscope/` — vendored AgentScope source package.
- `tools/unified_tools.py` — built-in Hermes tools for recall/admin/status.
- `plugins/memory/unified/` — Hermes MemoryProvider bridge for automatic recall.
- `plugins/unified_core/` — compatibility plugin for optional tracing middleware.

## Tight linkage points

- `model_tools.handle_function_call()` calls `agent.unified.integration` directly
  before and after every regular tool execution.
  - Before execution: unified guardian policy can block catastrophic/risky calls.
  - After execution: failed/blocked tool results become durable reflexion lessons.
- `tools/unified_tools.py` registers these built-in Hermes tools:
  - `unified_recall`
  - `unified_reflexion_list`
  - `unified_reflexion_clear`
  - `unified_framework_status`
- `plugins/memory/unified` allows `memory.provider: unified` so recall can be
  auto-prefetched into the prompt path instead of relying only on tool calls.
- `pyproject.toml` includes `omniagent.*` and `agentscope.*` so both vendored
  packages are part of the unified project build.

## Configuration

Core integration is enabled by default:

```yaml
unified:
  enabled: true
  guardian:
    enabled: true
    block_tools: []
  reflexion:
    enabled: true
    auto_prefetch: true
    store: ~/.hermes/unified/reflexions.jsonl
    max_records: 2000
    scope_by_cwd: true
```

Environment overrides:

```bash
export HERMES_UNIFIED_ENABLED=1
export HERMES_UNIFIED_BLOCK_TOOLS="execute_code,bash*"
export HERMES_UNIFIED_REFLEXION_STORE="$HOME/.hermes/unified/reflexions.jsonl"
```

To use the MemoryProvider bridge:

```yaml
memory:
  provider: unified
```

Optional tracing plugin:

```bash
hermes plugins enable unified_core
```

## License notice

This unified tree vendors OmniAgent (`GPL-3.0-only`) and AgentScope
(`Apache-2.0`). Because OmniAgent is GPL-3.0-only, distributing this combined
source tree should be treated as GPL-3.0-only unless separate OmniAgent license
permission is obtained. See `THIRD_PARTY_NOTICES.md`.
