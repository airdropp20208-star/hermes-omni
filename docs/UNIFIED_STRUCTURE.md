# Hermes Unified Agent - GitHub-ready structure

This repository is structured as a single unified source tree with **Hermes Agent
as the runtime kernel** and vendored source packages from **OmniAgent** and
**AgentScope**.

## Top-level layout

```text
hermes-unified-agent/
├── agent/                         # Hermes core agent modules
│   └── unified/                   # Unified integration layer
│       ├── config.py              # config.yaml/env bridge
│       ├── events.py              # in-process event bus
│       ├── integration.py         # direct core dispatcher facade
│       ├── memory_provider.py     # Hermes MemoryProvider bridge
│       ├── policy.py              # Guardian policy engine
│       ├── reflexion.py           # durable scoped reflexion store
│       └── tracing.py             # optional OpenTelemetry spans
├── tools/
│   └── unified_tools.py           # built-in unified tools registered in Hermes
├── plugins/
│   ├── memory/unified/            # selectable memory.provider: unified
│   └── unified_core/              # optional tracing/compat plugin
├── omniagent/                     # vendored OmniAgent source package
├── agentscope/                    # vendored AgentScope source package
├── tests/unified/                 # unified integration tests
├── third_party_licenses/          # copied upstream license files
├── THIRD_PARTY_NOTICES.md         # third-party source/license notice
├── README_UNIFIED_INTEGRATION.md  # integration overview
├── pyproject.toml                 # packaging includes Hermes + vendored packages
└── MANIFEST.in                    # sdist includes vendored packages/notices
```

## Core execution flow

```text
LLM tool call
  ↓
model_tools.handle_function_call()
  ↓
agent.unified.integration.before_tool_call()
  ↓
Hermes plugin hooks / approval / guardrails
  ↓
Hermes tools.registry.dispatch()
  ↓
agent.unified.integration.after_tool_call()
  ↓
Tool result returned to conversation
```

## Built-in unified tools

The unified tools are registered through Hermes' normal tool registry in
`tools/unified_tools.py`:

- `unified_recall` — recall reflexion lessons related to the current task.
- `unified_reflexion_list` — inspect stored lessons.
- `unified_reflexion_clear` — clear lessons with `confirm=true`.
- `unified_framework_status` — verify unified config and vendored package imports.

## Memory integration

To make recall automatic through Hermes' memory prefetch path:

```yaml
memory:
  provider: unified
```

## Unified config

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

## Optional plugin

The core integration works without plugin loading. The optional plugin adds
compatibility hooks and tracing middleware:

```bash
hermes plugins enable unified_core
```

## Local validation

```bash
python -m py_compile model_tools.py agent/unified/*.py tools/unified_tools.py plugins/unified_core/__init__.py plugins/memory/unified/__init__.py
python -m pytest tests/unified/test_unified_core.py -q
```

Expected result:

```text
8 passed
```

## License/distribution warning

This unified tree vendors OmniAgent (`GPL-3.0-only`) and AgentScope
(`Apache-2.0`). Because OmniAgent is GPL-3.0-only, distributing this combined
source tree should be treated as GPL-3.0-only unless you obtain different terms
from the OmniAgent copyright holders. See `THIRD_PARTY_NOTICES.md` and
`third_party_licenses/`.
