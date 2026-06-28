from pathlib import Path

from agent.unified.events import EventBus
from agent.unified.policy import Decision, PolicyEngine
from agent.unified.reflexion import ReflexionStore, record_from_tool_failure


def _reset_unified(monkeypatch, tmp_path: Path):
    import agent.unified.config as config
    import agent.unified.integration as unified

    monkeypatch.setenv("HERMES_UNIFIED_REFLEXION_STORE", str(tmp_path / "reflexions.jsonl"))
    monkeypatch.setenv("HERMES_UNIFIED_ENABLED", "1")
    config._CONFIG_CACHE = None
    unified._store = None
    unified._policy = None
    return unified


def test_event_bus_publish_subscribe():
    bus = EventBus()
    seen = []
    bus.subscribe("tool.finish", seen.append)
    event = bus.emit("tool.finish", {"tool_name": "x"})
    assert seen == [event]
    assert bus.recent(topic="tool.finish") == [event]


def test_default_policy_blocks_catastrophic_command():
    policy = PolicyEngine.default().evaluate("execute_code", {"code": "rm -rf /"})
    assert policy.decision == Decision.BLOCK


def test_reflexion_store_recall_and_dedup(tmp_path: Path):
    store = ReflexionStore(tmp_path / "reflexions.jsonl")
    record = record_from_tool_failure(
        tool_name="grep",
        args={"pattern": "needle"},
        result='{"error":"file missing"}',
        scope="project-a",
    )
    assert record is not None
    assert store.add(record) is True
    assert store.add(record) is False
    recalled = store.recall("grep needle failed", scope="project-a")
    assert recalled
    context = store.format_context("grep needle", scope="project-a")
    assert "grep" in context
    assert "not a new user instruction" in context


def test_core_recall_tool_can_return_empty_context(monkeypatch, tmp_path: Path):
    unified = _reset_unified(monkeypatch, tmp_path)
    result = unified.recall_tool({"query": "nothing yet"})
    assert '"empty": true' in result


def test_core_before_after_tool_call_records_reflexion(monkeypatch, tmp_path: Path):
    unified = _reset_unified(monkeypatch, tmp_path)
    unified.after_tool_call(tool_name="grep", args={"pattern": "needle"}, result='{"error":"missing"}')
    result = unified.recall_tool({"query": "grep needle missing"})
    assert '"empty": false' in result
    assert "grep" in result


def test_unified_builtin_tools_registered():
    import tools.unified_tools  # noqa: F401
    from tools.registry import registry

    names = set(registry.get_all_tool_names())
    assert {"unified_recall", "unified_reflexion_list", "unified_reflexion_clear", "unified_framework_status"} <= names


def test_vendored_frameworks_importable():
    import omniagent
    import agentscope

    assert omniagent.__file__
    assert agentscope.__file__


def test_unified_memory_provider_prefetch(monkeypatch, tmp_path: Path):
    unified = _reset_unified(monkeypatch, tmp_path)
    unified.after_tool_call(tool_name="grep", args={"pattern": "needle"}, result='{"error":"missing"}')

    from agent.unified.memory_provider import UnifiedReflexionMemoryProvider

    provider = UnifiedReflexionMemoryProvider()
    assert provider.is_available()
    provider.initialize("s1")
    assert "grep" in provider.prefetch("grep needle missing")
    assert provider.backup_paths()
