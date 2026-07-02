"""Dashboard Server v2 — Full Control Center with Chat + File Upload + Auto-Start.

STARTS AGENT AUTOMATICALLY when dashboard launches.
Chat directly in browser. Upload any file. JSON auto-formatted.

Usage:
    python -m agent.unified.dashboard_server --port 8788

Features:
1. Auto-start Hermes agent on launch
2. WebSocket chat (real-time, streaming)
3. File upload (any type, saved to workspace)
4. JSON formatter (pretty-print tool results)
5. All v1 dashboard features (providers, config, skills, costs, logs)
6. Gateway control (start/stop Telegram)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass

# ─── Config ─────────────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
REPO_ROOT = Path(__file__).parent.parent.parent


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _read_jsonl(path: Path, limit: int = 100) -> list[dict]:
    if not path.exists():
        return []
    records = []
    try:
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
            if len(records) >= limit:
                break
    except Exception:
        pass
    return records


def _scan_skills() -> list[dict]:
    skills = []
    local_repos = REPO_ROOT / "skills" / "local-repos"
    if local_repos.exists():
        for skill_md in local_repos.rglob("SKILL.md"):
            try:
                rel = skill_md.relative_to(local_repos)
                parts = rel.parts
                if len(parts) < 2:
                    continue
                repo_name = parts[0]
                skill_name = parts[-2] if len(parts) >= 2 else skill_md.stem
                content = skill_md.read_text(encoding="utf-8", errors="ignore")[:500]
                desc = ""
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end > 0:
                        for line in content[3:end].splitlines():
                            if line.strip().startswith("description:"):
                                desc = line.split(":", 1)[1].strip().strip('"').strip("'")[:120]
                if not desc:
                    desc = f"Skill from {repo_name}/{skill_name}"
                skills.append({"id": f"local-{repo_name}-{skill_name}", "repo": f"local/{repo_name}", "desc": desc, "category": "local", "stars": "local", "installed": True})
            except Exception:
                continue
    return skills


def _get_current_config() -> dict:
    """Read current provider/model config from config.yaml + .env."""
    config = _read_yaml(HERMES_HOME / "config.yaml") or {}
    model_cfg = config.get("model", {})
    # Read .env for API key
    env_path = HERMES_HOME / ".env"
    api_key = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "_API_KEY=" in line:
                api_key = line.split("=", 1)[1].strip()
                break
    return {
        "provider": model_cfg.get("provider", ""),
        "model": model_cfg.get("default", ""),
        "base_url": model_cfg.get("base_url", ""),
        "has_key": bool(api_key),
        "key_preview": (api_key[:4] + "..." + api_key[-4:]) if len(api_key) > 8 else ("***" if api_key else ""),
    }


def _get_status() -> dict:
    config = _read_yaml(HERMES_HOME / "config.yaml")
    unified = config.get("unified", {}) if config else {}
    feature_keys = ["reasoning","reflexion","smart_guardian","verifier","constitution","slow_thinking","ensemble","embedding","user_model","clarifier","longrun","tool_router","cost_tracker","response_cache","output_formatter","skill_registry","api_registry","multi_provider","learning","skill_synthesis","task_planner"]
    enabled = sum(1 for k in feature_keys if unified.get(k, {}).get("enabled", False))
    mp_config = _read_yaml(HERMES_HOME / "multi_provider.yaml")
    providers = mp_config.get("providers", {}) if mp_config else {}
    total_keys = sum(len(p.get("keys", [])) for p in providers.values())
    skills = _scan_skills()
    costs = _read_jsonl(HERMES_HOME / "unified" / "cost_log.jsonl", limit=1000)
    total_tokens = sum(c.get("total_tokens", 0) for c in costs)
    return {"providers": len(providers), "totalKeys": total_keys, "activeKeys": total_keys, "exhaustedKeys": 0, "totalUsedTokens": total_tokens, "totalQuotaTokens": 0, "installedSkills": len(skills), "totalSkills": len(skills), "enabledFeatures": enabled, "totalFeatures": len(feature_keys), "totalCostTokens": total_tokens, "totalCalls": len(costs), "strategy": mp_config.get("strategy", "none") if mp_config else "none", "uptime": "—"}


def _get_providers() -> list[dict]:
    config = _read_yaml(HERMES_HOME / "multi_provider.yaml")
    if not config:
        return []
    result = []
    for name, pdata in (config.get("providers") or {}).items():
        keys = []
        for k in (pdata.get("keys") or []):
            if isinstance(k, dict):
                key_str = k.get("key", "")
                quota = k.get("quota_tokens", 0)
                keys.append({"id": f"k{len(keys)+1}", "keyPreview": key_str[:4]+"..."+key_str[-4:] if len(key_str)>8 else "***", "quota": quota, "used": 0, "remaining": quota or 999999999, "exhausted": quota>0 and 0>=quota, "enabled": k.get("enabled", True), "errorCount": 0, "lastUsed": None})
            elif isinstance(k, str):
                keys.append({"id": f"k{len(keys)+1}", "keyPreview": k[:4]+"..."+k[-4:] if len(k)>8 else "***", "quota": 0, "used": 0, "remaining": 999999999, "exhausted": False, "enabled": True, "errorCount": 0, "lastUsed": None})
        result.append({"id": name, "name": name, "baseUrl": pdata.get("base_url", ""), "enabled": pdata.get("enabled", True), "models": pdata.get("models") or [], "keys": keys, "activeKeys": len(keys), "totalKeys": len(keys)})
    return result


def _get_config() -> list[dict]:
    config = _read_yaml(HERMES_HOME / "config.yaml")
    unified = config.get("unified", {}) if config else {}
    fields = [
        ("reasoning","unified.reasoning.enabled","Nền tảng","Lập kế hoạch → đánh giá → thực hiện → rút kinh nghiệm"),
        ("reflexion","unified.reflexion.enabled","Nền tảng","Học từ lỗi sai"),
        ("smart_guardian","unified.smart_guardian.enabled","Bảo vệ","Bảo vệ thông minh (LLM đánh giá)"),
        ("verifier","unified.verifier.enabled","Bảo vệ","Tự kiểm tra trước khi gửi"),
        ("constitution","unified.constitution.enabled","Bảo vệ","Nguyên tắc đạo đức"),
        ("slow_thinking","unified.slow_thinking.enabled","Suy luận sâu","4 mức: nhanh/cân bằng/sâu/tối đa"),
        ("ensemble","unified.ensemble.enabled","Nâng cao","Nhiều mô hình + giám khảo (3x token)"),
        ("embedding","unified.embedding.enabled","Nâng cao","Ghi nhớ thông minh (+40%)"),
        ("user_model","unified.user_model.enabled","Nâng cao","Cá nhân hóa theo người dùng"),
        ("clarifier","unified.clarifier.enabled","Nâng cao","Phát hiện mơ hồ → hỏi lại"),
        ("longrun","unified.longrun.enabled","Hạ tầng","Chạy tác vụ nền"),
        ("tool_router","unified.tool_router.enabled","Hạ tầng","Tự chọn công cụ"),
        ("cost_tracker","unified.cost_tracker.enabled","Hạ tầng","Đếm token + ngân sách"),
        ("response_cache","unified.response_cache.enabled","Hạ tầng","Lưu cache (tiết kiệm token)"),
        ("output_formatter","unified.output_formatter.enabled","Hạ tầng","Định dạng Telegram/Slack"),
        ("skill_registry","unified.skill_registry.enabled","Thư viện","Thư viện 113 kỹ năng"),
        ("api_registry","unified.api_registry.enabled","Thư viện","1500+ API công khai"),
        ("multi_provider","unified.multi_provider.enabled","Đa nhà cung cấp","Gộp nhiều API key"),
        ("learning","unified.learning.enabled","Học tập","Học từ mọi tương tác"),
        ("skill_synthesis","unified.skill_synthesis.enabled","Học tập","Tự tạo kỹ năng"),
        ("task_planner","unified.task_planner.enabled","Lập kế hoạch","Chia task + theo dõi"),
    ]
    result = []
    for key, path, cat, desc in fields:
        section = unified.get(key, {})
        enabled = section.get("enabled", False) if isinstance(section, dict) else False
        result.append({"key": key, "path": path, "value": enabled, "defaultValue": False, "enabled": enabled, "category": cat, "description": desc})
    return result


def _get_costs() -> list[dict]:
    costs = _read_jsonl(HERMES_HOME / "unified" / "cost_log.jsonl", limit=1000)
    by_phase: dict[str, dict] = {}
    for c in costs:
        phase = c.get("phase", "unknown")
        if phase not in by_phase:
            by_phase[phase] = {"phase": phase, "tokens": 0, "calls": 0}
        by_phase[phase]["tokens"] += c.get("total_tokens", 0)
        by_phase[phase]["calls"] += 1
    return list(by_phase.values())


def _get_logs() -> list[dict]:
    log_path = HERMES_HOME / "agent.log"
    if not log_path.exists():
        costs = _read_jsonl(HERMES_HOME / "unified" / "cost_log.jsonl", limit=20)
        return [{"timestamp": time.strftime("%H:%M:%S", time.localtime(c.get("timestamp", 0))), "level": "success" if not c.get("cache_hit") else "info", "module": c.get("phase", "unknown"), "message": f"{c.get('phase','?')} call: {c.get('total_tokens',0)} tokens" + (" (cached)" if c.get("cache_hit") else "")} for c in reversed(costs)]
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        logs = []
        for line in reversed(lines[-200:]):
            if len(logs) >= 50:
                break
            level = "info"
            if "ERROR" in line or "error" in line.lower(): level = "error"
            elif "WARN" in line or "warning" in line.lower(): level = "warn"
            elif "success" in line.lower() or "✓" in line: level = "success"
            ts = line[:8] if len(line) > 8 and line[2] == ":" and line[5] == ":" else ""
            if ts: line = line[9:]
            logs.append({"timestamp": ts or time.strftime("%H:%M:%S"), "level": level, "module": "agent", "message": line[:200]})
        return logs
    except Exception:
        return []


# ─── Agent process management ────────────────────────────────────────────────

_agent_process: subprocess.Popen | None = None
_gateway_process: subprocess.Popen | None = None
_agent_instance = None
_conversation_history = []


def _start_agent() -> dict:
    """Start hermes agent in background (interactive mode, piped)."""
    global _agent_process
    if _agent_process is not None and _agent_process.poll() is None:
        return {"success": True, "message": "Agent already running", "pid": _agent_process.pid}
    try:
        _agent_process = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env={**os.environ, "HERMES_HOME": str(HERMES_HOME)},
        )
        return {"success": True, "message": f"Agent started (PID: {_agent_process.pid})", "pid": _agent_process.pid}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _stop_agent() -> dict:
    global _agent_process
    if _agent_process is None or _agent_process.poll() is not None:
        return {"success": True, "message": "Agent not running"}
    try:
        _agent_process.terminate()
        _agent_process.wait(timeout=5)
        _agent_process = None
        return {"success": True, "message": "Agent stopped"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _start_gateway() -> dict:
    """Start hermes gateway for Telegram/etc."""
    global _gateway_process
    if _gateway_process is not None and _gateway_process.poll() is None:
        return {"success": True, "message": "Gateway already running", "pid": _gateway_process.pid}
    try:
        _gateway_process = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env={**os.environ, "HERMES_HOME": str(HERMES_HOME)},
        )
        return {"success": True, "message": f"Gateway started (PID: {_gateway_process.pid})", "pid": _gateway_process.pid}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _stop_gateway() -> dict:
    global _gateway_process
    if _gateway_process is None:
        return {"success": True, "message": "Gateway not running"}
    try:
        _gateway_process.terminate()
        _gateway_process.wait(timeout=5)
        _gateway_process = None
        return {"success": True, "message": "Gateway stopped"}
    except Exception:
        return {"success": False, "error": "Failed to stop"}


def _send_to_agent(message: str) -> dict:
    """Send message directly to agent runtime — in-process, no subprocess.

    Dashboard = agent runtime. Chat goes through run_conversation() directly
    with all cognitive modules active (reasoning, verifier, etc).
    Conversation history persists in-memory between messages.
    """
    global _agent_instance, _conversation_history
    import sys, os

    # Initialize agent on first call
    if _agent_instance is None:
        try:
            os.environ["HERMES_YOLO_MODE"] = "1"
            os.environ["HERMES_ACCEPT_HOOKS"] = "1"

            from hermes_cli.config import load_config
            from hermes_cli.runtime_provider import resolve_runtime_provider
            from hermes_cli.tools_config import _get_platform_tools
            from hermes_cli.fallback_config import get_fallback_chain
            from run_agent import AIAgent
            from hermes_cli.oneshot import _oneshot_clarify_callback, _create_session_db_for_oneshot

            cfg = load_config()
            model_cfg = cfg.get("model") or {}
            cfg_model = model_cfg.get("default") or model_cfg.get("model") or "" if isinstance(model_cfg, dict) else str(model_cfg)
            env_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
            effective_model = env_model or cfg_model

            cfg_provider = ""
            if isinstance(model_cfg, dict):
                cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            current_provider = cfg_provider or os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower() or "auto"

            runtime = resolve_runtime_provider(
                requested=current_provider,
                target_model=effective_model or None,
            )

            toolsets_list = sorted(_get_platform_tools(cfg, "cli"))
            session_db = _create_session_db_for_oneshot()
            _fb = get_fallback_chain(cfg)

            _agent_instance = AIAgent(
                api_key=runtime.get("api_key"),
                base_url=runtime.get("base_url"),
                provider=runtime.get("provider"),
                api_mode=runtime.get("api_mode"),
                model=effective_model,
                enabled_toolsets=toolsets_list,
                quiet_mode=True,
                platform="cli",
                session_db=session_db,
                credential_pool=runtime.get("credential_pool"),
                fallback_model=_fb or None,
                clarify_callback=_oneshot_clarify_callback,
            )
            _agent_instance.suppress_status_output = True
            _agent_instance.stream_delta_callback = None
            _agent_instance.tool_gen_callback = None
            _conversation_history = []
        except Exception as exc:
            return {"success": False, "error": f"Khong the khoi tao agent: {exc!r}"}

    # Run conversation directly
    try:
        from agent.conversation_loop import run_conversation
        result = run_conversation(
            _agent_instance,
            user_message=message,
            conversation_history=_conversation_history,
        )
        _conversation_history = result.get("messages", _conversation_history)
        response = result.get("final_response", "")
        if not response:
            for msg in reversed(_conversation_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    response = msg["content"]
                    break
        return {"success": bool(response), "response": response or "(Agent khong tra loi)"}
    except Exception as exc:
        return {"success": False, "error": f"Loi agent: {exc!r}"}


# ─── Action handlers ────────────────────────────────────────────────────────

def _action_add_key(data: dict) -> dict:
    provider_id = data.get("providerId", "")
    key = data.get("key", "")
    quota = int(data.get("quota", 0) or 0)
    if not provider_id or not key: return {"success": False, "error": "providerId and key required"}
    config_path = HERMES_HOME / "multi_provider.yaml"
    config = _read_yaml(config_path) or {"providers": {}, "strategy": "round-robin"}
    providers = config.setdefault("providers", {})
    if provider_id not in providers: return {"success": False, "error": f"Provider '{provider_id}' not found"}
    keys = providers[provider_id].setdefault("keys", [])
    for existing in keys:
        ek = existing.get("key", "") if isinstance(existing, dict) else existing
        if ek == key: return {"success": False, "error": "Key already exists"}
    keys.append({"key": key, "quota_tokens": quota} if quota else {"key": key})
    try:
        import yaml
        config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        return {"success": True, "providers": _get_providers()}
    except Exception as e: return {"success": False, "error": str(e)}


def _action_toggle_config(data: dict) -> dict:
    path = data.get("path", "")
    if not path: return {"success": False, "error": "path required"}
    config_path = HERMES_HOME / "config.yaml"
    config = _read_yaml(config_path) or {}
    parts = path.split(".")
    if len(parts) < 3: return {"success": False, "error": "Invalid path"}
    current = config
    for part in parts[:-1]:
        if part not in current: current[part] = {}
        current = current[part]
    current[parts[-1]] = not current.get(parts[-1], False)
    try:
        import yaml
        config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        return {"success": True, "config": _get_config()}
    except Exception as e: return {"success": False, "error": str(e)}


def _action_add_provider(data: dict) -> dict:
    name = data.get("name", "")
    base_url = data.get("baseUrl", "")
    models = data.get("models", [])
    key = data.get("key", "")
    quota = int(data.get("quota", 0) or 0)
    if not name or not base_url: return {"success": False, "error": "name and baseUrl required"}
    config_path = HERMES_HOME / "multi_provider.yaml"
    config = _read_yaml(config_path) or {"providers": {}, "strategy": "round-robin"}
    providers = config.setdefault("providers", {})
    if name in providers: return {"success": False, "error": "Provider exists"}
    providers[name] = {"base_url": base_url, "enabled": True, "models": models if isinstance(models, list) else [models], "keys": ([{"key": key, "quota_tokens": quota}] if quota else [{"key": key}]) if key else []}
    try:
        import yaml
        config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        return {"success": True, "providers": _get_providers()}
    except Exception as e: return {"success": False, "error": str(e)}


def _action_set_mode(data: dict) -> dict:
    """Set thinking level + reasoning mode + verify on/off. Writes to config.yaml."""
    thinking = data.get("thinking", "balanced")
    reasoning = data.get("reasoning", "standard")
    verify = data.get("verify", "on")
    config_path = HERMES_HOME / "config.yaml"
    config = _read_yaml(config_path) or {}
    unified = config.setdefault("unified", {})
    st = unified.setdefault("slow_thinking", {})
    if thinking == "fast":
        st["enabled"] = False
    else:
        st["enabled"] = True
        st["default_level"] = thinking
    sr = unified.setdefault("reasoning", {})
    if reasoning == "off":
        sr["enabled"] = False
    else:
        sr["enabled"] = True
        if reasoning == "high":
            st["enabled"] = True
            st["default_level"] = "deep"
        elif reasoning == "max":
            st["enabled"] = True
            st["default_level"] = "max"
    sv = unified.setdefault("verifier", {})
    sv["enabled"] = verify == "on"
    sg = unified.setdefault("smart_guardian", {})
    sg["enabled"] = verify == "on"
    try:
        import yaml
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        return {"success": True, "message": f"Mode: thinking={thinking}, reasoning={reasoning}, verify={verify}", "config": _get_config()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _action_setup_provider(data: dict) -> dict:
    """Setup LLM provider — syncs with hermes config.yaml + .env + provider profiles.

    This writes the SAME format that `hermes setup` / `hermes model` uses:
    - config.yaml: model.provider, model.default, model.base_url
    - .env: PROVIDER_API_KEY=xxx
    - Also writes to provider profile if using plugins/model-providers
    """
    provider = data.get("provider", "")
    api_key = data.get("apiKey", "")
    model = data.get("model", "")
    base_url = data.get("baseUrl", "")
    if not provider or not api_key:
        return {"success": False, "error": "Vui lòng chọn nhà cung cấp và nhập API key"}

    config_path = HERMES_HOME / "config.yaml"
    config = _read_yaml(config_path) or {}

    # Set model config (same format as hermes model command)
    model_cfg = config.setdefault("model", {})
    model_cfg["provider"] = provider
    if model:
        model_cfg["default"] = model
    if base_url:
        model_cfg["base_url"] = base_url

    # Map provider to env var name (same as hermes_cli/auth.py)
    PROVIDER_ENV_KEYS = {
        "zai": "ZAI_API_KEY",
        "xiaomi": "XIAOMI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "custom": "CUSTOM_API_KEY",
    }
    PROVIDER_BASE_URLS = {
        "zai": "https://open.bigmodel.cn/api/paas/v4",
        "xiaomi": "https://api.xiaomimimo.com/v1",
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "openrouter": "https://openrouter.ai/api/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }
    # Auto-fill base_url if not provided
    if not base_url and provider in PROVIDER_BASE_URLS:
        base_url = PROVIDER_BASE_URLS[provider]
    key_var = PROVIDER_ENV_KEYS.get(provider, f"{provider.upper().replace('-', '_')}_API_KEY")

    # Save API key to .env
    env_path = HERMES_HOME / ".env"
    env_lines = []
    if env_path.exists():
        env_lines = env_path.read_text(encoding="utf-8").splitlines()
    # Remove old key for this provider
    env_lines = [l for l in env_lines if not l.startswith(f"{key_var}=")]
    env_lines.append(f"{key_var}={api_key}")

    # Also set HERMES_INFERENCE_PROVIDER env
    env_lines = [l for l in env_lines if not l.startswith("HERMES_INFERENCE_PROVIDER=")]
    env_lines.append(f"HERMES_INFERENCE_PROVIDER={provider}")

    try:
        import yaml
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        return {
            "success": True,
            "message": f"Đã lưu! Provider: {provider}, Model: {model or 'mặc định'}, Key: {api_key[:4]}...{api_key[-4:]}",
            "config": _get_config(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _action_hermes_setup(data: dict) -> dict:
    """Run 'hermes setup' or 'hermes model' in background."""
    import subprocess
    import sys
    cmd = data.get("command", "model")  # "setup" or "model"
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env={**os.environ, "HERMES_HOME": str(HERMES_HOME)},
        )
        # Wait briefly for output
        import select
        readable, _, _ = select.select([proc.stdout], [], [], 5)
        output = ""
        if readable:
            output = proc.stdout.read(4096).decode("utf-8", errors="ignore")
        return {"success": True, "output": output, "pid": proc.pid}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _action_test_key(data: dict) -> dict:
    """Test an API key by making a simple chat request."""
    provider = data.get("provider", "")
    api_key = data.get("apiKey", "")
    base_url = data.get("baseUrl", "")
    model = data.get("model", "")
    if not api_key or not base_url:
        return {"success": False, "error": "apiKey and baseUrl required"}
    try:
        from urllib.request import Request, urlopen
        url = f"{base_url.rstrip('/')}/chat/completions"
        body = json.dumps({
            "model": model or "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hi, respond with just 'OK'"}],
            "max_tokens": 5,
        }).encode("utf-8")
        req = Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }, method="POST")
        with urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"success": True, "message": f"Key works! Response: {content}", "model": result.get("model", "")}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _action_remove_key(data: dict) -> dict:
    """Remove API key from multi_provider.yaml."""
    provider_id = data.get("providerId", "")
    key_preview = data.get("keyPreview", "")
    if not provider_id or not key_preview:
        return {"success": False, "error": "providerId and keyPreview required"}
    config_path = HERMES_HOME / "multi_provider.yaml"
    config = _read_yaml(config_path)
    if not config:
        return {"success": False, "error": "No multi_provider.yaml found"}
    providers = config.get("providers", {})
    if provider_id not in providers:
        return {"success": False, "error": f"Provider '{provider_id}' not found"}
    keys = providers[provider_id].get("keys", [])
    new_keys = []
    for k in keys:
        k_str = k.get("key", "") if isinstance(k, dict) else k
        preview = k_str[:4] + "..." + k_str[-4:] if len(k_str) > 8 else "***"
        if preview != key_preview:
            new_keys.append(k)
    providers[provider_id]["keys"] = new_keys
    try:
        import yaml
        config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        return {"success": True, "providers": _get_providers()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _action_toggle_key(data: dict) -> dict:
    """Enable/disable an API key."""
    provider_id = data.get("providerId", "")
    key_preview = data.get("keyPreview", "")
    if not provider_id or not key_preview:
        return {"success": False, "error": "providerId and keyPreview required"}
    config_path = HERMES_HOME / "multi_provider.yaml"
    config = _read_yaml(config_path)
    if not config:
        return {"success": False, "error": "No multi_provider.yaml found"}
    providers = config.get("providers", {})
    if provider_id not in providers:
        return {"success": False, "error": f"Provider '{provider_id}' not found"}
    keys = providers[provider_id].get("keys", [])
    for k in keys:
        if not isinstance(k, dict):
            continue
        k_str = k.get("key", "")
        preview = k_str[:4] + "..." + k_str[-4:] if len(k_str) > 8 else "***"
        if preview == key_preview:
            k["enabled"] = not k.get("enabled", True)
            try:
                import yaml
                config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True), encoding="utf-8")
                return {"success": True, "providers": _get_providers()}
            except Exception as e:
                return {"success": False, "error": str(e)}
    return {"success": False, "error": "Key not found"}


# ─── HTML (full control center) ─────────────────────────────────────────────


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes-Omni · Bảng điều khiển</title>
<style>
:root{
  --bg:#f5f0e8;--card:#faf6ef;--card-2:#ffffff;--border:#e0d5c3;--border-2:#c9bda6;
  --text:#3d3528;--dim:#7a6f5c;--muted:#a89a82;
  --accent:#c8860d;--accent-2:#b0770a;--accent-soft:rgba(200,134,13,.08);
  --green:#5a8a3a;--green-soft:rgba(90,138,58,.12);
  --red:#c44d4d;--red-soft:rgba(196,77,77,.12);
  --blue:#4a7ba8;--blue-soft:rgba(74,123,168,.12);
  --sidebar:#ede5d6;--hover:#e5dcc9;
  --shadow-xs:0 1px 2px rgba(61,53,40,.04);
  --shadow-sm:0 2px 6px rgba(61,53,40,.05);
  --shadow-md:0 6px 18px rgba(61,53,40,.07);
  --shadow-lg:0 16px 40px rgba(61,53,40,.09);
  --r-sm:6px;--r:10px;--r-lg:14px;--r-xl:18px;
  --mono:"SF Mono",ui-monospace,Menlo,Monaco,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,"Helvetica Neue",sans-serif;
  background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;
  display:flex;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--border-2)}
*{scrollbar-width:thin;scrollbar-color:var(--border) transparent}

/* ─── Sidebar ─── */
.sidebar{
  width:56px;background:var(--sidebar);border-right:1px solid var(--border);
  display:flex;flex-direction:column;align-items:center;
  padding:10px 0 12px;gap:4px;position:sticky;top:0;height:100vh;
  flex-shrink:0;z-index:10;
}
.logo{
  width:38px;height:38px;border-radius:11px;
  background:linear-gradient(135deg,#e0a030 0%,var(--accent) 100%);
  color:#fff;display:flex;align-items:center;justify-content:center;
  font-size:1.05rem;margin-bottom:8px;
  box-shadow:0 4px 12px rgba(200,134,13,.35),inset 0 1px 0 rgba(255,255,255,.25);
}
.nav-btn{
  width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;
  font-size:1.05rem;cursor:pointer;border:none;background:transparent;color:var(--dim);
  transition:background .18s ease,color .18s ease,transform .12s ease;
}
.nav-btn:hover{background:var(--hover);color:var(--text)}
.nav-btn:active{transform:scale(.94)}
.nav-btn.active{
  background:var(--accent);color:#fff;
  box-shadow:0 3px 10px rgba(200,134,13,.28),inset 0 1px 0 rgba(255,255,255,.18);
}
.nav-spacer{flex:1}
.nav-btn.start{color:var(--green);background:var(--green-soft)}
.nav-btn.start:hover{background:rgba(90,138,58,.22)}
.nav-btn.stop{color:var(--red);background:var(--red-soft)}
.nav-btn.stop:hover{background:rgba(196,77,77,.22)}

/* ─── Main ─── */
.main{flex:1;display:flex;flex-direction:column;min-width:0;height:100vh}

/* ─── Topbar ─── */
.topbar{
  padding:14px 22px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:var(--card);gap:12px;flex-shrink:0;
}
.topbar-title{font-weight:700;font-size:1.05rem;letter-spacing:-.01em}
.topbar-actions{display:flex;gap:6px;align-items:center}
.badge{
  padding:4px 11px;border-radius:9999px;font-size:.68rem;font-weight:600;
  letter-spacing:.02em;display:inline-flex;align-items:center;gap:4px;
}
.badge-g{background:var(--green-soft);color:var(--green)}
.badge-r{background:var(--red-soft);color:var(--red)}
.btn{
  padding:8px 14px;border-radius:var(--r-sm);font-size:.76rem;font-weight:600;
  cursor:pointer;border:none;transition:all .18s ease;font-family:inherit;
  display:inline-flex;align-items:center;gap:5px;line-height:1;
}
.btn-p{background:var(--accent);color:#fff;box-shadow:0 2px 6px rgba(200,134,13,.25)}
.btn-p:hover{background:var(--accent-2);box-shadow:0 4px 12px rgba(200,134,13,.32)}
.btn-g{background:var(--green);color:#fff}
.btn-g:hover{background:#4d7a30}
.btn-r{background:var(--red);color:#fff}
.btn-r:hover{background:#b04040}
.btn-h{background:var(--card);color:var(--dim);border:1px solid var(--border)}
.btn-h:hover{background:var(--hover);color:var(--text);border-color:var(--border-2)}

/* ─── Views ─── */
.view{display:none;flex:1;overflow-y:auto;padding:22px}
.view.active{display:block}
.chat-view{display:none;flex:1;flex-direction:column;min-height:0}
.chat-view.active{display:flex}

/* ─── Chat messages ─── */
.chat-msgs{flex:1;overflow-y:auto;padding:22px 22px 8px;scroll-behavior:smooth}
.load-more{
  text-align:center;padding:8px 12px;color:var(--accent);font-size:.72rem;
  cursor:pointer;margin-bottom:14px;font-weight:600;
  border:1px solid transparent;border-radius:var(--r-sm);transition:all .15s ease;
}
.load-more:hover{text-decoration:underline;background:var(--accent-soft);border-color:var(--border)}
.msg{
  max-width:72%;margin-bottom:14px;padding:11px 15px;border-radius:14px;
  font-size:.85rem;line-height:1.55;word-break:break-word;white-space:pre-wrap;
  animation:msgIn .22s ease;
}
@keyframes msgIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
.msg.user{
  background:linear-gradient(135deg,#d49615 0%,var(--accent) 100%);color:#fff;
  margin-left:auto;border-bottom-right-radius:4px;
  box-shadow:0 4px 14px rgba(200,134,13,.22);
}
.msg.agent{
  background:var(--card);border:1px solid var(--border);margin-right:auto;
  border-bottom-left-radius:4px;box-shadow:var(--shadow-xs);
}
.msg.sys{
  background:transparent;color:var(--muted);font-size:.7rem;text-align:center;
  margin:0 auto;max-width:90%;padding:6px;font-style:italic;
}
.msg pre{
  background:rgba(61,53,40,.06);padding:10px 12px;border-radius:var(--r-sm);
  font-size:.78rem;margin-top:8px;overflow-x:auto;
  font-family:var(--mono);color:var(--text);line-height:1.5;
}
.msg code{
  background:rgba(61,53,40,.06);padding:2px 5px;border-radius:4px;
  font-family:var(--mono);font-size:.85em;
}
.msg.agent pre{background:#2d2620;color:#f0e8d8;border:1px solid #3d3528}

/* ─── Chat bottom ─── */
.chat-bottom{padding:12px 22px 16px;border-top:1px solid var(--border);background:var(--card);flex-shrink:0}
.file-area{
  padding:11px;border:1.5px dashed var(--border-2);border-radius:var(--r);
  text-align:center;font-size:.72rem;color:var(--muted);cursor:pointer;
  margin-bottom:10px;transition:all .18s ease;background:rgba(245,240,232,.5);
}
.file-area:hover{border-color:var(--accent);background:var(--accent-soft);color:var(--accent-2)}
.file-list{font-size:.7rem;color:var(--dim);margin-top:4px;margin-bottom:8px;font-family:var(--mono)}
.mode-bar{
  display:flex;gap:10px;align-items:center;padding:10px 0 8px;
  border-top:1px solid var(--border);margin-top:6px;flex-wrap:wrap;
}
.mode-group{display:flex;align-items:center;gap:4px}
.mode-label{
  font-size:.62rem;color:var(--muted);font-weight:700;text-transform:uppercase;
  letter-spacing:.04em;margin-right:4px;
}
.mode-btn{
  padding:4px 10px;border-radius:var(--r-sm);font-size:.68rem;font-weight:600;
  cursor:pointer;border:1px solid var(--border);background:var(--card);color:var(--dim);
  transition:all .15s ease;font-family:inherit;
}
.mode-btn:hover{border-color:var(--accent);color:var(--text)}
.mode-btn.active{background:var(--accent);color:#fff;border-color:var(--accent);box-shadow:0 2px 6px rgba(200,134,13,.25)}
.mode-divider{width:1px;height:18px;background:var(--border);margin:0 4px}
.mode-info{font-size:.62rem;color:var(--muted);margin-left:auto;font-family:var(--mono);font-weight:600}
.chat-input-row{display:flex;gap:8px;align-items:flex-end;margin-top:8px}
.chat-input-row textarea{
  flex:1;padding:11px 14px;border:1px solid var(--border);border-radius:var(--r);
  font-size:.85rem;background:var(--bg);color:var(--text);resize:none;
  font-family:inherit;line-height:1.5;max-height:120px;transition:all .15s ease;
}
.chat-input-row textarea:focus{outline:none;border-color:var(--accent);background:var(--card-2);box-shadow:0 0 0 3px var(--accent-soft)}
.chat-input-row .btn{padding:11px 20px;font-size:.8rem}

/* ─── Overview stat grid ─── */
.sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px;margin-bottom:16px}
.sc{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:18px 18px 16px;box-shadow:var(--shadow-xs);transition:all .2s ease;
  position:relative;overflow:hidden;
}
.sc::before{
  content:"";position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--accent),#e0a030);opacity:.7;
}
.sc:hover{box-shadow:var(--shadow-md);transform:translateY(-2px);border-color:var(--border-2)}
.sl{font-size:.68rem;text-transform:uppercase;color:var(--muted);font-weight:700;letter-spacing:.06em}
.sv{font-size:1.85rem;font-weight:800;margin:8px 0 2px;color:var(--text);letter-spacing:-.02em;line-height:1.1}
.ss{font-size:.72rem;color:var(--dim)}

/* ─── Cards ─── */
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px;margin-bottom:12px;box-shadow:var(--shadow-xs)}

/* ─── Provider cards ─── */
.pc{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px;margin-bottom:12px;box-shadow:var(--shadow-xs);transition:border-color .15s ease}
.pc:hover{border-color:var(--border-2)}
.kr{display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:var(--r-sm);background:var(--bg);margin-bottom:5px}
.kp{font-family:var(--mono);font-size:.68rem;color:var(--dim);min-width:110px;font-weight:600}
.qb{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.qf{height:100%;border-radius:3px;transition:width .3s ease}
.qf.g{background:var(--green)} .qf.y{background:#d4a017} .qf.r{background:var(--red)}
.ks{font-size:.58rem;padding:2px 7px;border-radius:4px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;min-width:64px;text-align:center}
.ks.active{background:var(--green-soft);color:var(--green)}
.ks.exhausted{background:var(--red-soft);color:var(--red)}
.ks.disabled{background:rgba(153,153,153,.15);color:var(--muted)}
.mb{display:inline-block;background:var(--accent-soft);color:var(--accent);padding:2px 8px;border-radius:4px;font-size:.6rem;font-family:var(--mono);margin:0 4px 4px 0;font-weight:600}

/* ─── Config ─── */
.cf{display:flex;align-items:center;justify-content:space-between;padding:11px 14px;border-radius:var(--r-sm);background:var(--card);margin-bottom:4px;transition:background .15s ease;border:1px solid transparent}
.cf:hover{background:var(--hover);border-color:var(--border)}
.cfp{font-family:var(--mono);font-size:.66rem;color:var(--muted)}
.cfd{font-size:.76rem;color:var(--dim);margin-top:3px}
.tg{width:38px;height:21px;background:var(--border-2);border-radius:11px;cursor:pointer;position:relative;transition:background .2s ease;flex-shrink:0}
.tg.on{background:var(--green)}
.tg::after{content:"";position:absolute;top:2px;left:2px;width:17px;height:17px;background:#fff;border-radius:50%;transition:left .2s ease;box-shadow:0 1px 3px rgba(0,0,0,.18)}
.tg.on::after{left:19px}

/* ─── Skills grid ─── */
.sk-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px}
.skc{background:var(--card);border:1px solid var(--border);border-radius:var(--r-sm);padding:12px 14px;transition:all .15s ease}
.skc:hover{border-color:var(--border-2);box-shadow:var(--shadow-xs)}
.skn{font-weight:700;font-size:.76rem}
.skr{font-size:.58rem;color:var(--muted);font-family:var(--mono)}
.skde{font-size:.72rem;color:var(--dim);margin:4px 0 2px}

/* ─── JSON / Search ─── */
.json-in{width:100%;height:80px;padding:10px;border:1px solid var(--border);border-radius:var(--r-sm);font-family:var(--mono);font-size:.72rem;background:var(--bg);color:var(--text);resize:vertical}
.json-in:focus{outline:none;border-color:var(--accent)}
.json-out{padding:10px;border:1px solid var(--border);border-radius:var(--r-sm);background:var(--bg);font-family:var(--mono);font-size:.72rem;white-space:pre-wrap;word-break:break-word;max-height:240px;overflow-y:auto;margin-top:8px}
.search{width:100%;padding:9px 13px;border:1px solid var(--border);border-radius:var(--r-sm);font-size:.8rem;background:var(--card);color:var(--text);margin-bottom:10px;font-family:inherit;transition:all .15s ease}
.search:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
select.search{cursor:pointer;appearance:none;background-image:url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%237a6f5c' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");background-repeat:no-repeat;background-position:right 10px center;background-size:14px;padding-right:32px}
select.search option{background:var(--card);color:var(--text)}

/* ─── Section headers ─── */
.section-title{font-weight:700;font-size:.92rem;margin-bottom:4px;letter-spacing:-.01em}
.section-sub{font-size:.74rem;color:var(--muted);margin-bottom:14px}
.cfg-cat{font-weight:700;color:var(--accent);font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;margin:14px 0 6px;padding-bottom:4px;border-bottom:1px solid var(--border)}

/* ─── Mobile ─── */
@media(max-width:768px){
  body{flex-direction:column}
  .sidebar{width:100%;height:auto;flex-direction:row;position:sticky;top:0;border-right:none;border-bottom:1px solid var(--border);padding:6px 10px;gap:3px;overflow-x:auto;background:rgba(237,229,214,.97);backdrop-filter:blur(10px)}
  .logo{margin-bottom:0;margin-right:6px;flex-shrink:0;width:34px;height:34px;font-size:.95rem}
  .nav-spacer{display:none}
  .nav-btn{width:36px;height:36px;flex-shrink:0}
  .main{height:calc(100vh - 50px)}
  .msg{max-width:88%}
  .sg{grid-template-columns:1fr 1fr;gap:8px}
  .sc{padding:14px}
  .sv{font-size:1.4rem}
  .chat-msgs{padding:14px}
  .chat-bottom{padding:10px 14px 12px}
  .topbar{padding:10px 14px}
  .topbar-title{font-size:.95rem}
  .mode-bar{gap:6px}
  .mode-divider{display:none}
  .view{padding:14px}
}
</style></head>
<body>
<nav class="sidebar">
  <div class="logo">⚡</div>
  <button class="nav-btn active" onclick="nv(event,'chat')" title="Trò chuyện">💬</button>
  <button class="nav-btn" onclick="nv(event,'overview')" title="Tổng quan">📊</button>
  <button class="nav-btn" onclick="nv(event,'providers')" title="Nhà cung cấp API">🔌</button>
  <button class="nav-btn" onclick="nv(event,'config')" title="Cấu hình tính năng">⚙️</button>
  <button class="nav-btn" onclick="nv(event,'costs')" title="Thống kê chi phí">💰</button>
  <button class="nav-btn" onclick="nv(event,'logs')" title="Nhật ký hoạt động">📋</button>
  <div class="nav-spacer"></div>
  <button class="nav-btn start" onclick="startAgent()" title="Khởi động agent">▶</button>
  <button class="nav-btn stop" onclick="stopAgent()" title="Dừng agent">⏹</button>
</nav>
<div class="main">
<div class="topbar"><div class="topbar-title" id="vt">Trò chuyện</div><div class="topbar-actions">
<span class="badge badge-r" id="gs">GW: Off</span>
<button class="btn btn-h" onclick="startGW()">Bật Gateway</button></div></div>

<div id="chat" class="chat-view active">
  <div class="chat-msgs" id="cm"><div class="msg sys">Hermes-Omni • Hỏi gì cũng được</div></div>
  <div class="chat-bottom">
    <div class="file-area" id="fa" onclick="document.getElementById('fi').click()">📎 Kéo thả tệp hoặc nhấn để chọn<input type="file" id="fi" multiple style="display:none" onchange="upF(this.files)"></div>
    <div class="file-list" id="fl"></div>
    <div class="mode-bar" id="modeBar" style="display:none">
      <div class="mode-group"><span class="mode-label">🧠 Thinking</span>
        <button class="mode-btn" onclick="setMode('thinking','fast',this)">Fast</button>
        <button class="mode-btn active" onclick="setMode('thinking','balanced',this)">Balanced</button>
        <button class="mode-btn" onclick="setMode('thinking','deep',this)">Deep</button>
        <button class="mode-btn" onclick="setMode('thinking','max',this)">Max</button>
      </div>
      <div class="mode-divider"></div>
      <div class="mode-group"><span class="mode-label">⚡ Reasoning</span>
        <button class="mode-btn" onclick="setMode('reasoning','off',this)">Off</button>
        <button class="mode-btn active" onclick="setMode('reasoning','standard',this)">Std</button>
        <button class="mode-btn" onclick="setMode('reasoning','high',this)">High</button>
        <button class="mode-btn" onclick="setMode('reasoning','max',this)">Max</button>
      </div>
      <div class="mode-divider"></div>
      <div class="mode-group"><span class="mode-label">🛡️ Verify</span>
        <button class="mode-btn" onclick="setMode('verify','off',this)">Off</button>
        <button class="mode-btn active" onclick="setMode('verify','on',this)">On</button>
      </div>
      <span class="mode-info" id="mode-info" style="display:none">balanced · std · verify</span>
    </div>
    <div class="chat-input-row">
      <textarea id="ci" placeholder="Nhập tin nhắn cho Hermes..." rows="1" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}" oninput="this.style.height='auto';this.style.height=Math.min(this.scrollHeight,100)+'px'"></textarea>
      <button class="btn btn-h" onclick="document.getElementById('modePanel').style.display=document.getElementById('modePanel').style.display=='none'?'block':'none'" style="font-size:.65rem;padding:.3rem .5rem" title="Chế độ suy luận">⚙️</button>
<button class="btn btn-p" onclick="send()">Gửi ➤</button>
    </div>
  </div>
</div>

<div id="overview" class="view"><div class="sg" id="sc"></div></div>

<div id="providers" class="view">
  <div id="pl"></div>
  <div class="card" style="margin-top:10px">
    <div class="section-title">🔑 Thiết lập API Key</div>
    <div class="section-sub">Cấu hình nhà cung cấp AI cho agent</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <select id="sp-provider" class="search" style="margin:0">
        <option value="">Chọn nhà cung cấp...</option>
        <option value="zai">GLM (z.ai) — Miễn phí</option>
        <option value="xiaomi">Xiaomi MiMo</option>
        <option value="openai">OpenAI (GPT)</option>
        <option value="anthropic">Anthropic (Claude)</option>
        <option value="openrouter">OpenRouter (nhiều model)</option>
        <option value="deepseek">DeepSeek</option>
        <option value="custom">Tùy chỉnh</option>
      </select>
      <input id="sp-model" class="search" style="margin:0" placeholder="Model (vd: glm-4.6)">
      <input id="sp-baseurl" class="search" style="margin:0" placeholder="Base URL (vd: https://open.bigmodel.cn/api/paas/v4)">
      <input id="sp-apikey" class="search" style="margin:0" type="password" placeholder="API Key">
    </div>
    <div style="display:flex;gap:8px;margin-top:12px">
      <button class="btn btn-p" onclick="setupProvider()">💾 Lưu cấu hình</button>
      <button class="btn btn-h" onclick="testKey()">🧪 Kiểm tra key</button>
    </div>
    <div id="sp-result" style="font-size:.74rem;margin-top:10px;color:var(--dim);padding:8px 10px;background:var(--bg);border-radius:var(--r-sm);min-height:18px"></div>
  </div>
  <button class="btn btn-h" onclick="addP()" style="margin-top:8px">+ Thêm nhà cung cấp (nhiều key)</button>
</div>

<div id="config" class="view">
  <input class="search" placeholder="🔍 Tìm cấu hình..." onkeyup="fc(this.value)">
  <div id="cl"></div>
</div>

<div id="costs" class="view"><div id="cs"></div></div>

<div id="logs" class="view"><div class="card" style="max-height:520px;overflow:auto" id="lf"></div></div>

</div>

<script>
const T={chat:'Trò chuyện',overview:'Tổng quan',providers:'Nhà cung cấp API',config:'Cấu hình',costs:'Chi phí',logs:'Nhật ký hoạt động'};
let msgs=[],cnt=25;
let curMode={thinking:'balanced',reasoning:'standard',verify:'on'};
function setMode(type,val,btn){document.querySelectorAll('.mode-group').forEach(g=>{if(g.querySelector('.mode-label').textContent.toLowerCase().includes(type==='thinking'?'thinking':type==='reasoning'?'reasoning':'verify')){g.querySelectorAll('.mode-btn').forEach(b=>b.classList.remove('active'))}});btn.classList.add('active');curMode[type]=val;const info=document.getElementById('mode-info');info.textContent=curMode.thinking+' · '+curMode.reasoning+' · '+(curMode.verify==='on'?'verify':'no-verify');post('set-mode',curMode).then(r=>{if(r&&r.success)addM('sys','⚙️ '+type+' = '+val)})}
function nv(e,id){document.querySelectorAll('.view,.chat-view').forEach(v=>v.classList.remove('active'));document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));document.getElementById(id).classList.add('active');e.target.closest('.nav-btn').classList.add('active');document.getElementById('vt').textContent=T[id]}
function esc(t){return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function fmtR(t){if(!t)return'';try{return'<pre>'+esc(JSON.stringify(JSON.parse(t),null,2))+'</pre>'}catch(e){return esc(t)}}
function render(){const c=document.getElementById('cm');const v=msgs.slice(-cnt);let h='';if(msgs.length>cnt)h+='<div class="load-more" onclick="cnt+=20;render()">↑ Tải cũ hơn ('+(msgs.length-cnt)+')</div>';v.forEach(m=>h+='<div class="msg '+m.t+'">'+m.c+'</div>');c.innerHTML=h;c.scrollTop=c.scrollHeight}
function addM(t,c){msgs.push({t,c});render()}
async function send(){const i=document.getElementById('ci');const m=i.value.trim();if(!m)return;i.value='';i.style.height='auto';addM('user',esc(m));addM('sys','⏳...');const r=await post('chat',{message:m});const sm=document.querySelectorAll('.msg.sys');if(sm.length)sm[sm.length-1].remove();if(r&&r.success)addM('agent',fmtR(r.response));else addM('sys','❌ '+(r?r.error:'No response'))}
async function upF(fs){const fl=document.getElementById('fl');for(const f of fs){const fd=new FormData();fd.append('file',f);try{const r=await fetch('/api/upload',{method:'POST',body:fd});const res=await r.json();if(res.success){fl.innerHTML+='✓ '+f.name+'<br>';addM('sys','📎 '+f.name)}else fl.innerHTML+='❌ '+f.name+'<br>'}catch(e){fl.innerHTML+='❌<br>'}}}
const fa=document.getElementById('fa');fa.ondragover=e=>{e.preventDefault();fa.style.borderColor='var(--accent)'};fa.ondragleave=()=>fa.style.borderColor='';fa.ondrop=e=>{e.preventDefault();fa.style.borderColor='';upF(e.dataTransfer.files)};
function fj(){const v=document.getElementById('ji').value.trim(),f=document.getElementById('jf').value.trim(),o=document.getElementById('jo');if(!v){o.textContent='Kết quả...';return}try{let j=JSON.parse(v);if(f){if(j[f]!==undefined)j=j[f];else{const n={};for(const[k,v2]of Object.entries(j))if(k.includes(f))n[k]=v2;j=Object.keys(n).length?n:j}}o.innerHTML='<pre>'+esc(JSON.stringify(j,null,2))+'</pre>'}catch(e){o.textContent='❌ '+e.message}}
async function api(p){try{return await(await fetch('/api/'+p)).json()}catch(e){return null}}
async function post(a,d){try{return await api('action/'+a,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d||{})})}catch(e){return null}}
function fn(n){return n>1e6?(n/1e6).toFixed(1)+'M':n>1e3?(n/1e3).toFixed(0)+'K':n}
async function ca(){const s=await api('agent-status');const e=document.getElementById('as');if(s&&s.running){e.className='badge badge-g';e.textContent='● PID:'+s.pid}else{e.className='badge badge-r';e.textContent='● Off'}}
async function startAgent(){const r=await post('start-agent');if(r&&r.success){addM('sys','🟢 '+r.message);ca()}else addM('sys','❌ '+(r?r.error:'Fail'))}
async function stopAgent(){const r=await post('stop-agent');if(r&&r.success){addM('sys','🔴 Stopped');ca()}}
async function startGW(){const r=await post('start-gateway');const e=document.getElementById('gs');if(r&&r.success){e.className='badge badge-g';e.textContent='GW: Bật'}else addM('sys','❌ '+(r?r.error:'Fail'))}
async function ref(){const s=await api('status');if(!s)return;document.getElementById('sc').innerHTML='<div class="sc"><div class="sl">Providers</div><div class="sv">'+s.providers+'</div><div class="ss">'+s.activeKeys+' keys</div></div><div class="sc"><div class="sl">Skills</div><div class="sv">'+s.installedSkills+'</div><div class="ss">installed</div></div><div class="sc"><div class="sl">Features</div><div class="sv">'+s.enabledFeatures+'/'+s.totalFeatures+'</div><div class="ss">enabled</div></div><div class="sc"><div class="sl">Tokens</div><div class="sv">'+fn(s.totalCostTokens)+'</div><div class="ss">'+s.totalCalls+' calls</div></div>'}
async function rP(){const p=await api('providers');if(!p)return;document.getElementById('pl').innerHTML=p.map(pr=>'<div class="pc"><div style="font-weight:700;font-size:.85rem">'+pr.name+'</div><div style="font-size:.6rem;color:var(--muted);font-family:monospace">'+pr.baseUrl+'</div><div style="margin:.15rem 0">'+pr.models.map(m=>'<span class="mb">'+m+'</span>').join('')+'</div>'+pr.keys.map(k=>{const pct=k.quota>0?Math.round(k.used/k.quota*100):0;const c=pct>80?'r':pct>50?'y':'g';const st=k.exhausted?'exhausted':k.enabled?'active':'disabled';return '<div class="kr"><span class="kp">'+k.keyPreview+'</span><div class="qb"><div class="qf '+c+'" style="width:'+Math.max(pct,2)+'%"></div></div><span class="ks '+st+'">'+st+'</span></div>'}).join('')+'<button class="btn btn-h" style="font-size:.6rem;margin-top:.15rem" onclick="addK(\''+pr.id+'\')">+ Add Key</button></div>').join('')}
async function rC(){const c=await api('config');if(!c)return;const g={};c.forEach(f=>{if(!g[f.category])g[f.category]=[];g[f.category].push(f)});let h='';for(const[cat,fs]of Object.entries(g)){const en=fs.filter(f=>f.enabled).length;h+='<div style="margin-bottom:.6rem"><div style="font-weight:700;color:var(--accent);font-size:.7rem;text-transform:uppercase;margin-bottom:.15rem">'+cat+' ('+en+'/'+fs.length+')</div>'+fs.map(f=>'<div class="cf"><div><div class="cfp">'+f.path+'</div><div class="cfd">'+f.description+'</div></div><div class="tg '+(f.enabled?'on':'')+'" onclick="tc(\''+f.path+'\',this)"></div></div>').join('')+'</div>'}document.getElementById('cl').innerHTML=h}
async function rS(){const s=await api('skills');if(!s)return;let h='<div class="card"><b>📚 Thư viện kỹ năng</b><br><span style="font-size:.75rem;color:var(--dim)">'+s.length+' kỹ năng. Agent tự tìm khi cần.</span></div><div class="card">';s.slice(0,30).forEach(sk=>{const sn=sk.id.replace('local-','').replace('claude-','').replace('senior-','');h+='<div style="padding:.2rem 0;font-size:.72rem;border-bottom:1px solid var(--border)"><b>'+sn+'</b> <span style="color:var(--muted)">'+sk.desc.substring(0,60)+'</span></div>'});if(s.length>30)h+='<div style="text-align:center;padding:.3rem;color:var(--accent);font-size:.7rem">+'+(s.length-30)+' kỹ năng khác</div>';h+='</div>';document.getElementById('sl').innerHTML=h}
async function rCost(){const c=await api('costs');if(!c)return;const max=Math.max(...c.map(x=>x.tokens),1);document.getElementById('cs').innerHTML='<div class="sg"><div class="sc"><div class="sl">Tokens</div><div class="sv">'+fn(c.reduce((s,x)=>s+x.tokens,0))+'</div></div><div class="sc"><div class="sl">Calls</div><div class="sv">'+c.reduce((s,x)=>s+x.calls,0)+'</div></div></div><div class="card"><div style="display:flex;align-items:flex-end;gap:.2rem;height:140px">'+c.map(x=>{const h=Math.round(x.tokens/max*120);return '<div style="flex:1;background:var(--accent);border-radius:3px 3px 0 0;height:'+h+'px;min-height:2px;position:relative"><div style="position:absolute;top:-10px;left:50%;transform:translateX(-50%);font-size:.5rem;color:var(--dim)">'+fn(x.tokens)+'</div><div style="position:absolute;bottom:-12px;left:50%;transform:translateX(-50%);font-size:.45rem;color:var(--muted);white-space:nowrap">'+x.phase+'</div></div>'}).join('')+'</div></div>'}
async function rL(){const l=await api('logs');if(!l)return;document.getElementById('lf').innerHTML=l.map(x=>'<div style="display:flex;gap:.2rem;padding:.1rem;font-size:.65rem;border-bottom:1px solid var(--border)"><span style="font-family:monospace;color:var(--muted)">'+x.timestamp+'</span><span style="font-weight:700;color:var(--'+(x.level==='success'?'green':x.level==='error'?'red':x.level==='warn'?'#d4a017':'blue')+'")">'+x.level.toUpperCase()+'</span><span style="color:var(--accent);font-family:monospace">'+x.module+'</span><span style="flex:1">'+x.message+'</span></div>').join('')}
async function tc(p,el){const r=await post('toggle-config',{path:p});if(r&&r.success){el.classList.toggle('on');ref()}}
function addK(pid){const k=prompt('Key:');if(!k)return;const q=prompt('Quota (0=∞):','0')||'0';post('add-key',{providerId:pid,key:k,quota:parseInt(q)}).then(r=>{if(r&&r.success)rP();else alert(r?r.error:'Fail')})}
function addP(){const n=prompt('Name:');if(!n)return;const u=prompt('URL:');if(!u)return;const m=prompt('Models:');const k=prompt('Key:');post('add-provider',{name:n,baseUrl:u,models:m?m.split(','):[],key:k||'',quota:0}).then(r=>{if(r&&r.success)rP();else alert(r?r.error:'Fail')})}
function setupProvider(){const p=document.getElementById('sp-provider').value,m=document.getElementById('sp-model').value,u=document.getElementById('sp-baseurl').value,k=document.getElementById('sp-apikey').value;if(!p||!k){document.getElementById('sp-result').innerHTML='❌ Chọn provider + nhập key';return}post('setup-provider',{provider:p,apiKey:k,model:m,baseUrl:u}).then(r=>{if(r&&r.success){document.getElementById('sp-result').innerHTML='✅ '+r.message;ref()}else document.getElementById('sp-result').innerHTML='❌ '+(r?r.error:'Fail')})}
function testKey(){const p=document.getElementById('sp-provider').value,m=document.getElementById('sp-model').value,u=document.getElementById('sp-baseurl').value,k=document.getElementById('sp-apikey').value;if(!k||!u){document.getElementById('sp-result').innerHTML='❌ Nhập key + base URL';return}document.getElementById('sp-result').innerHTML='⏳ Đang test...';post('test-key',{provider:p,apiKey:k,baseUrl:u,model:m}).then(r=>{if(r&&r.success)document.getElementById('sp-result').innerHTML='✅ '+r.message;else document.getElementById('sp-result').innerHTML='❌ '+(r?r.error:'Fail')})}
function fc(q){q=q.toLowerCase();document.querySelectorAll('.cf').forEach(f=>f.style.display=f.textContent.toLowerCase().includes(q)?'':'none')}
function fs(q){q=q.toLowerCase();document.querySelectorAll('.skc').forEach(c=>c.style.display=c.textContent.toLowerCase().includes(q)?'':'none')}
async function loadCurrentConfig(){const c=await api('current-config');if(!c)return;const el=document.getElementById('sp-result');if(c.provider){el.innerHTML='📋 Hiện tại: <b>'+c.provider+'</b> | Model: <b>'+c.model+'</b> | Key: '+(c.has_key?'<b>'+c.key_preview+'</b> ✅':'<b>chưa có</b> ❌');document.getElementById('sp-provider').value=c.provider;if(c.model)document.getElementById('sp-model').value=c.model;if(c.base_url)document.getElementById('sp-baseurl').value=c.base_url}else{el.innerHTML='⚠ Chưa cấu hình nhà cung cấp nào. Hãy chọn provider + nhập key.'}}
ca();ref();rP();rC();rCost();rL();loadCurrentConfig();
setInterval(ca,5000);setInterval(ref,10000);setInterval(rL,3000);
// Auto-start: không cần — chat dùng hermes -m one-shot trực tiếp
</script>
</body></html>"""
# ─── HTTP Server ────────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html": self._send_html(DASHBOARD_HTML)
        elif path.startswith("/api/status"): self._send_json(_get_status())
        elif path.startswith("/api/providers"): self._send_json(_get_providers())
        elif path.startswith("/api/config"): self._send_json(_get_config())
        elif path.startswith("/api/skills"): self._send_json(_scan_skills())
        elif path.startswith("/api/costs"): self._send_json(_get_costs())
        elif path.startswith("/api/logs"): self._send_json(_get_logs())
        elif path.startswith("/api/agent-status"): self._send_json({"running": _agent_process is not None and _agent_process.poll() is None, "pid": _agent_process.pid if _agent_process else None})
        elif path.startswith("/api/current-config"): self._send_json(_get_current_config())
        elif path.startswith("/api/download/"): self._serve_file(path.replace("/api/download/", ""))
        else: self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            length = int(self.headers.get("Content-Length", 0))
            if path == "/api/upload":
                self._handle_upload()
                return
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if path == "/api/action/start-agent": self._send_json(_start_agent())
        elif path == "/api/action/stop-agent": self._send_json(_stop_agent())
        elif path == "/api/action/start-gateway": self._send_json(_start_gateway())
        elif path == "/api/action/chat": self._send_json(_send_to_agent(data.get("message", "")))
        elif path == "/api/action/add-key": self._send_json(_action_add_key(data))
        elif path == "/api/action/remove-key": self._send_json(_action_remove_key(data))
        elif path == "/api/action/toggle-key": self._send_json(_action_toggle_key(data))
        elif path == "/api/action/toggle-config": self._send_json(_action_toggle_config(data))
        elif path == "/api/action/add-provider": self._send_json(_action_add_provider(data))
        elif path == "/api/action/set-mode": self._send_json(_action_set_mode(data))
        elif path == "/api/action/setup-provider": self._send_json(_action_setup_provider(data))
        elif path == "/api/action/hermes-setup": self._send_json(_action_hermes_setup(data))
        elif path == "/api/action/test-key": self._send_json(_action_test_key(data))
        elif path == "/api/action/run-eval":
            import subprocess, sys as _sys
            eval_script = REPO_ROOT / "scripts" / "evaluate_cognitive.py"
            try:
                r = subprocess.run([_sys.executable, str(eval_script)], capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT))
                self._send_json({"success": r.returncode == 0, "output": r.stdout[-2000:]})
            except Exception as e: self._send_json({"success": False, "error": str(e)})
        else: self._send_json({"error": "unknown"}, 404)

    def _handle_upload(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json({"success": False, "error": "Not multipart"})
            return
        upload_dir = HERMES_HOME / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        # Simple multipart parse
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            boundary = content_type.split("boundary=")[1].encode()
            parts = body.split(b"--" + boundary)
            saved = []
            for part in parts:
                if b"Content-Disposition" not in part:
                    continue
                # Extract filename
                disp_start = part.find(b'filename="')
                if disp_start < 0:
                    continue
                disp_start += 10
                disp_end = part.find(b'"', disp_start)
                filename = part[disp_start:disp_end].decode("utf-8", errors="ignore")
                if not filename:
                    continue
                # Find content start (after \r\n\r\n)
                content_start = part.find(b"\r\n\r\n")
                if content_start < 0:
                    continue
                content_start += 4
                content_end = part.rfind(b"\r\n--")
                if content_end < 0:
                    content_end = len(part)
                file_data = part[content_start:content_end]
                # Save
                safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")
                filepath = upload_dir / safe_name
                filepath.write_bytes(file_data)
                saved.append({"name": filename, "size": len(file_data), "path": str(filepath)})
            if saved:
                self._send_json({"success": True, "files": saved})
            else:
                self._send_json({"success": False, "error": "No files found in upload"})
        except Exception as exc:
            self._send_json({"success": False, "error": str(exc)})

    def _serve_file(self, filename: str):
        filepath = HERMES_HOME / "uploads" / filename
        if not filepath.exists():
            self._send_json({"error": "File not found"}, 404)
            return
        data = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data, code: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def run_server(port: int = 8788, *, auto_start_agent: bool = True) -> None:
    """Run dashboard server. Auto-starts agent if auto_start_agent=True."""
    if auto_start_agent:
        print("Auto-starting Hermes agent...")
        result = _start_agent()
        print(f"  {result.get('message', result.get('error', 'unknown'))}")

    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"═══ Hermes-Omni Control Center v4 ═══")
    print(f"  URL:      http://localhost:{port}")
    print(f"  Hermes:   {HERMES_HOME}")
    print(f"  Skills:   {REPO_ROOT / 'skills' / 'local-repos'}")
    print(f"  Uploads:  {HERMES_HOME / 'uploads'}")
    print(f"  Auto-start agent: {auto_start_agent}")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        _stop_agent()
        _stop_gateway()
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hermes-Omni Dashboard v4")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--no-auto-start", action="store_true", help="Don't auto-start agent")
    args = parser.parse_args()
    run_server(port=args.port, auto_start_agent=not args.no_auto_start)
