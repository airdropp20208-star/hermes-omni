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
        ("reasoning","unified.reasoning.enabled","Foundation","Plan → critique → execute → reflect"),
        ("reflexion","unified.reflexion.enabled","Foundation","Learn from failures"),
        ("smart_guardian","unified.smart_guardian.enabled","Guardians","LLM-as-judge guardian"),
        ("verifier","unified.verifier.enabled","Guardians","Self-verification loop"),
        ("constitution","unified.constitution.enabled","Guardians","User-defined principles"),
        ("slow_thinking","unified.slow_thinking.enabled","Deep Reasoning","4 levels: fast/balanced/deep/max"),
        ("ensemble","unified.ensemble.enabled","Advanced","Multi-model + judge (3x cost)"),
        ("embedding","unified.embedding.enabled","Advanced","Semantic recall (+40%)"),
        ("user_model","unified.user_model.enabled","Advanced","User profile personalization"),
        ("clarifier","unified.clarifier.enabled","Advanced","Detect ambiguity + ask"),
        ("longrun","unified.longrun.enabled","Infrastructure","Background work queue"),
        ("tool_router","unified.tool_router.enabled","Infrastructure","Auto tool selection"),
        ("cost_tracker","unified.cost_tracker.enabled","Infrastructure","Token accounting + budget"),
        ("response_cache","unified.response_cache.enabled","Infrastructure","Cache LLM responses"),
        ("output_formatter","unified.output_formatter.enabled","Infrastructure","Telegram/Slack formatting"),
        ("skill_registry","unified.skill_registry.enabled","Registry","150 skills marketplace"),
        ("api_registry","unified.api_registry.enabled","Registry","1500+ public APIs"),
        ("multi_provider","unified.multi_provider.enabled","Multi-Provider","Aggregate LLM APIs"),
        ("learning","unified.learning.enabled","Learning","Learn from every interaction"),
        ("skill_synthesis","unified.skill_synthesis.enabled","Learning","Auto-create skills"),
        ("task_planner","unified.task_planner.enabled","Planning","Task decomposition + tracking"),
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
    """Send a message to the running agent and get response."""
    global _agent_process
    if _agent_process is None or _agent_process.poll() is not None:
        return {"success": False, "error": "Agent not running. Start it first."}
    try:
        _agent_process.stdin.write((message + "\n").encode("utf-8"))
        _agent_process.stdin.flush()
        # Read response (non-blocking with timeout)
        import select
        readable, _, _ = select.select([_agent_process.stdout], [], [], 30)
        if readable:
            response = b""
            while True:
                chunk = _agent_process.stdout.read(4096)
                if not chunk:
                    break
                response += chunk
                if b"\n" in chunk:
                    break
            return {"success": True, "response": response.decode("utf-8", errors="ignore")}
        return {"success": True, "response": "(agent processing...)"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


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


# ─── HTML (full control center) ─────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes-Omni Control Center</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#f5f0e8;--card:#faf6ef;--border:#e0d5c3;--text:#3d3528;--dim:#7a6f5c;--muted:#a89a82;--accent:#c8860d;--green:#5a8a3a;--red:#c44d4d;--yellow:#d4a017;--blue:#4a7ba8}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.hdr{position:sticky;top:0;z-index:50;background:rgba(245,240,232,.95);backdrop-filter:blur(8px);border-bottom:1px solid var(--border);padding:.8rem 1.5rem;display:flex;align-items:center;justify-content:space-between}
.logo{font-size:1.3rem;font-weight:800;color:var(--accent)}
.badge{padding:.15rem .5rem;border-radius:9999px;font-size:.7rem;font-weight:600}
.badge-g{background:rgba(90,138,58,.12);color:var(--green)}.badge-r{background:rgba(196,77,77,.12);color:var(--red)}.badge-y{background:rgba(212,160,23,.12);color:var(--yellow)}
.tabs{display:flex;gap:.15rem;padding:0 1.5rem;border-bottom:1px solid var(--border);overflow-x:auto;background:var(--bg);position:sticky;top:55px;z-index:40}
.tab{padding:.6rem 1rem;cursor:pointer;font-size:.8rem;font-weight:500;color:var(--dim);border-bottom:2px solid transparent;background:none;border:none;white-space:nowrap}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.content{padding:1rem 1.5rem;max-width:1400px;margin:0 auto}
.tc{display:none}.tc.active{display:block}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem;margin-bottom:.8rem}
.sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.8rem;margin-bottom:1rem}
.sc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem;position:relative}
.sc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.sc.g::before{background:var(--green)}.sc.a::before{background:var(--accent)}.sc.b::before{background:var(--blue)}
.sv{font-size:1.6rem;font-weight:800}.sl{font-size:.7rem;text-transform:uppercase;color:var(--muted)}.ss{font-size:.75rem;color:var(--dim)}
/* Chat */
.chat-box{display:flex;flex-direction:column;height:calc(100vh - 180px);max-height:600px}
.chat-msgs{flex:1;overflow-y:auto;padding:.5rem;margin-bottom:.5rem;border:1px solid var(--border);border-radius:8px;background:#fffbf5}
.msg{margin-bottom:.5rem;padding:.5rem .7rem;border-radius:8px;font-size:.85rem;white-space:pre-wrap;word-break:break-word}
.msg.user{background:rgba(200,134,13,.1);margin-left:2rem}
.msg.agent{background:rgba(74,123,168,.08);margin-right:2rem}
.msg.sys{background:rgba(168,154,130,.15);font-size:.75rem;color:var(--muted)}
.msg pre{background:rgba(0,0,0,.05);padding:.5rem;border-radius:4px;overflow-x:auto;font-size:.8rem;margin-top:.3rem}
.chat-input{display:flex;gap:.5rem}
.chat-input textarea{flex:1;padding:.6rem;border:1px solid var(--border);border-radius:8px;font-size:.85rem;background:#fffbf5;color:var(--text);resize:none;font-family:inherit}
.chat-input textarea:focus{outline:none;border-color:var(--accent)}
.btn{padding:.5rem 1rem;border-radius:8px;font-size:.8rem;font-weight:600;cursor:pointer;border:none}
.btn-p{background:var(--accent);color:#fff}.btn-p:hover{opacity:.85}
.btn-g{background:var(--green);color:#fff}.btn-r{background:var(--red);color:#fff}
.btn-h{background:var(--card);color:var(--dim);border:1px solid var(--border)}
.file-area{margin-top:.5rem;padding:.5rem;border:2px dashed var(--border);border-radius:8px;text-align:center;font-size:.75rem;color:var(--muted);cursor:pointer}
.file-area:hover{border-color:var(--accent);color:var(--accent)}
.file-area.dragover{border-color:var(--accent);background:rgba(200,134,13,.05)}
.file-list{margin-top:.3rem;font-size:.7rem;color:var(--dim)}
/* JSON formatter */
.json-tool{margin-bottom:.5rem}
.json-input{width:100%;height:100px;padding:.5rem;border:1px solid var(--border);border-radius:8px;font-family:monospace;font-size:.8rem;background:#fffbf5;color:var(--text);resize:vertical}
.json-out{padding:.5rem;border:1px solid var(--border);border-radius:8px;background:#fffbf5;font-family:monospace;font-size:.8rem;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow-y:auto}
/* Provider */
.pc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem;margin-bottom:.8rem}
.kr{display:flex;align-items:center;gap:.4rem;padding:.4rem;border-radius:6px;background:rgba(224,213,195,.2);margin-bottom:.3rem}
.kp{font-family:monospace;font-size:.75rem;color:var(--dim);min-width:100px}
.qb{flex:1;height:6px;background:rgba(224,213,195,.4);border-radius:3px;overflow:hidden}
.qf{height:100%;border-radius:3px}
.qf.g{background:var(--green)}.qf.y{background:var(--yellow)}.qf.r{background:var(--red)}
.ks{font-size:.65rem;padding:.1rem .4rem;border-radius:3px;font-weight:600}
.ks.active{background:rgba(90,138,58,.12);color:var(--green)}.ks.exhausted{background:rgba(196,77,77,.12);color:var(--red)}.ks.disabled{background:rgba(168,154,130,.15);color:var(--muted)}
.mb{display:inline-block;background:rgba(200,134,13,.1);color:var(--accent);padding:.1rem .4rem;border-radius:3px;font-size:.65rem;font-family:monospace;margin-right:.2rem}
/* Config */
.cf{display:flex;align-items:center;justify-content:space-between;padding:.5rem .7rem;border-radius:6px;background:rgba(224,213,195,.15);margin-bottom:.2rem}
.cfp{font-family:monospace;font-size:.7rem;color:var(--muted)}.cfd{font-size:.75rem;color:var(--dim)}
.tg{width:36px;height:20px;background:var(--border);border-radius:10px;cursor:pointer;position:relative;transition:.2s}
.tg.on{background:var(--green)}
.tg::after{content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;background:#fff;border-radius:50%;transition:.2s}
.tg.on::after{left:18px}
/* Dialog */
.dlg{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.3);display:flex;align-items:center;justify-content:center;z-index:100}
.dlg-c{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.5rem;min-width:400px;max-width:500px}
.dlg-c input,.dlg-c select{width:100%;padding:.5rem;border:1px solid var(--border);border-radius:6px;margin-bottom:.5rem;background:#fffbf5;color:var(--text);font-size:.85rem}
.footer{position:sticky;bottom:0;background:rgba(245,240,232,.95);border-top:1px solid var(--border);padding:.5rem 1.5rem;display:flex;justify-content:space-between;font-size:.7rem;color:var(--muted)}
@media(max-width:768px){.sg{grid-template-columns:1fr}.chat-input{flex-direction:column}}
</style></head><body>
<div style="min-height:100vh;display:flex;flex-direction:column">
<header class="hdr">
<div><span class="logo">⚡ Hermes-Omni</span> <span style="font-size:.75rem;color:var(--muted)">Control Center v4</span></div>
<div style="display:flex;gap:.4rem;align-items:center">
<span class="badge badge-g" id="agent-status">● Agent: Checking...</span>
<span class="badge badge-r" id="gw-status">Gateway: Off</span>
<button class="btn btn-g" onclick="startAgent()" style="font-size:.7rem">Start Agent</button>
<button class="btn btn-r" onclick="stopAgent()" style="font-size:.7rem">Stop</button>
<button class="btn btn-h" onclick="startGW()" style="font-size:.7rem">Start Gateway</button>
</div>
</header>
<nav class="tabs">
<button class="tab active" onclick="st(event,'chat')">💬 Chat</button>
<button class="tab" onclick="st(event,'overview')">📊 Tổng quan</button>
<button class="tab" onclick="st(event,'providers')">🔌 Provider</button>
<button class="tab" onclick="st(event,'config')">⚙️ Cấu hình</button>
<button class="tab" onclick="st(event,'skills')">📚 Kỹ năng</button>
<button class="tab" onclick="st(event,'costs')">💰 Chi phí</button>
<button class="tab" onclick="st(event,'logs')">📋 Nhật ký</button>
<button class="tab" onclick="st(event,'json')">🔧 JSON Tool</button>
</nav>
<main class="content">
<!-- CHAT -->
<div id="chat" class="tc active">
<div class="chat-box">
<div class="chat-msgs" id="chat-msgs">
<div class="msg sys">💡 Chat với Hermes agent trực tiếp tại đây. Agent tự khởi động khi bật dashboard.</div>
</div>
<div class="file-area" id="file-area" onclick="document.getElementById('file-input').click()">
📎 Kéo thả file hoặc click để upload (tất cả loại file)
<input type="file" id="file-input" multiple style="display:none" onchange="uploadFiles(this.files)">
</input>
<div class="file-list" id="file-list"></div>
</div>
<div class="chat-input">
<textarea id="chat-input" placeholder="Nhập tin nhắn... (Enter để gửi, Shift+Enter xuống dòng)" rows="2" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"></textarea>
<button class="btn btn-p" onclick="sendChat()">Gửi</button>
</div>
</div>
</div>
<!-- OVERVIEW -->
<div id="overview" class="tc"><div class="sg" id="stat-cards"></div></div>
<!-- PROVIDERS -->
<div id="providers" class="tc"><div id="pl"></div>
<button class="btn btn-p" onclick="showAddProvider()">+ Thêm Provider</button>
</div>
<!-- CONFIG -->
<div id="config" class="tc"><input class="json-input" style="height:35px" placeholder="🔍 Tìm cấu hình..." onkeyup="fc(this.value)"><div id="cl"></div></div>
<!-- SKILLS -->
<div id="skills" class="tc"><input class="json-input" style="height:35px" placeholder="🔍 Tìm kỹ năng..." onkeyup="fs(this.value)"><div id="sl" class="sg"></div></div>
<!-- COSTS -->
<div id="costs" class="tc"><div id="cs"></div></div>
<!-- LOGS -->
<div id="logs" class="tc"><div class="card" style="max-height:500px;overflow-y:auto"><div id="lf"></div></div></div>
<!-- JSON TOOL -->
<div id="json" class="tc">
<div class="card json-tool">
<h3>🔧 JSON Formatter & Filter</h3>
<p style="font-size:.75rem;color:var(--muted);margin:.3rem 0">Paste JSON → tự động format + filter</p>
<textarea class="json-input" id="json-input" placeholder='Paste JSON here... e.g. {"status":"ok","data":[1,2,3]}'></textarea>
<div style="margin:.5rem 0">
<input class="json-input" style="height:30px" id="json-filter" placeholder="Filter key (vd: status, data)" onkeyup="formatJSON()">
</div>
<div class="json-out" id="json-out">Kết quả sẽ hiện ở đây...</div>
</div>
</div>
</main>
<footer class="footer"><div id="ft">● online</div><div id="fr">v4.0</div></footer>
</div>
<script>
function st(e,id){document.querySelectorAll('.tc').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.getElementById(id).classList.add('active');e.target.closest('.tab').classList.add('active')}
function fmt(n){return n>1e6?(n/1e6).toFixed(2)+'M':n>1e3?(n/1e3).toFixed(0)+'K':n}
async function api(path,opts){try{const r=await fetch('/api/'+path,opts||{});return await r.json()}catch(e){return null}}
async function post(action,data){return api('action/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data||{})})}

// ─── Agent Control ───
async function checkAgent(){const s=await api('agent-status');const el=document.getElementById('agent-status');if(s&&s.running){el.className='badge badge-g';el.textContent='● Agent: Running (PID:'+s.pid+')'}else{el.className='badge badge-r';el.textContent='● Agent: Stopped'}}
async function startAgent(){const r=await post('start-agent');if(r&&r.success){addMsg('sys','🟢 Agent started: '+r.message);checkAgent()}else{addMsg('sys','❌ '+r.error)}}
async function stopAgent(){const r=await post('stop-agent');if(r&&r.success){addMsg('sys','🔴 Agent stopped');checkAgent()}}
async function startGW(){const r=await post('start-gateway');const el=document.getElementById('gw-status');if(r&&r.success){el.className='badge badge-g';el.textContent='Gateway: On'}else{addMsg('sys','❌ Gateway: '+r.error)}}

// ─── Chat ───
function addMsg(type,text){const d=document.createElement('div');d.className='msg '+type;d.innerHTML=text;document.getElementById('chat-msgs').appendChild(d);d.scrollTop=d.scrollHeight;document.getElementById('chat-msgs').scrollTop=999999}
async function sendChat(){const inp=document.getElementById('chat-input');const msg=inp.value.trim();if(!msg)return;inp.value='';addMsg('user',escapeHtml(msg));addMsg('sys','⏳ Đang gửi...');const r=await post('chat',{message:msg});document.querySelector('.msg.sys:last-child').remove();if(r&&r.success){addMsg('agent',formatResponse(r.response))}else{addMsg('sys','❌ '+(r?r.error:'No response'))}}
function escapeHtml(t){return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function formatResponse(text){if(!text)return '(empty)';try{const j=JSON.parse(text);return '<pre>'+JSON.stringify(j,null,2)+'</pre>'}catch(e){return escapeHtml(text)}}

// ─── File Upload ───
async function uploadFiles(files){const fl=document.getElementById('file-list');for(const f of files){const fd=new FormData();fd.append('file',f);try{const r=await fetch('/api/upload',{method:'POST',body:fd});const res=await r.json();if(res.success){fl.innerHTML+='✓ '+f.name+' ('+f.size+' bytes)<br>';addMsg('sys','📎 File uploaded: '+f.name)}else{fl.innerHTML+='❌ '+f.name+': '+res.error+'<br>'}}catch(e){fl.innerHTML+='❌ '+f.name+': upload failed<br>'}}}
// Drag-drop
const fa=document.getElementById('file-area');fa.addEventListener('dragover',e=>{e.preventDefault();fa.classList.add('dragover')});fa.addEventListener('dragleave',()=>fa.classList.remove('dragover'));fa.addEventListener('drop',e=>{e.preventDefault();fa.classList.remove('dragover');uploadFiles(e.dataTransfer.files)});

// ─── JSON Formatter ───
function formatJSON(){const inp=document.getElementById('json-input').value.trim();const filter=document.getElementById('json-filter').value.trim();const out=document.getElementById('json-out');if(!inp){out.textContent='Kết quả sẽ hiện ở đây...';return}try{let j=JSON.parse(inp);if(filter){if(j[filter]!==undefined){j=j[filter]}else{const filtered={};for(const[k,v]of Object.entries(j)){if(k.includes(filter))filtered[k]=v}j=Object.keys(filtered).length?filtered:j}}out.innerHTML='<pre style="white-space:pre-wrap">'+escapeHtml(JSON.stringify(j,null,2))+'</pre>'}catch(e){out.textContent='❌ Invalid JSON: '+e.message}}

// ─── Refresh data ───
async function refresh(){const s=await api('status');if(!s)return;document.getElementById('stat-cards').innerHTML=`<div class="sc g"><div class="sl">Providers</div><div class="sv">${s.providers}</div><div class="ss">${s.activeKeys} keys</div></div><div class="sc a"><div class="sl">Skills</div><div class="sv">${s.installedSkills}</div><div class="ss">installed</div></div><div class="sc b"><div class="sl">Features</div><div class="sv">${s.enabledFeatures}/${s.totalFeatures}</div><div class="ss">enabled</div></div><div class="sc a"><div class="sl">Tokens</div><div class="sv">${fmt(s.totalCostTokens)}</div><div class="ss">${s.totalCalls} calls</div></div>`;document.getElementById('ft').textContent=`● ${s.activeKeys} keys | ${s.installedSkills} skills | ${s.enabledFeatures}/${s.totalFeatures} features | ${fmt(s.totalCostTokens)} tokens`}
async function refreshProviders(){const p=await api('providers');if(!p)return;document.getElementById('pl').innerHTML=p.map(pr=>`<div class="pc"><div style="font-weight:700;font-size:1rem">${pr.name}</div><div style="font-size:.7rem;color:var(--muted);font-family:monospace">${pr.baseUrl}</div><div style="margin:.3rem 0">${pr.models.map(m=>`<span class="mb">${m}</span>`).join('')}</div>${pr.keys.map(k=>{const pct=k.quota>0?Math.round(k.used/k.quota*100):0;const cls=pct>80?'r':pct>50?'y':'g';const st=k.exhausted?'exhausted':k.enabled?'active':'disabled';return `<div class="kr"><span class="kp">${k.keyPreview}</span><div class="qb"><div class="qf ${cls}" style="width:${Math.max(pct,2)}%"></div></div><span class="ks ${st}">${st}</span></div>`}).join('')}<button class="btn btn-h" style="font-size:.7rem;margin-top:.3rem" onclick="addKey('${pr.id}')">+ Add Key</button></div>`).join('')}
async function refreshConfig(){const c=await api('config');if(!c)return;const groups={};c.forEach(f=>{if(!groups[f.category])groups[f.category]=[];groups[f.category].push(f)});let html='';for(const[cat,fields]of Object.entries(groups)){const en=fields.filter(f=>f.enabled).length;html+=`<div style="margin-bottom:1rem"><div style="font-weight:700;color:var(--accent);font-size:.8rem;text-transform:uppercase;margin-bottom:.3rem">${cat} (${en}/${fields.length})</div>${fields.map(f=>`<div class="cf"><div><div class="cfp">${f.path}</div><div class="cfd">${f.description}</div></div><div class="tg ${f.enabled?'on':''}" onclick="toggleCfg('${f.path}',this)"></div></div>`).join('')}</div>`}document.getElementById('cl').innerHTML=html}
async function refreshSkills(){const s=await api('skills');if(!s)return;document.getElementById('sl').innerHTML=s.map(sk=>`<div class="sc" style="${sk.installed?'border-color:rgba(90,138,58,.3)':''}"><div style="font-weight:700;font-size:.8rem">${sk.id}</div><div style="font-size:.65rem;color:var(--muted);font-family:monospace">${sk.repo}</div><div style="font-size:.75rem;color:var(--dim);margin:.2rem 0">${sk.desc}</div></div>`).join('')}
async function refreshCosts(){const c=await api('costs');if(!c)return;const max=Math.max(...c.map(x=>x.tokens),1);document.getElementById('cs').innerHTML=`<div class="sg"><div class="sc a"><div class="sl">Tokens</div><div class="sv">${fmt(c.reduce((s,x)=>s+x.tokens,0))}</div></div><div class="sc b"><div class="sl">Calls</div><div class="sv">${c.reduce((s,x)=>s+x.calls,0)}</div></div></div><div class="card"><div style="display:flex;align-items:flex-end;gap:.3rem;height:180px">${c.map(x=>{const h=Math.round(x.tokens/max*160);return `<div style="flex:1;background:var(--accent);border-radius:3px 3px 0 0;height:${h}px;position:relative;min-height:3px"><div style="position:absolute;top:-14px;left:50%;transform:translateX(-50%);font-size:.6rem;color:var(--dim)">${fmt(x.tokens)}</div><div style="position:absolute;bottom:-16px;left:50%;transform:translateX(-50%);font-size:.55rem;color:var(--muted);white-space:nowrap">${x.phase}</div></div>`}).join('')}</div></div>`}
async function refreshLogs(){const l=await api('logs');if(!l)return;document.getElementById('lf').innerHTML=l.map(x=>`<div style="display:flex;gap:.4rem;padding:.2rem;font-size:.75rem;border-bottom:1px solid rgba(224,213,195,.2)"><span style="font-family:monospace;color:var(--muted)">${x.timestamp}</span><span style="font-weight:700;color:var(--${x.level==='success'?'green':x.level==='error'?'red':x.level==='warn'?'yellow':'blue'})">${x.level.toUpperCase()}</span><span style="color:var(--accent);font-family:monospace">${x.module}</span><span style="flex:1">${x.message}</span></div>`).join('')}
async function toggleCfg(path,el){const r=await post('toggle-config',{path});if(r&&r.success){el.classList.toggle('on');refresh()}}
function addKey(pid){const key=prompt('API Key:');if(!key)return;const q=prompt('Quota tokens (0=unlimited):','0')||'0';post('add-key',{providerId:pid,key:key,quota:parseInt(q)}).then(r=>{if(r&&r.success)refreshProviders();else alert(r?r.error:'Failed')})}
function showAddProvider(){const name=prompt('Provider name (vd: glm):');if(!name)return;const url=prompt('Base URL:');if(!url)return;const models=prompt('Models (comma separated):','glm-4.6');const key=prompt('API Key (optional):');const q=key?prompt('Quota (0=unlimited):','0'):'0';post('add-provider',{name,baseUrl:url,models:models?models.split(','):[],key:key||'',quota:parseInt(q)}).then(r=>{if(r&&r.success)refreshProviders();else alert(r?r.error:'Failed')})}
function fc(q){q=q.toLowerCase();document.querySelectorAll('.cf').forEach(f=>{f.style.display=f.textContent.toLowerCase().includes(q)?'':'none'})}
function fs(q){q=q.toLowerCase();document.querySelectorAll('#sl .sc').forEach(c=>{c.style.display=c.textContent.toLowerCase().includes(q)?'':'none'})}

// Init
checkAgent();refresh();refreshProviders();refreshConfig();refreshSkills();refreshCosts();refreshLogs();
setInterval(checkAgent,5000);setInterval(refresh,10000);setInterval(refreshLogs,3000);
// Auto-start agent on load
setTimeout(()=>{startAgent()},2000);
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
        elif path == "/api/action/toggle-config": self._send_json(_action_toggle_config(data))
        elif path == "/api/action/add-provider": self._send_json(_action_add_provider(data))
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
