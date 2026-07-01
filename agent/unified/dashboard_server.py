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


def _action_set_mode(data: dict) -> dict:
    """Set thinking level + reasoning mode + verify on/off. Writes to config.yaml."""
    thinking = data.get("thinking", "balanced")
    reasoning = data.get("reasoning", "standard")
    verify = data.get("verify", "on")
    config_path = HERMES_HOME / "config.yaml"
    config = _read_yaml(config_path) or {}
    unified = config.setdefault("unified", {})
    # Thinking level
    st = unified.setdefault("slow_thinking", {})
    if thinking == "fast":
        st["enabled"] = False
    else:
        st["enabled"] = True
        st["default_level"] = thinking
    # Reasoning level
    sr = unified.setdefault("reasoning", {})
    if reasoning == "off":
        sr["enabled"] = False
    else:
        sr["enabled"] = True
        # "standard" = default, "high" = slow_thinking deep, "max" = slow_thinking max
        if reasoning == "high":
            st["enabled"] = True
            st["default_level"] = "deep"
        elif reasoning == "max":
            st["enabled"] = True
            st["default_level"] = "max"
    # Verify
    sv = unified.setdefault("verifier", {})
    sv["enabled"] = verify == "on"
    # Also toggle smart_guardian with verify
    sg = unified.setdefault("smart_guardian", {})
    sg["enabled"] = verify == "on"
    try:
        import yaml
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        return {"success": True, "message": f"Mode set: thinking={thinking}, reasoning={reasoning}, verify={verify}", "config": _get_config()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── HTML (full control center) ─────────────────────────────────────────────


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes-Omni</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#f5f0e8;--card:#faf6ef;--border:#e0d5c3;--text:#3d3528;--dim:#7a6f5c;--muted:#a89a82;--accent:#c8860d;--green:#5a8a3a;--red:#c44d4d;--blue:#4a7ba8;--sidebar:#ede5d6;--hover:#e5dcc9}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex}
.sidebar{width:56px;background:var(--sidebar);border-right:1px solid var(--border);display:flex;flex-direction:column;align-items:center;padding:.8rem 0;gap:.3rem;position:sticky;top:0;height:100vh}
.nav-btn{width:38px;height:38px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;cursor:pointer;border:none;background:none;color:var(--dim);transition:.15s}
.nav-btn:hover{background:var(--hover);color:var(--text)}.nav-btn.active{background:var(--accent);color:#fff}
.nav-spacer{flex:1}
.main{flex:1;display:flex;flex-direction:column;min-width:0;height:100vh}
.topbar{padding:.5rem .8rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:var(--card)}
.topbar-title{font-weight:700;font-size:.9rem}.topbar-actions{display:flex;gap:.3rem;align-items:center}
.badge{padding:.12rem .4rem;border-radius:9999px;font-size:.6rem;font-weight:600}
.badge-g{background:rgba(90,138,58,.12);color:var(--green)}.badge-r{background:rgba(196,77,77,.12);color:var(--red)}
.btn{padding:.35rem .7rem;border-radius:7px;font-size:.7rem;font-weight:600;cursor:pointer;border:none}
.btn-p{background:var(--accent);color:#fff}.btn-g{background:var(--green);color:#fff}.btn-r{background:var(--red);color:#fff}.btn-h{background:var(--card);color:var(--dim);border:1px solid var(--border)}
.view{display:none;flex:1;overflow-y:auto;padding:.8rem}.view.active{display:block}
.chat-view{display:none;flex:1;flex-direction:column}.chat-view.active{display:flex}
.chat-msgs{flex:1;overflow-y:auto;padding:.8rem;scroll-behavior:smooth}
.chat-msgs::-webkit-scrollbar{width:5px}.chat-msgs::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.load-more{text-align:center;padding:.4rem;color:var(--accent);font-size:.7rem;cursor:pointer;margin-bottom:.3rem}
.load-more:hover{text-decoration:underline}
.msg{max-width:72%;margin-bottom:.6rem;padding:.6rem .9rem;border-radius:11px;font-size:.82rem;line-height:1.5;word-break:break-word;white-space:pre-wrap}
.msg.user{background:var(--accent);color:#fff;margin-left:auto;border-bottom-right-radius:4px}
.msg.agent{background:var(--card);border:1px solid var(--border);margin-right:auto;border-bottom-left-radius:4px}
.msg.sys{background:transparent;color:var(--muted);font-size:.65rem;text-align:center;margin:0 auto;max-width:90%}
.msg pre{background:rgba(0,0,0,.05);padding:.4rem;border-radius:5px;font-size:.75rem;margin-top:.3rem;overflow-x:auto}
.chat-bottom{padding:.5rem .8rem;border-top:1px solid var(--border);background:var(--card)}
.chat-input-row{display:flex;gap:.4rem;align-items:flex-end}
.chat-input-row textarea{flex:1;padding:.4rem .6rem;border:1px solid var(--border);border-radius:9px;font-size:.82rem;background:var(--bg);color:var(--text);resize:none;font-family:inherit;max-height:100px}
.chat-input-row textarea:focus{outline:none;border-color:var(--accent)}
.file-area{padding:.2rem;border:1px dashed var(--border);border-radius:7px;text-align:center;font-size:.65rem;color:var(--muted);cursor:pointer;margin-bottom:.3rem}
.file-area:hover{border-color:var(--accent)}.file-list{font-size:.6rem;color:var(--dim);margin-top:.15rem}
/* Mode selector */
.mode-bar{display:flex;gap:.3rem;align-items:center;padding:.3rem 0;border-top:1px solid var(--border);margin-top:.3rem;flex-wrap:wrap}
.mode-group{display:flex;align-items:center;gap:.2rem}
.mode-label{font-size:.6rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.mode-btn{padding:.2rem .5rem;border-radius:5px;font-size:.65rem;font-weight:600;cursor:pointer;border:1px solid var(--border);background:var(--card);color:var(--dim);transition:.15s}
.mode-btn:hover{border-color:var(--accent);color:var(--text)}
.mode-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.mode-divider{width:1px;height:16px;background:var(--border);margin:0 .2rem}
.mode-info{font-size:.55rem;color:var(--muted);margin-left:auto}
.sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.6rem;margin-bottom:.8rem}
.sc{background:var(--card);border:1px solid var(--border);border-radius:9px;padding:.8rem}
.sl{font-size:.6rem;text-transform:uppercase;color:var(--muted);font-weight:600}.sv{font-size:1.3rem;font-weight:800;margin:.15rem 0}.ss{font-size:.65rem;color:var(--dim)}
.card{background:var(--card);border:1px solid var(--border);border-radius:9px;padding:.8rem;margin-bottom:.6rem}
.pc{background:var(--card);border:1px solid var(--border);border-radius:9px;padding:.8rem;margin-bottom:.5rem}
.kr{display:flex;align-items:center;gap:.3rem;padding:.25rem;border-radius:5px;background:var(--bg);margin-bottom:.15rem}
.kp{font-family:monospace;font-size:.65rem;color:var(--dim);min-width:90px}
.qb{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.qf{height:100%;border-radius:2px}.qf.g{background:var(--green)}.qf.y{background:#d4a017}.qf.r{background:var(--red)}
.ks{font-size:.55rem;padding:.06rem .25rem;border-radius:3px;font-weight:600}
.ks.active{background:rgba(90,138,58,.12);color:var(--green)}.ks.exhausted{background:rgba(196,77,77,.12);color:var(--red)}.ks.disabled{background:rgba(153,153,153,.12);color:var(--muted)}
.mb{display:inline-block;background:rgba(200,134,13,.08);color:var(--accent);padding:.06rem .3rem;border-radius:3px;font-size:.55rem;font-family:monospace;margin-right:.1rem}
.cf{display:flex;align-items:center;justify-content:space-between;padding:.35rem .5rem;border-radius:5px;background:var(--card);margin-bottom:.1rem}
.cf:hover{background:var(--hover)}.cfp{font-family:monospace;font-size:.6rem;color:var(--muted)}.cfd{font-size:.68rem;color:var(--dim)}
.tg{width:32px;height:16px;background:var(--border);border-radius:8px;cursor:pointer;position:relative;transition:.2s;flex-shrink:0}
.tg.on{background:var(--green)}.tg::after{content:'';position:absolute;top:2px;left:2px;width:12px;height:12px;background:#fff;border-radius:50%;transition:.2s}.tg.on::after{left:18px}
.sk-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:.5rem}
.skc{background:var(--card);border:1px solid var(--border);border-radius:7px;padding:.7rem}
.skn{font-weight:700;font-size:.72rem}.skr{font-size:.55rem;color:var(--muted);font-family:monospace}.skde{font-size:.68rem;color:var(--dim);margin:.15rem 0}
.json-in{width:100%;height:70px;padding:.4rem;border:1px solid var(--border);border-radius:7px;font-family:monospace;font-size:.7rem;background:var(--card);color:var(--text);resize:vertical}
.json-out{padding:.4rem;border:1px solid var(--border);border-radius:7px;background:var(--card);font-family:monospace;font-size:.7rem;white-space:pre-wrap;word-break:break-word;max-height:200px;overflow-y:auto;margin-top:.3rem}
.search{width:100%;padding:.35rem .5rem;border:1px solid var(--border);border-radius:7px;font-size:.75rem;background:var(--card);color:var(--text);margin-bottom:.4rem}
.search:focus{outline:none;border-color:var(--accent)}
@media(max-width:768px){body{flex-direction:column}.sidebar{width:100%;height:auto;flex-direction:row;position:sticky;top:0;border-right:none;border-bottom:1px solid var(--border);padding:.4rem;gap:.2rem}.nav-spacer{display:none}.main{height:calc(100vh - 52px)}}
</style></head><body>
<nav class="sidebar">
<button class="nav-btn active" onclick="nv(event,'chat')" title="Chat">💬</button>
<button class="nav-btn" onclick="nv(event,'overview')" title="Tổng quan">📊</button>
<button class="nav-btn" onclick="nv(event,'providers')" title="Provider">🔌</button>
<button class="nav-btn" onclick="nv(event,'config')" title="Cấu hình">⚙️</button>
<button class="nav-btn" onclick="nv(event,'skills')" title="Kỹ năng">📚</button>
<button class="nav-btn" onclick="nv(event,'costs')" title="Chi phí">💰</button>
<button class="nav-btn" onclick="nv(event,'logs')" title="Nhật ký">📋</button>
<button class="nav-btn" onclick="nv(event,'json')" title="JSON">🔧</button>
<div class="nav-spacer"></div>
<button class="nav-btn" onclick="startAgent()" title="Start" style="color:var(--green)">▶</button>
<button class="nav-btn" onclick="stopAgent()" title="Stop" style="color:var(--red)">⏹</button>
</nav>
<div class="main">
<div class="topbar"><div class="topbar-title" id="vt">Chat</div><div class="topbar-actions">
<span class="badge badge-g" id="as">● Agent</span><span class="badge badge-r" id="gs">GW: Off</span>
<button class="btn btn-h" onclick="startGW()" style="font-size:.6rem">Start GW</button></div></div>
<div id="chat" class="chat-view active"><div class="chat-msgs" id="cm"><div class="msg sys">Chat với Hermes. Agent tự khởi động.</div></div>
<div class="chat-bottom"><div class="file-area" id="fa" onclick="document.getElementById('fi').click()">📎 Kéo thả hoặc click<input type="file" id="fi" multiple style="display:none" onchange="upF(this.files)"></div><div class="file-list" id="fl"></div>
<div class="mode-bar">
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
<span class="mode-info" id="mode-info">balanced · std · verify</span>
</div>
<div class="chat-input-row" style="margin-top:.2rem"><textarea id="ci" placeholder="Nhập tin nhắn..." rows="1" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}" oninput="this.style.height='auto';this.style.height=Math.min(this.scrollHeight,100)+'px'"></textarea><button class="btn btn-p" onclick="send()">Gửi</button></div></div></div>
<div id="overview" class="view"><div class="sg" id="sc"></div></div>
<div id="providers" class="view"><div id="pl"></div><button class="btn btn-p" onclick="addP()">+ Provider</button></div>
<div id="config" class="view"><input class="search" placeholder="🔍 Tìm..." onkeyup="fc(this.value)"><div id="cl"></div></div>
<div id="skills" class="view"><input class="search" placeholder="🔍 Tìm..." onkeyup="fs(this.value)"><div id="sl" class="sk-grid"></div></div>
<div id="costs" class="view"><div id="cs"></div></div>
<div id="logs" class="view"><div class="card" style="max-height:400px;overflow:auto" id="lf"></div></div>
<div id="json" class="view"><div class="card"><b>JSON Formatter</b><br><textarea class="json-in" id="ji" placeholder="Paste JSON..." oninput="fj()"></textarea><input class="search" style="margin-top:.2rem" id="jf" placeholder="Filter key..." oninput="fj()"><div class="json-out" id="jo">Kết quả...</div></div></div>
</div></div>
<script>
const T={chat:'Chat',overview:'Tổng quan',providers:'Provider',config:'Cấu hình',skills:'Kỹ năng',costs:'Chi phí',logs:'Nhật ký',json:'JSON Tool'};
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
async function startGW(){const r=await post('start-gateway');const e=document.getElementById('gs');if(r&&r.success){e.className='badge badge-g';e.textContent='GW: On'}else addM('sys','❌ '+(r?r.error:'Fail'))}
async function ref(){const s=await api('status');if(!s)return;document.getElementById('sc').innerHTML='<div class="sc"><div class="sl">Providers</div><div class="sv">'+s.providers+'</div><div class="ss">'+s.activeKeys+' keys</div></div><div class="sc"><div class="sl">Skills</div><div class="sv">'+s.installedSkills+'</div><div class="ss">installed</div></div><div class="sc"><div class="sl">Features</div><div class="sv">'+s.enabledFeatures+'/'+s.totalFeatures+'</div><div class="ss">enabled</div></div><div class="sc"><div class="sl">Tokens</div><div class="sv">'+fn(s.totalCostTokens)+'</div><div class="ss">'+s.totalCalls+' calls</div></div>'}
async function rP(){const p=await api('providers');if(!p)return;document.getElementById('pl').innerHTML=p.map(pr=>'<div class="pc"><div style="font-weight:700;font-size:.85rem">'+pr.name+'</div><div style="font-size:.6rem;color:var(--muted);font-family:monospace">'+pr.baseUrl+'</div><div style="margin:.15rem 0">'+pr.models.map(m=>'<span class="mb">'+m+'</span>').join('')+'</div>'+pr.keys.map(k=>{const pct=k.quota>0?Math.round(k.used/k.quota*100):0;const c=pct>80?'r':pct>50?'y':'g';const st=k.exhausted?'exhausted':k.enabled?'active':'disabled';return '<div class="kr"><span class="kp">'+k.keyPreview+'</span><div class="qb"><div class="qf '+c+'" style="width:'+Math.max(pct,2)+'%"></div></div><span class="ks '+st+'">'+st+'</span></div>'}).join('')+'<button class="btn btn-h" style="font-size:.6rem;margin-top:.15rem" onclick="addK(\''+pr.id+'\')">+ Add Key</button></div>').join('')}
async function rC(){const c=await api('config');if(!c)return;const g={};c.forEach(f=>{if(!g[f.category])g[f.category]=[];g[f.category].push(f)});let h='';for(const[cat,fs]of Object.entries(g)){const en=fs.filter(f=>f.enabled).length;h+='<div style="margin-bottom:.6rem"><div style="font-weight:700;color:var(--accent);font-size:.7rem;text-transform:uppercase;margin-bottom:.15rem">'+cat+' ('+en+'/'+fs.length+')</div>'+fs.map(f=>'<div class="cf"><div><div class="cfp">'+f.path+'</div><div class="cfd">'+f.description+'</div></div><div class="tg '+(f.enabled?'on':'')+'" onclick="tc(\''+f.path+'\',this)"></div></div>').join('')+'</div>'}document.getElementById('cl').innerHTML=h}
async function rS(){const s=await api('skills');if(!s)return;document.getElementById('sl').innerHTML=s.map(sk=>'<div class="skc"><div class="skn">'+sk.id+'</div><div class="skr">'+sk.repo+'</div><div class="skde">'+sk.desc+'</div></div>').join('')}
async function rCost(){const c=await api('costs');if(!c)return;const max=Math.max(...c.map(x=>x.tokens),1);document.getElementById('cs').innerHTML='<div class="sg"><div class="sc"><div class="sl">Tokens</div><div class="sv">'+fn(c.reduce((s,x)=>s+x.tokens,0))+'</div></div><div class="sc"><div class="sl">Calls</div><div class="sv">'+c.reduce((s,x)=>s+x.calls,0)+'</div></div></div><div class="card"><div style="display:flex;align-items:flex-end;gap:.2rem;height:140px">'+c.map(x=>{const h=Math.round(x.tokens/max*120);return '<div style="flex:1;background:var(--accent);border-radius:3px 3px 0 0;height:'+h+'px;min-height:2px;position:relative"><div style="position:absolute;top:-10px;left:50%;transform:translateX(-50%);font-size:.5rem;color:var(--dim)">'+fn(x.tokens)+'</div><div style="position:absolute;bottom:-12px;left:50%;transform:translateX(-50%);font-size:.45rem;color:var(--muted);white-space:nowrap">'+x.phase+'</div></div>'}).join('')+'</div></div>'}
async function rL(){const l=await api('logs');if(!l)return;document.getElementById('lf').innerHTML=l.map(x=>'<div style="display:flex;gap:.2rem;padding:.1rem;font-size:.65rem;border-bottom:1px solid var(--border)"><span style="font-family:monospace;color:var(--muted)">'+x.timestamp+'</span><span style="font-weight:700;color:var(--'+(x.level==='success'?'green':x.level==='error'?'red':x.level==='warn'?'#d4a017':'blue')+'")">'+x.level.toUpperCase()+'</span><span style="color:var(--accent);font-family:monospace">'+x.module+'</span><span style="flex:1">'+x.message+'</span></div>').join('')}
async function tc(p,el){const r=await post('toggle-config',{path:p});if(r&&r.success){el.classList.toggle('on');ref()}}
function addK(pid){const k=prompt('Key:');if(!k)return;const q=prompt('Quota (0=∞):','0')||'0';post('add-key',{providerId:pid,key:k,quota:parseInt(q)}).then(r=>{if(r&&r.success)rP();else alert(r?r.error:'Fail')})}
function addP(){const n=prompt('Name:');if(!n)return;const u=prompt('URL:');if(!u)return;const m=prompt('Models:');const k=prompt('Key:');post('add-provider',{name:n,baseUrl:u,models:m?m.split(','):[],key:k||'',quota:0}).then(r=>{if(r&&r.success)rP();else alert(r?r.error:'Fail')})}
function fc(q){q=q.toLowerCase();document.querySelectorAll('.cf').forEach(f=>f.style.display=f.textContent.toLowerCase().includes(q)?'':'none')}
function fs(q){q=q.toLowerCase();document.querySelectorAll('.skc').forEach(c=>c.style.display=c.textContent.toLowerCase().includes(q)?'':'none')}
ca();ref();rP();rC();rS();rCost();rL();
setInterval(ca,5000);setInterval(ref,10000);setInterval(rL,3000);
setTimeout(startAgent,2000);
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
        elif path == "/api/action/set-mode": self._send_json(_action_set_mode(data))
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
