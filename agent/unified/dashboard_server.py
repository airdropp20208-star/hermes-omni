"""Dashboard Server v5 — Complete rewrite.

Architecture:
- HTTP server (ThreadingHTTPServer) on port 8788
- Chat via subprocess `hermes -z` (isolated, won't crash server)
- All tabs working: Chat, Overview, Providers, Config, Skills, Costs, Logs
- Provider setup writes config.yaml + .env (same format as `hermes model`)
- Mode panel: thinking / reasoning / verify → writes to config.yaml
- File upload to ~/.hermes/uploads/
- Auto-installs missing deps on startup (yaml, openai, etc.)
- Graceful fallback if any unified module fails to import

Usage:
    python -m agent.unified.dashboard_server --port 8788
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ─── Auto-install missing deps ──────────────────────────────────────────────

def _ensure_dep(import_name: str, pip_name: str = None) -> bool:
    """Try import; if fail, pip install. Returns True if available."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        pip_name = pip_name or import_name
        print(f"[setup] installing {pip_name}...", flush=True)
        import subprocess
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "--user", pip_name],
                timeout=60, check=False,
            )
            __import__(import_name)
            print(f"[setup] ✓ {pip_name} installed", flush=True)
            return True
        except Exception as e:
            print(f"[setup] ✗ {pip_name} failed: {e}", flush=True)
            return False

# Critical deps for dashboard
_ensure_dep("yaml", "pyyaml")
_ensure_dep("openai")
_ensure_dep("httpx")

# ─── Paths ───────────────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = HERMES_HOME / "uploads"
UNIFIED_DIR = HERMES_HOME / "unified"

# Auto-create HERMES_HOME structure
for _d in [HERMES_HOME, UPLOAD_DIR, UNIFIED_DIR, HERMES_HOME / "logs", HERMES_HOME / "memories"]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Bootstrap default config if missing ────────────────────────────────────

def _bootstrap_config():
    """Create default config.yaml + .env if they don't exist, so dashboard
    works on a fresh install without `hermes setup`."""
    cfg_path = HERMES_HOME / "config.yaml"
    if not cfg_path.exists():
        default_cfg = {
            "model": {
                "provider": "xiaomi",
                "default": "mimo-v2.5",
                "base_url": "https://api.xiaomimimo.com/v1",
            },
            "unified": {
                "reasoning": {"enabled": True},
                "reflexion": {"enabled": True},
                "smart_guardian": {"enabled": True},
                "verifier": {"enabled": True},
                "constitution": {"enabled": True},
                "slow_thinking": {"enabled": True, "default_level": "balanced"},
                "ensemble": {"enabled": False},
                "embedding": {"enabled": False},
                "user_model": {"enabled": True},
                "clarifier": {"enabled": True},
                "longrun": {"enabled": False},
                "tool_router": {"enabled": True},
                "cost_tracker": {"enabled": True},
                "response_cache": {"enabled": True},
                "output_formatter": {"enabled": True},
                "skill_registry": {"enabled": True},
                "api_registry": {"enabled": True},
                "multi_provider": {"enabled": False},
                "learning": {"enabled": True},
                "skill_synthesis": {"enabled": True},
                "task_planner": {"enabled": True},
            },
        }
        try:
            import yaml
            cfg_path.write_text(
                yaml.dump(default_cfg, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            print(f"[setup] ✓ created default {cfg_path}", flush=True)
        except Exception as e:
            print(f"[setup] ✗ failed to write default config: {e}", flush=True)

    env_path = HERMES_HOME / ".env"
    if not env_path.exists():
        default_env = """# Hermes-Omni environment
# Set your API key here, e.g.:
# XIAOMI_API_KEY=sk-...
# ZAI_API_KEY=...
HERMES_YOLO_MODE=1
HERMES_ACCEPT_HOOKS=1
"""
        try:
            env_path.write_text(default_env, encoding="utf-8")
            print(f"[setup] ✓ created default {env_path}", flush=True)
        except Exception:
            pass

    mp_path = HERMES_HOME / "multi_provider.yaml"
    if not mp_path.exists():
        try:
            import yaml
            mp_path.write_text(
                yaml.dump({"providers": {}, "strategy": "round-robin"}, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
        except Exception:
            pass

_bootstrap_config()

# ─── Small helpers ───────────────────────────────────────────────────────────

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _write_yaml(path: Path, data: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _read_jsonl(path: Path, limit: int = 1000) -> list:
    if not path.exists():
        return []
    out = []
    try:
        for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


# ─── Provider maps ──────────────────────────────────────────────────────────

PROVIDER_ENV_KEYS = {
    "zai": "ZAI_API_KEY",
    "xiaomi": "XIAOMI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "custom": "CUSTOM_API_KEY",
}

PROVIDER_BASE_URLS = {
    "zai": "https://open.bigmodel.cn/api/paas/v4",
    "xiaomi": "https://api.xiaomimimo.com/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
}

PROVIDER_DEFAULT_MODELS = {
    "zai": "glm-4.6",
    "xiaomi": "mimo-v2.5",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "openrouter": "openai/gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "groq": "llama-3.1-70b-versatile",
    "together": "meta-llama/Llama-3-70b-chat-hf",
    "mistral": "mistral-large-latest",
}

PROVIDER_LABELS = {
    "zai": "GLM (z.ai) — miễn phí",
    "xiaomi": "Xiaomi MiMo",
    "openai": "OpenAI (GPT)",
    "anthropic": "Anthropic (Claude)",
    "openrouter": "OpenRouter (đa model)",
    "deepseek": "DeepSeek",
    "groq": "Groq (siêu nhanh)",
    "together": "Together AI",
    "mistral": "Mistral",
    "custom": "Tùy chỉnh",
}

# ─── Status / config readers ────────────────────────────────────────────────

def _scan_skills() -> list:
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
                content = skill_md.read_text(encoding="utf-8", errors="ignore")[:600]
                desc = ""
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end > 0:
                        for line in content[3:end].splitlines():
                            if line.strip().startswith("description:"):
                                desc = line.split(":", 1)[1].strip().strip('"').strip("'")[:140]
                                break
                if not desc:
                    desc = f"Skill from {repo_name}/{skill_name}"
                skills.append({
                    "id": f"{repo_name}/{skill_name}",
                    "repo": repo_name,
                    "name": skill_name,
                    "desc": desc,
                    "installed": True,
                })
            except Exception:
                continue
    return skills


def _get_current_config() -> dict:
    config = _read_yaml(HERMES_HOME / "config.yaml")
    model_cfg = config.get("model", {}) if config else {}
    env_path = HERMES_HOME / ".env"
    api_key = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or not line or "=" not in line:
                continue
            if "_API_KEY=" in line:
                v = line.split("=", 1)[1].strip()
                if v and not v.startswith("your-"):
                    api_key = v
                    break
    return {
        "provider": model_cfg.get("provider", ""),
        "model": model_cfg.get("default", ""),
        "base_url": model_cfg.get("base_url", ""),
        "has_key": bool(api_key),
        "key_preview": (api_key[:6] + "..." + api_key[-4:]) if len(api_key) > 12 else ("***" if api_key else ""),
    }


def _get_status() -> dict:
    config = _read_yaml(HERMES_HOME / "config.yaml")
    unified = config.get("unified", {}) if config else {}
    feature_keys = [
        "reasoning", "reflexion", "smart_guardian", "verifier", "constitution",
        "slow_thinking", "ensemble", "embedding", "user_model", "clarifier",
        "longrun", "tool_router", "cost_tracker", "response_cache",
        "output_formatter", "skill_registry", "api_registry", "multi_provider",
        "learning", "skill_synthesis", "task_planner",
    ]
    enabled = sum(1 for k in feature_keys if unified.get(k, {}).get("enabled", False))
    skills = _scan_skills()
    costs = _read_jsonl(UNIFIED_DIR / "cost_log.jsonl", limit=2000)
    total_tokens = sum(int(c.get("total_tokens", 0)) for c in costs)
    mp = _read_yaml(HERMES_HOME / "multi_provider.yaml")
    providers = mp.get("providers", {}) if mp else {}
    total_keys = sum(len(p.get("keys", [])) for p in providers.values())
    cur = _get_current_config()
    return {
        "providers": len(providers),
        "totalKeys": total_keys,
        "installedSkills": len(skills),
        "enabledFeatures": enabled,
        "totalFeatures": len(feature_keys),
        "totalCostTokens": total_tokens,
        "totalCalls": len(costs),
        "currentProvider": cur["provider"],
        "currentModel": cur["model"],
        "hasKey": cur["has_key"],
    }


def _get_providers() -> list:
    config = _read_yaml(HERMES_HOME / "multi_provider.yaml")
    if not config:
        return []
    out = []
    for name, pdata in (config.get("providers") or {}).items():
        keys = []
        for k in (pdata.get("keys") or []):
            if isinstance(k, dict):
                ks = k.get("key", "")
                quota = int(k.get("quota_tokens", 0) or 0)
                preview = ks[:6] + "..." + ks[-4:] if len(ks) > 12 else "***"
                keys.append({
                    "id": f"k{len(keys)+1}",
                    "keyPreview": preview,
                    "quota": quota,
                    "used": 0,
                    "remaining": quota or 999999999,
                    "exhausted": False,
                    "enabled": k.get("enabled", True),
                })
            elif isinstance(k, str):
                preview = k[:6] + "..." + k[-4:] if len(k) > 12 else "***"
                keys.append({
                    "id": f"k{len(keys)+1}",
                    "keyPreview": preview,
                    "quota": 0,
                    "used": 0,
                    "remaining": 999999999,
                    "exhausted": False,
                    "enabled": True,
                })
        out.append({
            "id": name,
            "name": name,
            "baseUrl": pdata.get("base_url", ""),
            "enabled": pdata.get("enabled", True),
            "models": pdata.get("models") or [],
            "keys": keys,
        })
    return out


def _get_config() -> list:
    config = _read_yaml(HERMES_HOME / "config.yaml")
    unified = config.get("unified", {}) if config else {}
    fields = [
        ("reasoning", "unified.reasoning.enabled", "Nền tảng", "Lập kế hoạch → đánh giá → thực hiện → rút kinh nghiệm"),
        ("reflexion", "unified.reflexion.enabled", "Nền tảng", "Học từ lỗi sai"),
        ("smart_guardian", "unified.smart_guardian.enabled", "Bảo vệ", "Bảo vệ thông minh (LLM đánh giá)"),
        ("verifier", "unified.verifier.enabled", "Bảo vệ", "Tự kiểm tra trước khi gửi"),
        ("constitution", "unified.constitution.enabled", "Bảo vệ", "Nguyên tắc đạo đức"),
        ("slow_thinking", "unified.slow_thinking.enabled", "Suy luận sâu", "4 mức: nhanh/cân bằng/sâu/tối đa"),
        ("ensemble", "unified.ensemble.enabled", "Nâng cao", "Nhiều mô hình + giám khảo (3x token)"),
        ("embedding", "unified.embedding.enabled", "Nâng cao", "Ghi nhớ thông minh (+40%)"),
        ("user_model", "unified.user_model.enabled", "Nâng cao", "Cá nhân hóa theo người dùng"),
        ("clarifier", "unified.clarifier.enabled", "Nâng cao", "Phát hiện mơ hồ → hỏi lại"),
        ("longrun", "unified.longrun.enabled", "Hạ tầng", "Chạy tác vụ nền"),
        ("tool_router", "unified.tool_router.enabled", "Hạ tầng", "Tự chọn công cụ"),
        ("cost_tracker", "unified.cost_tracker.enabled", "Hạ tầng", "Đếm token + ngân sách"),
        ("response_cache", "unified.response_cache.enabled", "Hạ tầng", "Lưu cache (tiết kiệm token)"),
        ("output_formatter", "unified.output_formatter.enabled", "Hạ tầng", "Định dạng Telegram/Slack"),
        ("skill_registry", "unified.skill_registry.enabled", "Thư viện", "Thư viện kỹ năng"),
        ("api_registry", "unified.api_registry.enabled", "Thư viện", "1500+ API công khai"),
        ("multi_provider", "unified.multi_provider.enabled", "Đa nhà cung cấp", "Gộp nhiều API key"),
        ("learning", "unified.learning.enabled", "Học tập", "Học từ mọi tương tác"),
        ("skill_synthesis", "unified.skill_synthesis.enabled", "Học tập", "Tự tạo kỹ năng"),
        ("task_planner", "unified.task_planner.enabled", "Lập kế hoạch", "Chia task + theo dõi"),
    ]
    out = []
    for key, path, cat, desc in fields:
        section = unified.get(key, {})
        enabled = section.get("enabled", False) if isinstance(section, dict) else False
        out.append({
            "key": key, "path": path, "enabled": enabled,
            "category": cat, "description": desc,
        })
    return out


def _get_costs() -> list:
    costs = _read_jsonl(UNIFIED_DIR / "cost_log.jsonl", limit=2000)
    by_phase = {}
    for c in costs:
        phase = c.get("phase", "unknown")
        if phase not in by_phase:
            by_phase[phase] = {"phase": phase, "tokens": 0, "calls": 0}
        by_phase[phase]["tokens"] += int(c.get("total_tokens", 0))
        by_phase[phase]["calls"] += 1
    return list(by_phase.values())


def _get_logs() -> list:
    log_path = HERMES_HOME / "agent.log"
    logs = []
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in reversed(lines[-300:]):
                if len(logs) >= 60:
                    break
                level = "info"
                low = line.lower()
                if "error" in low or "traceback" in low:
                    level = "error"
                elif "warn" in low:
                    level = "warn"
                elif "✓" in line or "success" in low:
                    level = "success"
                ts = line[:8] if len(line) > 8 and line[2:3] == ":" and line[5:6] == ":" else ""
                msg = line[9:] if ts else line
                logs.append({
                    "timestamp": ts or time.strftime("%H:%M:%S"),
                    "level": level, "module": "agent", "message": msg[:240],
                })
        except Exception:
            pass
    # Fallback: use cost log as activity
    if not logs:
        costs = _read_jsonl(UNIFIED_DIR / "cost_log.jsonl", limit=30)
        for c in reversed(costs):
            ts = c.get("timestamp")
            try:
                tstr = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else time.strftime("%H:%M:%S")
            except Exception:
                tstr = time.strftime("%H:%M:%S")
            logs.append({
                "timestamp": tstr,
                "level": "success" if not c.get("cache_hit") else "info",
                "module": c.get("phase", "?"),
                "message": f"{c.get('phase', '?')} — {c.get('total_tokens', 0)} tokens" + (" (cached)" if c.get("cache_hit") else ""),
            })
    return logs


# ─── Chat (real agent via subprocess for isolation) ─────────────────────────

_chat_lock = threading.Lock()


def _send_to_agent(message: str, timeout: int = 90) -> dict:
    """Run one-shot through the real Hermes agent via subprocess.

    Using subprocess (not in-process) so:
    - Each chat is isolated — a hang/crash doesn't take down the server
    - We can enforce a hard timeout
    - No global state leakage between chats
    """
    if not message or not message.strip():
        return {"success": False, "error": "Tin nhắn trống"}

    with _chat_lock:
        import subprocess

        # Make sure env is set
        env = dict(os.environ)
        env["HERMES_YOLO_MODE"] = "1"
        env["HERMES_ACCEPT_HOOKS"] = "1"
        env["HERMES_HOME"] = str(HERMES_HOME)

        # Load .env into env
        env_path = HERMES_HOME / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()

        # Log start
        start = time.time()
        print(f"[chat] start: {message[:80]!r}", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "hermes_cli.main", "-z", message, "--yolo"],
                capture_output=True, text=True,
                timeout=timeout,
                cwd=str(REPO_ROOT),
                env=env,
            )
            elapsed = time.time() - start
            response = (proc.stdout or "").strip()
            err_text = (proc.stderr or "").strip()
            print(f"[chat] done in {elapsed:.1f}s, rc={proc.returncode}, out={len(response)}B, err={len(err_text)}B", flush=True)
            if proc.returncode == 0 and response:
                # Filter out known non-response lines
                lines = [l for l in response.splitlines() if l.strip()
                         and not l.startswith("[Note:")
                         and not l.startswith("Calling tool")
                         and not l.startswith("Tool output")]
                cleaned = "\n".join(lines).strip() if lines else response
                return {"success": True, "response": cleaned, "elapsed": round(elapsed, 1)}
            if response:
                return {"success": True, "response": response, "warning": err_text[:200] if err_text else None, "elapsed": round(elapsed, 1)}
            return {
                "success": False,
                "error": (err_text[:300] if err_text else f"Agent exit code {proc.returncode}"),
            }
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"[chat] TIMEOUT after {elapsed:.1f}s", flush=True)
            return {"success": False, "error": f"Agent timeout sau {int(elapsed)}s — thử lại với câu ngắn hơn hoặc tắt tools (mode Fast)."}
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[chat] ERROR: {type(exc).__name__}: {exc}", flush=True)
            return {"success": False, "error": f"{type(exc).__name__}: {exc}", "traceback": tb[-800:]}


# ─── Actions ────────────────────────────────────────────────────────────────

def _action_setup_provider(data: dict) -> dict:
    provider = (data.get("provider") or "").strip().lower()
    api_key = (data.get("apiKey") or "").strip()
    model = (data.get("model") or "").strip()
    base_url = (data.get("baseUrl") or "").strip()
    if not provider or not api_key:
        return {"success": False, "error": "Vui lòng chọn nhà cung cấp và nhập API key"}

    if not base_url and provider in PROVIDER_BASE_URLS:
        base_url = PROVIDER_BASE_URLS[provider]
    if not model and provider in PROVIDER_DEFAULT_MODELS:
        model = PROVIDER_DEFAULT_MODELS[provider]

    config_path = HERMES_HOME / "config.yaml"
    config = _read_yaml(config_path) or {}
    model_cfg = config.setdefault("model", {})
    model_cfg["provider"] = provider
    if model:
        model_cfg["default"] = model
    if base_url:
        model_cfg["base_url"] = base_url
    _write_yaml(config_path, config)

    # .env
    env_path = HERMES_HOME / ".env"
    env_lines = []
    if env_path.exists():
        env_lines = env_path.read_text(encoding="utf-8").splitlines()
    key_var = PROVIDER_ENV_KEYS.get(provider, f"{provider.upper().replace('-', '_')}_API_KEY")
    env_lines = [l for l in env_lines if not l.startswith(f"{key_var}=")]
    env_lines.append(f"{key_var}={api_key}")
    env_lines = [l for l in env_lines if not l.startswith("HERMES_INFERENCE_PROVIDER=")]
    env_lines.append(f"HERMES_INFERENCE_PROVIDER={provider}")
    env_lines = [l for l in env_lines if not l.startswith("HERMES_INFERENCE_MODEL=")]
    if model:
        env_lines.append(f"HERMES_INFERENCE_MODEL={model}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    # Also push to env so chat works without server restart
    os.environ[key_var] = api_key
    os.environ["HERMES_INFERENCE_PROVIDER"] = provider
    if model:
        os.environ["HERMES_INFERENCE_MODEL"] = model

    return {
        "success": True,
        "message": f"Đã lưu! Provider: {provider}, Model: {model or 'mặc định'}, Key: {api_key[:6]}...{api_key[-4:]}",
        "config": _get_current_config(),
    }


def _action_test_key(data: dict) -> dict:
    provider = (data.get("provider") or "").strip().lower()
    api_key = (data.get("apiKey") or "").strip()
    base_url = (data.get("baseUrl") or "").strip()
    model = (data.get("model") or "").strip()
    if not api_key or not base_url:
        return {"success": False, "error": "Cần API key + base URL"}
    if not model and provider in PROVIDER_DEFAULT_MODELS:
        model = PROVIDER_DEFAULT_MODELS[provider]
    try:
        from urllib.request import Request, urlopen
        url = f"{base_url.rstrip('/')}/chat/completions"
        body = json.dumps({
            "model": model or "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Reply with just: OK"}],
            "max_tokens": 10,
        }).encode("utf-8")
        req = Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }, method="POST")
        with urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        content = ""
        try:
            content = result["choices"][0]["message"]["content"]
        except Exception:
            pass
        return {
            "success": True,
            "message": f"Key hoạt động! Model: {result.get('model', '?')}, Phản hồi: {content[:80]}",
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def _action_toggle_config(data: dict) -> dict:
    path = (data.get("path") or "").strip()
    if not path:
        return {"success": False, "error": "path required"}
    config_path = HERMES_HOME / "config.yaml"
    config = _read_yaml(config_path) or {}
    parts = path.split(".")
    if len(parts) < 3:
        return {"success": False, "error": "Invalid path"}
    cur = config
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = not bool(cur.get(parts[-1], False))
    _write_yaml(config_path, config)
    return {"success": True, "config": _get_config()}


def _action_set_mode(data: dict) -> dict:
    thinking = (data.get("thinking") or "balanced").strip()
    reasoning = (data.get("reasoning") or "standard").strip()
    verify = (data.get("verify") or "on").strip()
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
        if reasoning in ("high", "max"):
            st["enabled"] = True
            st["default_level"] = "deep" if reasoning == "high" else "max"
    sv = unified.setdefault("verifier", {})
    sv["enabled"] = verify == "on"
    sg = unified.setdefault("smart_guardian", {})
    sg["enabled"] = verify == "on"
    _write_yaml(config_path, config)
    return {
        "success": True,
        "message": f"Mode: thinking={thinking}, reasoning={reasoning}, verify={verify}",
        "config": _get_config(),
    }


def _action_add_key(data: dict) -> dict:
    provider_id = (data.get("providerId") or "").strip()
    key = (data.get("key") or "").strip()
    quota = int(data.get("quota", 0) or 0)
    if not provider_id or not key:
        return {"success": False, "error": "Thiếu providerId/key"}
    config_path = HERMES_HOME / "multi_provider.yaml"
    config = _read_yaml(config_path) or {"providers": {}, "strategy": "round-robin"}
    providers = config.setdefault("providers", {})
    if provider_id not in providers:
        return {"success": False, "error": f"Provider '{provider_id}' chưa có"}
    keys = providers[provider_id].setdefault("keys", [])
    for ex in keys:
        ek = ex.get("key", "") if isinstance(ex, dict) else ex
        if ek == key:
            return {"success": False, "error": "Key đã tồn tại"}
    keys.append({"key": key, "quota_tokens": quota} if quota else {"key": key})
    _write_yaml(config_path, config)
    return {"success": True, "providers": _get_providers()}


def _action_remove_key(data: dict) -> dict:
    provider_id = (data.get("providerId") or "").strip()
    key_preview = (data.get("keyPreview") or "").strip()
    if not provider_id or not key_preview:
        return {"success": False, "error": "Thiếu providerId/keyPreview"}
    config_path = HERMES_HOME / "multi_provider.yaml"
    config = _read_yaml(config_path)
    if not config:
        return {"success": False, "error": "Không có multi_provider.yaml"}
    providers = config.get("providers", {})
    if provider_id not in providers:
        return {"success": False, "error": f"Provider '{provider_id}' không có"}
    keys = providers[provider_id].get("keys", [])
    new_keys = []
    for k in keys:
        ks = k.get("key", "") if isinstance(k, dict) else k
        preview = ks[:6] + "..." + ks[-4:] if len(ks) > 12 else "***"
        if preview != key_preview:
            new_keys.append(k)
    providers[provider_id]["keys"] = new_keys
    _write_yaml(config_path, config)
    return {"success": True, "providers": _get_providers()}


def _action_toggle_key(data: dict) -> dict:
    provider_id = (data.get("providerId") or "").strip()
    key_preview = (data.get("keyPreview") or "").strip()
    if not provider_id or not key_preview:
        return {"success": False, "error": "Thiếu providerId/keyPreview"}
    config_path = HERMES_HOME / "multi_provider.yaml"
    config = _read_yaml(config_path)
    if not config:
        return {"success": False, "error": "Không có multi_provider.yaml"}
    providers = config.get("providers", {})
    if provider_id not in providers:
        return {"success": False, "error": f"Provider '{provider_id}' không có"}
    keys = providers[provider_id].get("keys", [])
    for k in keys:
        if not isinstance(k, dict):
            continue
        ks = k.get("key", "")
        preview = ks[:6] + "..." + ks[-4:] if len(ks) > 12 else "***"
        if preview == key_preview:
            k["enabled"] = not k.get("enabled", True)
            _write_yaml(config_path, config)
            return {"success": True, "providers": _get_providers()}
    return {"success": False, "error": "Không tìm thấy key"}


def _action_add_provider(data: dict) -> dict:
    name = (data.get("name") or "").strip().lower()
    base_url = (data.get("baseUrl") or "").strip()
    models = data.get("models") or []
    key = (data.get("key") or "").strip()
    quota = int(data.get("quota", 0) or 0)
    if not name or not base_url:
        return {"success": False, "error": "Cần name + baseUrl"}
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    config_path = HERMES_HOME / "multi_provider.yaml"
    config = _read_yaml(config_path) or {"providers": {}, "strategy": "round-robin"}
    providers = config.setdefault("providers", {})
    if name in providers:
        return {"success": False, "error": "Provider đã có"}
    providers[name] = {
        "base_url": base_url, "enabled": True, "models": models,
        "keys": ([{"key": key, "quota_tokens": quota}] if quota else [{"key": key}]) if key else [],
    }
    _write_yaml(config_path, config)
    return {"success": True, "providers": _get_providers()}


def _action_clear_chat(data: dict) -> dict:
    """No-op for compatibility — chat is one-shot so no server-side history."""
    return {"success": True}


# ─── HTML ───────────────────────────────────────────────────────────────────
# Built as a separate function to keep this file readable.

def _build_html() -> str:
    provider_options = "".join(
        f'<option value="{k}">{v}</option>' for k, v in PROVIDER_LABELS.items()
    )
    return _HTML_TEMPLATE.replace("__PROVIDER_OPTIONS__", provider_options)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
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
  --r-sm:6px;--r:10px;--r-lg:14px;
  --mono:"SF Mono",ui-monospace,Menlo,Monaco,Consolas,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;
  display:flex;-webkit-font-smoothing:antialiased;
}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--border-2)}

.sidebar{
  width:56px;background:var(--sidebar);border-right:1px solid var(--border);
  display:flex;flex-direction:column;align-items:center;
  padding:10px 0 12px;gap:4px;position:sticky;top:0;height:100vh;flex-shrink:0;z-index:10;
}
.logo{
  width:38px;height:38px;border-radius:11px;
  background:linear-gradient(135deg,#e0a030 0%,var(--accent) 100%);
  color:#fff;display:flex;align-items:center;justify-content:center;
  font-size:1.05rem;margin-bottom:8px;
  box-shadow:0 4px 12px rgba(200,134,13,.35);
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
  box-shadow:0 3px 10px rgba(200,134,13,.28);
}
.nav-spacer{flex:1}

.main{flex:1;display:flex;flex-direction:column;min-width:0;height:100vh}

.topbar{
  padding:14px 22px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:var(--card);gap:12px;flex-shrink:0;
}
.topbar-title{font-weight:700;font-size:1.05rem;letter-spacing:-.01em}
.topbar-actions{display:flex;gap:8px;align-items:center}
.badge{
  padding:4px 11px;border-radius:9999px;font-size:.68rem;font-weight:600;
  display:inline-flex;align-items:center;gap:4px;
}
.badge-g{background:var(--green-soft);color:var(--green)}
.badge-r{background:var(--red-soft);color:var(--red)}
.badge-y{background:rgba(212,160,23,.15);color:#9a7600}
.btn{
  padding:8px 14px;border-radius:var(--r-sm);font-size:.76rem;font-weight:600;
  cursor:pointer;border:none;transition:all .18s ease;font-family:inherit;
  display:inline-flex;align-items:center;gap:5px;line-height:1;
}
.btn-p{background:var(--accent);color:#fff;box-shadow:0 2px 6px rgba(200,134,13,.25)}
.btn-p:hover{background:var(--accent-2)}
.btn-g{background:var(--green);color:#fff}
.btn-g:hover{background:#4d7a30}
.btn-r{background:var(--red);color:#fff}
.btn-r:hover{background:#b04040}
.btn-h{background:var(--card);color:var(--dim);border:1px solid var(--border)}
.btn-h:hover{background:var(--hover);color:var(--text);border-color:var(--border-2)}
.btn:disabled{opacity:.5;cursor:not-allowed}

.view{display:none;flex:1;overflow-y:auto;padding:22px}
.view.active{display:block}
.chat-view{display:none;flex:1;flex-direction:column;min-height:0}
.chat-view.active{display:flex}

.chat-msgs{flex:1;overflow-y:auto;padding:22px 22px 8px;scroll-behavior:smooth}
.load-more{
  text-align:center;padding:8px 12px;color:var(--accent);font-size:.72rem;
  cursor:pointer;margin-bottom:14px;font-weight:600;
}
.load-more:hover{text-decoration:underline}
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
.msg.error{background:var(--red-soft);color:var(--red);border:1px solid var(--red);margin-right:auto;border-bottom-left-radius:4px}

.chat-bottom{padding:12px 22px 16px;border-top:1px solid var(--border);background:var(--card);flex-shrink:0}
.file-area{
  padding:11px;border:1.5px dashed var(--border-2);border-radius:var(--r);
  text-align:center;font-size:.72rem;color:var(--muted);cursor:pointer;
  margin-bottom:10px;transition:all .18s ease;background:rgba(245,240,232,.5);
}
.file-area:hover{border-color:var(--accent);background:var(--accent-soft);color:var(--accent-2)}
.file-list{font-size:.7rem;color:var(--dim);margin-top:4px;margin-bottom:8px;font-family:var(--mono)}

.mode-panel{
  display:none;padding:10px 14px;background:var(--bg);border:1px solid var(--border);
  border-radius:var(--r);margin-bottom:10px;
}
.mode-panel.open{display:block}
.mode-bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.mode-group{display:flex;align-items:center;gap:5px}
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

.chat-input-row{display:flex;gap:8px;align-items:flex-end;margin-top:8px}
.chat-input-row textarea{
  flex:1;padding:11px 14px;border:1px solid var(--border);border-radius:var(--r);
  font-size:.85rem;background:var(--bg);color:var(--text);resize:none;
  font-family:inherit;line-height:1.5;max-height:120px;transition:all .15s ease;
}
.chat-input-row textarea:focus{outline:none;border-color:var(--accent);background:var(--card-2);box-shadow:0 0 0 3px var(--accent-soft)}
.chat-input-row .btn{padding:11px 20px;font-size:.8rem}

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

.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px;margin-bottom:12px;box-shadow:var(--shadow-xs)}

.pc{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px;margin-bottom:12px;box-shadow:var(--shadow-xs);transition:border-color .15s ease}
.pc:hover{border-color:var(--border-2)}
.kr{display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:var(--r-sm);background:var(--bg);margin-bottom:5px}
.kp{font-family:var(--mono);font-size:.68rem;color:var(--dim);min-width:130px;font-weight:600}
.qb{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.qf{height:100%;border-radius:3px;transition:width .3s ease}
.qf.g{background:var(--green)} .qf.y{background:#d4a017} .qf.r{background:var(--red)}
.ks{font-size:.58rem;padding:2px 7px;border-radius:4px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;min-width:64px;text-align:center}
.ks.active{background:var(--green-soft);color:var(--green)}
.ks.exhausted{background:var(--red-soft);color:var(--red)}
.ks.disabled{background:rgba(153,153,153,.15);color:var(--muted)}
.mb{display:inline-block;background:var(--accent-soft);color:var(--accent);padding:2px 8px;border-radius:4px;font-size:.6rem;font-family:var(--mono);margin:0 4px 4px 0;font-weight:600}

.cf{display:flex;align-items:center;justify-content:space-between;padding:11px 14px;border-radius:var(--r-sm);background:var(--card);margin-bottom:4px;transition:background .15s ease;border:1px solid transparent}
.cf:hover{background:var(--hover);border-color:var(--border)}
.cfp{font-family:var(--mono);font-size:.66rem;color:var(--muted)}
.cfd{font-size:.76rem;color:var(--dim);margin-top:3px}
.tg{width:38px;height:21px;background:var(--border-2);border-radius:11px;cursor:pointer;position:relative;transition:background .2s ease;flex-shrink:0}
.tg.on{background:var(--green)}
.tg::after{content:"";position:absolute;top:2px;left:2px;width:17px;height:17px;background:#fff;border-radius:50%;transition:left .2s ease;box-shadow:0 1px 3px rgba(0,0,0,.18)}
.tg.on::after{left:19px}

.sk-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px}
.skc{background:var(--card);border:1px solid var(--border);border-radius:var(--r-sm);padding:12px 14px;transition:all .15s ease}
.skc:hover{border-color:var(--border-2);box-shadow:var(--shadow-xs)}
.skn{font-weight:700;font-size:.78rem;color:var(--accent-2)}
.skr{font-size:.6rem;color:var(--muted);font-family:var(--mono);margin-top:2px}
.skde{font-size:.72rem;color:var(--dim);margin:5px 0 2px}

.search{width:100%;padding:9px 13px;border:1px solid var(--border);border-radius:var(--r-sm);font-size:.8rem;background:var(--card);color:var(--text);margin-bottom:10px;font-family:inherit;transition:all .15s ease}
.search:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
select.search{cursor:pointer;appearance:none;background-image:url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%237a6f5c' stroke-width='2.5'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");background-repeat:no-repeat;background-position:right 10px center;background-size:14px;padding-right:32px}
select.search option{background:var(--card);color:var(--text)}

.section-title{font-weight:700;font-size:.92rem;margin-bottom:4px;letter-spacing:-.01em}
.section-sub{font-size:.74rem;color:var(--muted);margin-bottom:14px}
.cfg-cat{font-weight:700;color:var(--accent);font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;margin:14px 0 6px;padding-bottom:4px;border-bottom:1px solid var(--border)}

.empty{text-align:center;padding:40px 20px;color:var(--muted);font-size:.85rem}
.empty-icon{font-size:2.5rem;margin-bottom:8px;opacity:.5}

.log-row{display:flex;gap:8px;padding:6px 8px;font-size:.7rem;border-bottom:1px solid var(--border);align-items:flex-start}
.log-row:hover{background:var(--bg)}
.log-ts{font-family:var(--mono);color:var(--muted);flex-shrink:0;min-width:64px}
.log-lvl{font-weight:700;min-width:48px;font-size:.6rem;padding-top:1px}
.log-lvl.info{color:var(--blue)}
.log-lvl.success{color:var(--green)}
.log-lvl.warn{color:#9a7600}
.log-lvl.error{color:var(--red)}
.log-mod{color:var(--accent);font-family:var(--mono);min-width:80px;font-size:.66rem}
.log-msg{flex:1;color:var(--text);word-break:break-word}

@media(max-width:768px){
  body{flex-direction:column}
  .sidebar{width:100%;height:auto;flex-direction:row;position:sticky;top:0;border-right:none;border-bottom:1px solid var(--border);padding:6px 10px;gap:3px;overflow-x:auto;background:rgba(237,229,214,.97);backdrop-filter:blur(10px)}
  .logo{margin-bottom:0;margin-right:6px;flex-shrink:0;width:34px;height:34px}
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
  .mode-bar{gap:8px}
  .mode-divider{display:none}
  .view{padding:14px}
}
</style></head>
<body>
<nav class="sidebar">
  <div class="logo">⚡</div>
  <button class="nav-btn active" data-view="chat" title="Trò chuyện">💬</button>
  <button class="nav-btn" data-view="overview" title="Tổng quan">📊</button>
  <button class="nav-btn" data-view="providers" title="Nhà cung cấp API">🔌</button>
  <button class="nav-btn" data-view="config" title="Cấu hình">⚙️</button>
  <button class="nav-btn" data-view="skills" title="Kỹ năng">📚</button>
  <button class="nav-btn" data-view="costs" title="Chi phí">💰</button>
  <button class="nav-btn" data-view="logs" title="Nhật ký">📋</button>
  <div class="nav-spacer"></div>
</nav>
<div class="main">
<div class="topbar">
  <div class="topbar-title" id="vt">Trò chuyện</div>
  <div class="topbar-actions">
    <span class="badge badge-r" id="agentBadge">● Agent: Sẵn sàng</span>
    <span class="badge badge-y" id="curCfgBadge">…</span>
  </div>
</div>

<div id="chat" class="chat-view active">
  <div class="chat-msgs" id="cm">
    <div class="msg sys">Hermes-Omni • Sẵn sàng trò chuyện</div>
  </div>
  <div class="chat-bottom">
    <div class="file-area" id="fa">📎 Kéo thả tệp hoặc nhấn để chọn<input type="file" id="fi" multiple style="display:none"></div>
    <div class="file-list" id="fl"></div>
    <div class="mode-panel" id="modePanel">
      <div class="mode-bar">
        <div class="mode-group"><span class="mode-label">🧠 Thinking</span>
          <button class="mode-btn" data-mt="thinking" data-mv="fast">Fast</button>
          <button class="mode-btn active" data-mt="thinking" data-mv="balanced">Balanced</button>
          <button class="mode-btn" data-mt="thinking" data-mv="deep">Deep</button>
          <button class="mode-btn" data-mt="thinking" data-mv="max">Max</button>
        </div>
        <div class="mode-divider"></div>
        <div class="mode-group"><span class="mode-label">⚡ Reasoning</span>
          <button class="mode-btn" data-mt="reasoning" data-mv="off">Off</button>
          <button class="mode-btn active" data-mt="reasoning" data-mv="standard">Std</button>
          <button class="mode-btn" data-mt="reasoning" data-mv="high">High</button>
          <button class="mode-btn" data-mt="reasoning" data-mv="max">Max</button>
        </div>
        <div class="mode-divider"></div>
        <div class="mode-group"><span class="mode-label">🛡️ Verify</span>
          <button class="mode-btn" data-mt="verify" data-mv="off">Off</button>
          <button class="mode-btn active" data-mt="verify" data-mv="on">On</button>
        </div>
      </div>
    </div>
    <div class="chat-input-row">
      <textarea id="ci" placeholder="Nhập tin nhắn cho Hermes... (Enter để gửi, Shift+Enter xuống dòng)" rows="1"></textarea>
      <button class="btn btn-h" id="modeBtn" title="Chế độ suy luận">⚙️</button>
      <button class="btn btn-h" id="clearBtn" title="Xóa hội thoại">🗑️</button>
      <button class="btn btn-p" id="sendBtn">Gửi ➤</button>
    </div>
  </div>
</div>

<div id="overview" class="view"><div class="sg" id="ovGrid"></div><div id="ovExtra"></div></div>

<div id="providers" class="view">
  <div id="pl"></div>
  <div class="card">
    <div class="section-title">🔑 Thiết lập API Key</div>
    <div class="section-sub">Cấu hình nhà cung cấp AI cho agent. Đồng bộ với <code>config.yaml</code> + <code>.env</code>.</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <select id="sp-provider" class="search" style="margin:0">
        <option value="">Chọn nhà cung cấp...</option>
        __PROVIDER_OPTIONS__
      </select>
      <input id="sp-model" class="search" style="margin:0" placeholder="Model (vd: mimo-v2.5)">
      <input id="sp-baseurl" class="search" style="margin:0" placeholder="Base URL (auto nếu để trống)">
      <input id="sp-apikey" class="search" style="margin:0" type="password" placeholder="API Key">
    </div>
    <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">
      <button class="btn btn-p" id="setupBtn">💾 Lưu cấu hình</button>
      <button class="btn btn-h" id="testBtn">🧪 Kiểm tra key</button>
      <button class="btn btn-h" id="fillDefaultBtn">↩ Điền mặc định</button>
    </div>
    <div id="sp-result" style="font-size:.74rem;margin-top:10px;color:var(--dim);padding:8px 10px;background:var(--bg);border-radius:var(--r-sm);min-height:18px"></div>
  </div>
  <button class="btn btn-h" id="addProvBtn" style="margin-top:8px">+ Thêm nhà cung cấp (multi-key)</button>
</div>

<div id="config" class="view">
  <input class="search" id="cfgSearch" placeholder="🔍 Tìm cấu hình...">
  <div id="cl"></div>
</div>

<div id="skills" class="view">
  <input class="search" id="skSearch" placeholder="🔍 Tìm kỹ năng...">
  <div id="skList"></div>
</div>

<div id="costs" class="view"><div id="cs"></div></div>

<div id="logs" class="view">
  <div class="card" style="padding:8px;max-height:560px;overflow:auto" id="lf"></div>
</div>

</div>

<script>
const T = {
  chat:'Trò chuyện', overview:'Tổng quan', providers:'Nhà cung cấp API',
  config:'Cấu hình', skills:'Kỹ năng', costs:'Chi phí', logs:'Nhật ký hoạt động'
};
let msgs = [];
let visCount = 25;
let curMode = {thinking:'balanced', reasoning:'standard', verify:'on'};
let sending = false;

// ─── Navigation ───
document.querySelectorAll('.nav-btn[data-view]').forEach(btn => {
  btn.addEventListener('click', () => {
    const v = btn.dataset.view;
    document.querySelectorAll('.view,.chat-view').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(x => x.classList.remove('active'));
    document.getElementById(v).classList.add('active');
    btn.classList.add('active');
    document.getElementById('vt').textContent = T[v] || v;
    if (v === 'overview') loadOverview();
    if (v === 'providers') { loadProviders(); loadCurrentConfig(); }
    if (v === 'config') loadConfig();
    if (v === 'skills') loadSkills();
    if (v === 'costs') loadCosts();
    if (v === 'logs') loadLogs();
  });
});

// ─── Helpers ───
function esc(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function fmtR(t){
  if(!t) return '';
  t = String(t);
  // Detect JSON
  const trimmed = t.trim();
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
    try { return '<pre>'+esc(JSON.stringify(JSON.parse(trimmed),null,2))+'</pre>'; } catch(e) {}
  }
  // Markdown-ish: code blocks
  if (t.includes('```')) {
    return t.split(/```(\w*)/).slice(1).reduce((acc, part, i) => {
      if (i % 2 === 0) return acc + '<pre>';
      return acc + esc(part) + '</pre>';
    }, esc(t.split('```')[0]));
  }
  return esc(t).replace(/`([^`]+)`/g, '<code>$1</code>');
}
function fn(n){n=+n||0;return n>1e6?(n/1e6).toFixed(1)+'M':n>1e3?(n/1e3).toFixed(0)+'K':String(n)}

async function api(p){
  try { return await (await fetch('/api/'+p)).json(); }
  catch(e){ return null; }
}
async function post(a,d){
  try {
    const r = await fetch('/api/action/'+a, {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(d||{})
    });
    if (!r.ok) {
      try { return await r.json(); } catch(_) { return {success:false, error:'HTTP '+r.status}; }
    }
    return await r.json();
  } catch(e) { return {success:false, error:'Lỗi kết nối: '+e.message}; }
}

// ─── Chat ───
function render(){
  const c = document.getElementById('cm');
  const v = msgs.slice(-visCount);
  let h = '';
  if (msgs.length > visCount) {
    h += '<div class="load-more" id="lm">↑ Tải cũ hơn ('+(msgs.length-visCount)+' tin)</div>';
  }
  v.forEach(m => {
    h += '<div class="msg '+m.t+'">'+(m.t==='user'?esc(m.c):m.t==='sys'?m.c:fmtR(m.c))+'</div>';
  });
  c.innerHTML = h;
  c.scrollTop = c.scrollHeight;
  const lm = document.getElementById('lm');
  if (lm) lm.onclick = () => { visCount += 25; render(); };
}
function addM(t, c){ msgs.push({t, c}); render(); }

async function send(){
  if (sending) return;
  const i = document.getElementById('ci');
  const m = i.value.trim();
  if (!m) return;
  i.value = '';
  i.style.height = 'auto';
  sending = true;
  document.getElementById('sendBtn').disabled = true;
  document.getElementById('sendBtn').textContent = '⏳...';
  addM('user', m);
  addM('sys', '⏳ Đang suy nghĩ...');
  const r = await post('chat', {message:m});
  // remove the "thinking..." message
  const last = msgs[msgs.length-1];
  if (last && last.t === 'sys' && last.c.startsWith('⏳')) msgs.pop();
  if (r && r.success) {
    addM('agent', r.response || '(Không có phản hồi)');
    if (r.warning) addM('sys', '⚠ '+r.warning);
  } else {
    const err = r ? (r.error || 'Không có phản hồi') : 'Không có phản hồi';
    addM('error', '❌ ' + err);
    if (err.includes('key') || err.includes('provider') || err.includes('API') || err.includes('No module')) {
      addM('sys', '💡 Vào tab "Nhà cung cấp API" để cấu hình key');
    }
  }
  sending = false;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('sendBtn').textContent = 'Gửi ➤';
  loadOverview();
}

document.getElementById('sendBtn').addEventListener('click', send);
document.getElementById('ci').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
document.getElementById('ci').addEventListener('input', function(){
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 100) + 'px';
});
document.getElementById('clearBtn').addEventListener('click', () => {
  if (confirm('Xóa toàn bộ hội thoại?')) {
    msgs = []; visCount = 25;
    document.getElementById('cm').innerHTML = '<div class="msg sys">Hermes-Omni • Sẵn sàng trò chuyện</div>';
  }
});

// ─── File upload ───
const fa = document.getElementById('fa');
const fi = document.getElementById('fi');
fa.addEventListener('click', () => fi.click());
fa.ondragover = e => { e.preventDefault(); fa.style.borderColor = 'var(--accent)'; };
fa.ondragleave = () => { fa.style.borderColor = ''; };
fa.ondrop = e => { e.preventDefault(); fa.style.borderColor = ''; uploadFiles(e.dataTransfer.files); };
fi.addEventListener('change', () => uploadFiles(fi.files));

async function uploadFiles(fs){
  const fl = document.getElementById('fl');
  for (const f of fs) {
    const fd = new FormData();
    fd.append('file', f);
    try {
      const r = await fetch('/api/upload', {method:'POST', body:fd});
      const res = await r.json();
      if (res.success) {
        fl.innerHTML += '✓ '+esc(f.name)+'<br>';
        addM('sys', '📎 Đã tải lên: '+f.name);
      } else {
        fl.innerHTML += '❌ '+esc(f.name)+': '+(res.error||'?')+'<br>';
      }
    } catch(e) {
      fl.innerHTML += '❌ '+esc(f.name)+': '+e.message+'<br>';
    }
  }
}

// ─── Mode panel ───
document.getElementById('modeBtn').addEventListener('click', () => {
  document.getElementById('modePanel').classList.toggle('open');
});
document.querySelectorAll('.mode-btn[data-mt]').forEach(btn => {
  btn.addEventListener('click', async () => {
    const mt = btn.dataset.mt, mv = btn.dataset.mv;
    document.querySelectorAll('.mode-btn[data-mt="'+mt+'"]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    curMode[mt] = mv;
    const r = await post('set-mode', curMode);
    if (r && r.success) addM('sys', '⚙️ '+mt+' = '+mv);
  });
});

// ─── Overview ───
async function loadOverview(){
  const s = await api('status');
  if (!s) return;
  document.getElementById('ovGrid').innerHTML =
    '<div class="sc"><div class="sl">Provider</div><div class="sv">'+esc(s.currentProvider||'—')+'</div><div class="ss">'+esc(s.currentModel||'')+'</div></div>'+
    '<div class="sc"><div class="sl">API Key</div><div class="sv">'+(s.hasKey?'✅':'❌')+'</div><div class="ss">'+(s.hasKey?'đã cấu hình':'chưa có')+'</div></div>'+
    '<div class="sc"><div class="sl">Multi-keys</div><div class="sv">'+s.totalKeys+'</div><div class="ss">'+s.providers+' providers</div></div>'+
    '<div class="sc"><div class="sl">Skills</div><div class="sv">'+s.installedSkills+'</div><div class="ss">đã cài</div></div>'+
    '<div class="sc"><div class="sl">Features</div><div class="sv">'+s.enabledFeatures+'/'+s.totalFeatures+'</div><div class="ss">đã bật</div></div>'+
    '<div class="sc"><div class="sl">Tokens</div><div class="sv">'+fn(s.totalCostTokens)+'</div><div class="ss">'+s.totalCalls+' calls</div></div>';
  // Update badge
  const badge = document.getElementById('curCfgBadge');
  if (s.hasKey) {
    badge.className = 'badge badge-g';
    badge.textContent = '✓ '+s.currentProvider+' · '+s.currentModel;
  } else {
    badge.className = 'badge badge-r';
    badge.textContent = '⚠ Chưa có API key';
  }
  const ab = document.getElementById('agentBadge');
  ab.className = 'badge badge-g';
  ab.textContent = '● Sẵn sàng';
}

// ─── Providers ───
async function loadProviders(){
  const p = await api('providers');
  const el = document.getElementById('pl');
  if (!p || !p.length) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">🔌</div>Chưa có multi-provider nào. Dùng form bên dưới để thiết lập API key chính.</div>';
    return;
  }
  el.innerHTML = p.map(pr => 
    '<div class="pc">'+
      '<div style="font-weight:700;font-size:.85rem">'+esc(pr.name)+'</div>'+
      '<div style="font-size:.6rem;color:var(--muted);font-family:var(--mono);margin-bottom:6px">'+esc(pr.baseUrl)+'</div>'+
      '<div style="margin:.15rem 0">'+(pr.models||[]).map(m => '<span class="mb">'+esc(m)+'</span>').join('')+'</div>'+
      (pr.keys||[]).map(k => {
        const st = k.exhausted ? 'exhausted' : (k.enabled ? 'active' : 'disabled');
        const pct = k.quota > 0 ? Math.round(k.used/k.quota*100) : 0;
        const c = pct > 80 ? 'r' : (pct > 50 ? 'y' : 'g');
        return '<div class="kr"><span class="kp">'+esc(k.keyPreview)+'</span>'+
          '<div class="qb"><div class="qf '+c+'" style="width:'+Math.max(pct,2)+'%"></div></div>'+
          '<span class="ks '+st+'">'+st+'</span>'+
          '<button class="btn btn-h" style="font-size:.55rem;padding:3px 7px" data-tk="'+esc(pr.id)+'" data-tkprev="'+esc(k.keyPreview)+'">⏻</button>'+
          '<button class="btn btn-r" style="font-size:.55rem;padding:3px 7px" data-rk="'+esc(pr.id)+'" data-rkprev="'+esc(k.keyPreview)+'">✕</button>'+
          '</div>';
      }).join('')+
      '<button class="btn btn-h" style="font-size:.65rem;margin-top:.3rem" data-ak="'+esc(pr.id)+'">+ Add Key</button>'+
    '</div>'
  ).join('');
  // Bind buttons
  el.querySelectorAll('[data-ak]').forEach(b => b.onclick = () => addKey(b.dataset.ak));
  el.querySelectorAll('[data-tk]').forEach(b => b.onclick = () => toggleKey(b.dataset.tk, b.dataset.tkprev));
  el.querySelectorAll('[data-rk]').forEach(b => b.onclick = () => removeKey(b.dataset.rk, b.dataset.rkprev));
}

async function addKey(pid){
  const k = prompt('API Key:');
  if (!k) return;
  const q = prompt('Quota tokens (0=∞):', '0') || '0';
  const r = await post('add-key', {providerId:pid, key:k, quota:parseInt(q)});
  if (r && r.success) loadProviders();
  else alert(r ? r.error : 'Lỗi');
}
async function toggleKey(pid, prev){
  const r = await post('toggle-key', {providerId:pid, keyPreview:prev});
  if (r && r.success) loadProviders();
  else alert(r ? r.error : 'Lỗi');
}
async function removeKey(pid, prev){
  if (!confirm('Xóa key '+prev+'?')) return;
  const r = await post('remove-key', {providerId:pid, keyPreview:prev});
  if (r && r.success) loadProviders();
  else alert(r ? r.error : 'Lỗi');
}

document.getElementById('addProvBtn').addEventListener('click', async () => {
  const n = prompt('Tên provider (vd: openai2):');
  if (!n) return;
  const u = prompt('Base URL:');
  if (!u) return;
  const m = prompt('Models (cách nhau bởi dấu phẩy):');
  const k = prompt('API key (có thể bỏ trống):');
  const r = await post('add-provider', {name:n, baseUrl:u, models:m?m.split(','):[], key:k||'', quota:0});
  if (r && r.success) loadProviders();
  else alert(r ? r.error : 'Lỗi');
});

// ─── Provider setup form ───
async function loadCurrentConfig(){
  const c = await api('current-config');
  const el = document.getElementById('sp-result');
  if (!c) return;
  if (c.provider) {
    el.innerHTML = '📋 Hiện tại: <b>'+esc(c.provider)+'</b> | Model: <b>'+esc(c.model)+'</b> | Key: '+(c.has_key?'<b>'+esc(c.key_preview)+'</b> ✅':'<b>chưa có</b> ❌');
    document.getElementById('sp-provider').value = c.provider;
    if (c.model) document.getElementById('sp-model').value = c.model;
    if (c.base_url) document.getElementById('sp-baseurl').value = c.base_url;
  } else {
    el.innerHTML = '⚠ Chưa cấu hình. Chọn provider + nhập key rồi bấm Lưu.';
  }
}
document.getElementById('setupBtn').addEventListener('click', async () => {
  const p = document.getElementById('sp-provider').value;
  const m = document.getElementById('sp-model').value;
  const u = document.getElementById('sp-baseurl').value;
  const k = document.getElementById('sp-apikey').value;
  if (!p || !k) {
    document.getElementById('sp-result').innerHTML = '❌ Chọn provider + nhập key';
    return;
  }
  const r = await post('setup-provider', {provider:p, apiKey:k, model:m, baseUrl:u});
  if (r && r.success) {
    document.getElementById('sp-result').innerHTML = '✅ '+esc(r.message);
    document.getElementById('sp-apikey').value = '';
    loadOverview();
  } else {
    document.getElementById('sp-result').innerHTML = '❌ '+(r?r.error:'Fail');
  }
});
document.getElementById('testBtn').addEventListener('click', async () => {
  const p = document.getElementById('sp-provider').value;
  const m = document.getElementById('sp-model').value;
  const u = document.getElementById('sp-baseurl').value;
  const k = document.getElementById('sp-apikey').value;
  if (!k) { document.getElementById('sp-result').innerHTML = '❌ Nhập key'; return; }
  document.getElementById('sp-result').innerHTML = '⏳ Đang test...';
  const r = await post('test-key', {provider:p, apiKey:k, baseUrl:u, model:m});
  if (r && r.success) document.getElementById('sp-result').innerHTML = '✅ '+esc(r.message);
  else document.getElementById('sp-result').innerHTML = '❌ '+(r?r.error:'Fail');
});
document.getElementById('fillDefaultBtn').addEventListener('click', () => {
  const p = document.getElementById('sp-provider').value;
  const defaults = {
    zai: {model:'glm-4.6', url:'https://open.bigmodel.cn/api/paas/v4'},
    xiaomi: {model:'mimo-v2.5', url:'https://api.xiaomimimo.com/v1'},
    openai: {model:'gpt-4o-mini', url:'https://api.openai.com/v1'},
    anthropic: {model:'claude-3-5-sonnet-latest', url:'https://api.anthropic.com'},
    openrouter: {model:'openai/gpt-4o-mini', url:'https://openrouter.ai/api/v1'},
    deepseek: {model:'deepseek-chat', url:'https://api.deepseek.com/v1'},
    groq: {model:'llama-3.1-70b-versatile', url:'https://api.groq.com/openai/v1'},
    together: {model:'meta-llama/Llama-3-70b-chat-hf', url:'https://api.together.xyz/v1'},
    mistral: {model:'mistral-large-latest', url:'https://api.mistral.ai/v1'},
  };
  if (defaults[p]) {
    document.getElementById('sp-model').value = defaults[p].model;
    document.getElementById('sp-baseurl').value = defaults[p].url;
  }
});

// ─── Config ───
async function loadConfig(){
  const c = await api('config');
  if (!c) return;
  const g = {};
  c.forEach(f => { (g[f.category] = g[f.category] || []).push(f); });
  let h = '';
  for (const [cat, fs] of Object.entries(g)) {
    const en = fs.filter(f => f.enabled).length;
    h += '<div class="cfg-cat">'+esc(cat)+' ('+en+'/'+fs.length+')</div>';
    h += fs.map(f =>
      '<div class="cf"><div><div class="cfp">'+esc(f.path)+'</div><div class="cfd">'+esc(f.description)+'</div></div>'+
      '<div class="tg '+(f.enabled?'on':'')+'" data-cfg="'+esc(f.path)+'"></div></div>'
    ).join('');
  }
  document.getElementById('cl').innerHTML = h;
  document.querySelectorAll('[data-cfg]').forEach(t => {
    t.onclick = async () => {
      const r = await post('toggle-config', {path:t.dataset.cfg});
      if (r && r.success) {
        t.classList.toggle('on');
        loadOverview();
      }
    };
  });
}
document.getElementById('cfgSearch').addEventListener('input', function(){
  const q = this.value.toLowerCase();
  document.querySelectorAll('.cf').forEach(f => {
    f.style.display = f.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});

// ─── Skills ───
async function loadSkills(){
  const s = await api('skills');
  const el = document.getElementById('skList');
  if (!s || !s.length) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">📚</div>Chưa có kỹ năng nào.<br>Chạy <code>scripts/clone-skills.sh</code> để tải về.</div>';
    return;
  }
  el.innerHTML = '<div class="card" style="margin-bottom:12px"><b>📚 '+s.length+' kỹ năng</b><br><span style="font-size:.75rem;color:var(--dim)">Agent tự động tìm kỹ năng phù hợp khi cần.</span></div>'+
    '<div class="sk-grid">'+s.map(sk =>
      '<div class="skc"><div class="skn">'+esc(sk.name)+'</div>'+
      '<div class="skr">'+esc(sk.repo)+'</div>'+
      '<div class="skde">'+esc(sk.desc)+'</div></div>'
    ).join('')+'</div>';
}
document.getElementById('skSearch').addEventListener('input', function(){
  const q = this.value.toLowerCase();
  document.querySelectorAll('.skc').forEach(c => {
    c.style.display = c.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});

// ─── Costs ───
async function loadCosts(){
  const c = await api('costs');
  const el = document.getElementById('cs');
  if (!c || !c.length) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">💰</div>Chưa có dữ liệu chi phí.<br>Chat với agent để bắt đầu ghi nhận.</div>';
    return;
  }
  const total = c.reduce((s,x) => s + x.tokens, 0);
  const calls = c.reduce((s,x) => s + x.calls, 0);
  const max = Math.max(...c.map(x => x.tokens), 1);
  el.innerHTML =
    '<div class="sg"><div class="sc"><div class="sl">Tổng tokens</div><div class="sv">'+fn(total)+'</div></div>'+
    '<div class="sc"><div class="sl">Tổng calls</div><div class="sv">'+calls+'</div></div></div>'+
    '<div class="card"><div class="section-title">Phân bổ theo phase</div>'+
    '<div style="display:flex;align-items:flex-end;gap:6px;height:160px;margin-top:14px;padding-bottom:24px">'+
    c.map(x => {
      const h = Math.round(x.tokens/max*120);
      return '<div style="flex:1;background:var(--accent);border-radius:3px 3px 0 0;height:'+h+'px;min-height:2px;position:relative">'+
        '<div style="position:absolute;top:-16px;left:50%;transform:translateX(-50%);font-size:.55rem;color:var(--dim);white-space:nowrap">'+fn(x.tokens)+'</div>'+
        '<div style="position:absolute;bottom:-18px;left:50%;transform:translateX(-50%);font-size:.55rem;color:var(--muted);white-space:nowrap">'+esc(x.phase)+'</div>'+
      '</div>';
    }).join('')+
    '</div></div>';
}

// ─── Logs ───
async function loadLogs(){
  const l = await api('logs');
  const el = document.getElementById('lf');
  if (!l || !l.length) {
    el.innerHTML = '<div class="empty">Chưa có log nào.</div>';
    return;
  }
  el.innerHTML = l.map(x =>
    '<div class="log-row"><span class="log-ts">'+esc(x.timestamp)+'</span>'+
    '<span class="log-lvl '+esc(x.level)+'">'+esc(x.level.toUpperCase())+'</span>'+
    '<span class="log-mod">'+esc(x.module)+'</span>'+
    '<span class="log-msg">'+esc(x.message)+'</span></div>'
  ).join('');
}

// ─── Init ───
loadOverview();
setInterval(loadOverview, 10000);
setInterval(loadLogs, 5000);
</script>
</body></html>"""


# ─── HTTP handler ───────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 300  # 5 min for long chat requests

    def _safe_write(self, data: bytes) -> bool:
        """Write to wfile, swallowing BrokenPipeError / ConnectionResetError
        when the client closes early."""
        try:
            self.wfile.write(data)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return False
        except Exception:
            return False

    def do_OPTIONS(self):
        """CORS preflight."""
        try:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()
        except Exception:
            pass

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                self._send_html(_build_html())
            elif path == "/api/status":
                self._send_json(_get_status())
            elif path == "/api/providers":
                self._send_json(_get_providers())
            elif path == "/api/config":
                self._send_json(_get_config())
            elif path == "/api/skills":
                self._send_json(_scan_skills())
            elif path == "/api/costs":
                self._send_json(_get_costs())
            elif path == "/api/logs":
                self._send_json(_get_logs())
            elif path == "/api/current-config":
                self._send_json(_get_current_config())
            elif path == "/api/health":
                self._send_json(_health_check())
            elif path.startswith("/api/download/"):
                self._serve_file(path[len("/api/download/"):])
            else:
                self._send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Client gone — nothing to do
        except Exception as e:
            try:
                self._send_json({"error": f"server: {type(e).__name__}: {e}"}, 500)
            except Exception:
                pass

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except Exception:
            length = 0
        try:
            if path == "/api/upload":
                self._handle_upload()
                return
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception:
            data = {}

        routes = {
            "/api/action/chat": lambda: _send_to_agent(data.get("message", "")),
            "/api/action/setup-provider": lambda: _action_setup_provider(data),
            "/api/action/test-key": lambda: _action_test_key(data),
            "/api/action/toggle-config": lambda: _action_toggle_config(data),
            "/api/action/set-mode": lambda: _action_set_mode(data),
            "/api/action/add-key": lambda: _action_add_key(data),
            "/api/action/remove-key": lambda: _action_remove_key(data),
            "/api/action/toggle-key": lambda: _action_toggle_key(data),
            "/api/action/add-provider": lambda: _action_add_provider(data),
            "/api/action/clear-chat": lambda: _action_clear_chat(data),
        }
        try:
            if path in routes:
                try:
                    self._send_json(routes[path]())
                except Exception as e:
                    self._send_json({"success": False, "error": f"{type(e).__name__}: {e}"}, 500)
            else:
                self._send_json({"error": "unknown action"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # client gone

    def _handle_upload(self):
        ct = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ct:
            self._send_json({"success": False, "error": "Not multipart"})
            return
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length)
            boundary = ct.split("boundary=")[1].encode()
            parts = body.split(b"--" + boundary)
            saved = []
            for part in parts:
                if b"Content-Disposition" not in part:
                    continue
                ds = part.find(b'filename="')
                if ds < 0:
                    continue
                ds += 10
                de = part.find(b'"', ds)
                filename = part[ds:de].decode("utf-8", errors="ignore")
                if not filename:
                    continue
                cs = part.find(b"\r\n\r\n")
                if cs < 0:
                    continue
                cs += 4
                ce = part.rfind(b"\r\n--")
                if ce < 0:
                    ce = len(part)
                file_data = part[cs:ce]
                safe = "".join(c for c in filename if c.isalnum() or c in "._-")
                if not safe:
                    safe = "upload_" + str(int(time.time()))
                fp = UPLOAD_DIR / safe
                fp.write_bytes(file_data)
                saved.append({"name": filename, "size": len(file_data), "path": str(fp)})
            if saved:
                self._send_json({"success": True, "files": saved})
            else:
                self._send_json({"success": False, "error": "No files found"})
        except Exception as e:
            try:
                self._send_json({"success": False, "error": str(e)})
            except Exception:
                pass

    def _serve_file(self, filename: str):
        fp = UPLOAD_DIR / filename
        if not fp.exists():
            self._send_json({"error": "File not found"}, 404)
            return
        try:
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self._safe_write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _send_html(self, html: str):
        try:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._safe_write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _send_json(self, data, code: int = 200):
        try:
            body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self._safe_write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args):
        pass


def _health_check() -> dict:
    """Return system health info for /api/health endpoint."""
    health = {
        "ok": True,
        "python": sys.version.split()[0],
        "hermes_home": str(HERMES_HOME),
        "repo_root": str(REPO_ROOT),
        "deps": {},
        "modules": {},
        "config": {},
        "warnings": [],
    }

    # Check deps
    for dep, pip in [("yaml", "pyyaml"), ("openai", "openai"), ("httpx", "httpx"), ("requests", "requests")]:
        try:
            m = __import__(dep)
            ver = getattr(m, "__version__", "?")
            health["deps"][dep] = f"✓ {ver}"
        except Exception as e:
            health["deps"][dep] = f"✗ {e}"
            health["ok"] = False
            health["warnings"].append(f"Missing dep: {pip}")

    # Check critical modules
    for mod in ["hermes_cli.main", "hermes_cli.oneshot", "run_agent", "agent.conversation_loop"]:
        try:
            __import__(mod)
            health["modules"][mod] = "✓"
        except Exception as e:
            health["modules"][mod] = f"✗ {type(e).__name__}: {e}"
            health["warnings"].append(f"Module import fail: {mod}")

    # Check config
    cfg = _get_current_config()
    health["config"] = cfg
    if not cfg["has_key"]:
        health["warnings"].append("No API key set. Visit Providers tab to set one.")
    if not cfg["provider"]:
        health["warnings"].append("No provider set.")

    # Check files exist
    health["files"] = {
        "config.yaml": (HERMES_HOME / "config.yaml").exists(),
        ".env": (HERMES_HOME / ".env").exists(),
        "multi_provider.yaml": (HERMES_HOME / "multi_provider.yaml").exists(),
    }

    return health


def run_server(port: int = 8788) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"═══ Hermes-Omni Dashboard v5 ═══", flush=True)
    print(f"  URL:      http://localhost:{port}", flush=True)
    print(f"  Hermes:   {HERMES_HOME}", flush=True)
    print(f"  Repo:     {REPO_ROOT}", flush=True)
    print(f"  Uploads:  {UPLOAD_DIR}", flush=True)

    # Pre-flight check
    print(f"\n  Pre-flight check:", flush=True)
    health = _health_check()
    for dep, status in health["deps"].items():
        print(f"    dep {dep}: {status}", flush=True)
    for mod, status in health["modules"].items():
        print(f"    mod {mod}: {status}", flush=True)
    cur = health["config"]
    print(f"    provider: {cur.get('provider') or '(none)'}", flush=True)
    print(f"    model:    {cur.get('model') or '(none)'}", flush=True)
    print(f"    key:      {'✓' if cur.get('has_key') else '✗ (chưa có — vào tab Provider)'}", flush=True)
    if health["warnings"]:
        print(f"\n  ⚠ Cảnh báo:", flush=True)
        for w in health["warnings"]:
            print(f"    - {w}", flush=True)
    print(flush=True)

    # Try common ports if 8788 is busy
    ports_to_try = [port]
    if port == 8788:
        ports_to_try += [8789, 8790, 8899, 9000]
    server = None
    actual_port = None
    for p in ports_to_try:
        try:
            server = ThreadingHTTPServer(("0.0.0.0", p), DashboardHandler)
            actual_port = p
            break
        except OSError as e:
            print(f"  Port {p} busy: {e}", flush=True)
            continue
    if server is None:
        print(f"❌ Không mở được server trên ports {ports_to_try}", flush=True)
        sys.exit(1)
    if actual_port != port:
        print(f"  → Đổi sang port {actual_port}", flush=True)

    print(f"  ▶ Server chạy tại: http://localhost:{actual_port}", flush=True)
    print(f"  ▶ Health: http://localhost:{actual_port}/api/health", flush=True)
    print(f"  Ctrl+C để dừng\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hermes-Omni Dashboard v5")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    run_server(port=args.port)
