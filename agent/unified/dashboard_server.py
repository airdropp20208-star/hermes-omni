"""Real-time Dashboard Server — reads actual Hermes data, serves web UI.

USAGE
-----
    python -m agent.unified.dashboard_server [--port 8788]

Then open: http://localhost:8788

FEATURES
--------
- Real-time data from ~/.hermes/ files (config.yaml, multi_provider.yaml, etc.)
- Auto-refresh every 3 seconds (logs) + 10 seconds (stats)
- Read-only monitoring (changes via config file or hermes CLI)
- Dark theme dashboard with 7 tabs
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def _repo_root() -> Path:
    return Path(__file__).parent.parent.parent


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
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
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
    """Scan skills/local-repos/ + ~/.hermes/skills/cached/ for SKILL.md."""
    skills = []
    # Local repos
    local_repos = _repo_root() / "skills" / "local-repos"
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
                skills.append({
                    "id": f"local-{repo_name}-{skill_name}",
                    "repo": f"local/{repo_name}",
                    "desc": desc,
                    "category": "local",
                    "stars": "local",
                    "installed": True,
                })
            except Exception:
                continue
    # Cached skills
    cache_dir = _hermes_home() / "skills" / "cached"
    if cache_dir.exists():
        for skill_dir in cache_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                skills.append({
                    "id": skill_dir.name,
                    "repo": "cached",
                    "desc": f"Cached skill: {skill_dir.name}",
                    "category": "cached",
                    "stars": "—",
                    "installed": True,
                })
    return skills


def _get_status() -> dict:
    """Read real status from Hermes files."""
    home = _hermes_home()
    config = _read_yaml(home / "config.yaml")
    unified = config.get("unified", {}) if config else {}
    
    # Count cognitive features enabled
    feature_keys = [
        "reasoning", "reflexion", "smart_guardian", "verifier", "constitution",
        "slow_thinking", "ensemble", "embedding", "user_model", "clarifier",
        "longrun", "tool_router", "cost_tracker", "response_cache",
        "output_formatter", "skill_registry", "api_registry", "multi_provider",
        "learning", "skill_synthesis", "task_planner",
    ]
    enabled = sum(1 for k in feature_keys if unified.get(k, {}).get("enabled", False))
    
    # Providers
    mp_config = _read_yaml(home / "multi_provider.yaml")
    providers = mp_config.get("providers", {}) if mp_config else {}
    total_keys = sum(len(p.get("keys", [])) for p in providers.values())
    
    # Skills
    skills = _scan_skills()
    
    # Costs
    cost_path = home / "unified" / "cost_log.jsonl"
    costs = _read_jsonl(cost_path, limit=1000)
    total_tokens = sum(c.get("total_tokens", 0) for c in costs)
    total_calls = len(costs)
    
    return {
        "providers": len(providers),
        "totalKeys": total_keys,
        "activeKeys": total_keys,  # simplified
        "exhaustedKeys": 0,
        "totalUsedTokens": total_tokens,
        "totalQuotaTokens": 0,
        "installedSkills": len(skills),
        "totalSkills": len(skills),
        "enabledFeatures": enabled,
        "totalFeatures": len(feature_keys),
        "totalCostTokens": total_tokens,
        "totalCalls": total_calls,
        "strategy": mp_config.get("strategy", "round-robin") if mp_config else "none",
        "uptime": "—",
    }


def _get_providers() -> list[dict]:
    """Read real providers from multi_provider.yaml."""
    home = _hermes_home()
    config = _read_yaml(home / "multi_provider.yaml")
    if not config:
        return []
    result = []
    for name, pdata in (config.get("providers") or {}).items():
        keys = []
        for k in (pdata.get("keys") or []):
            if isinstance(k, dict):
                key_str = k.get("key", "")
                quota = k.get("quota_tokens", 0)
                keys.append({
                    "id": f"k{len(keys)+1}",
                    "keyPreview": key_str[:4] + "..." + key_str[-4:] if len(key_str) > 8 else "***",
                    "quota": quota,
                    "used": 0,
                    "remaining": quota or 999999999,
                    "exhausted": quota > 0 and 0 >= quota,
                    "enabled": True,
                    "errorCount": 0,
                    "lastUsed": None,
                })
            elif isinstance(k, str):
                keys.append({
                    "id": f"k{len(keys)+1}",
                    "keyPreview": k[:4] + "..." + k[-4:] if len(k) > 8 else "***",
                    "quota": 0,
                    "used": 0,
                    "remaining": 999999999,
                    "exhausted": False,
                    "enabled": True,
                    "errorCount": 0,
                    "lastUsed": None,
                })
        result.append({
            "id": name,
            "name": name,
            "baseUrl": pdata.get("base_url", ""),
            "enabled": pdata.get("enabled", True),
            "models": pdata.get("models") or [],
            "keys": keys,
            "activeKeys": len(keys),
            "totalKeys": len(keys),
        })
    return result


def _get_config() -> list[dict]:
    """Read real config from config.yaml."""
    home = _hermes_home()
    config = _read_yaml(home / "config.yaml")
    unified = config.get("unified", {}) if config else {}
    
    fields = [
        ("reasoning", "unified.reasoning.enabled", "Foundation", "Plan → critique → execute → reflect"),
        ("reflexion", "unified.reflexion.enabled", "Foundation", "Learn from failures"),
        ("smart_guardian", "unified.smart_guardian.enabled", "Guardians", "LLM-as-judge guardian"),
        ("verifier", "unified.verifier.enabled", "Guardians", "Self-verification loop"),
        ("constitution", "unified.constitution.enabled", "Guardians", "User-defined principles"),
        ("slow_thinking", "unified.slow_thinking.enabled", "Deep Reasoning", "4 levels: fast/balanced/deep/max"),
        ("ensemble", "unified.ensemble.enabled", "Advanced", "Multi-model + judge (3x cost)"),
        ("embedding", "unified.embedding.enabled", "Advanced", "Semantic recall (+40%)"),
        ("user_model", "unified.user_model.enabled", "Advanced", "User profile personalization"),
        ("clarifier", "unified.clarifier.enabled", "Advanced", "Detect ambiguity + ask"),
        ("longrun", "unified.longrun.enabled", "Infrastructure", "Background work queue"),
        ("tool_router", "unified.tool_router.enabled", "Infrastructure", "Auto tool selection"),
        ("cost_tracker", "unified.cost_tracker.enabled", "Infrastructure", "Token accounting + budget"),
        ("response_cache", "unified.response_cache.enabled", "Infrastructure", "Cache LLM responses"),
        ("output_formatter", "unified.output_formatter.enabled", "Infrastructure", "Telegram/Slack formatting"),
        ("skill_registry", "unified.skill_registry.enabled", "Registry", "150 skills marketplace"),
        ("api_registry", "unified.api_registry.enabled", "Registry", "1500+ public APIs"),
        ("multi_provider", "unified.multi_provider.enabled", "Multi-Provider", "Aggregate LLM APIs"),
        ("learning", "unified.learning.enabled", "Learning", "Learn from every interaction"),
        ("skill_synthesis", "unified.skill_synthesis.enabled", "Learning", "Auto-create skills"),
        ("task_planner", "unified.task_planner.enabled", "Planning", "Task decomposition + tracking"),
    ]
    result = []
    for key, path, cat, desc in fields:
        section = unified.get(key, {})
        enabled = section.get("enabled", False) if isinstance(section, dict) else False
        result.append({
            "key": key,
            "path": path,
            "value": enabled,
            "defaultValue": False,
            "enabled": enabled,
            "category": cat,
            "description": desc,
        })
    return result


def _get_costs() -> list[dict]:
    """Read real costs from cost_log.jsonl."""
    home = _hermes_home()
    costs = _read_jsonl(home / "unified" / "cost_log.jsonl", limit=1000)
    # Group by phase
    by_phase: dict[str, dict] = {}
    for c in costs:
        phase = c.get("phase", "unknown")
        if phase not in by_phase:
            by_phase[phase] = {"phase": phase, "tokens": 0, "calls": 0}
        by_phase[phase]["tokens"] += c.get("total_tokens", 0)
        by_phase[phase]["calls"] += 1
    return list(by_phase.values())


def _get_logs() -> list[dict]:
    """Read recent log entries from agent.log."""
    home = _hermes_home()
    log_path = home / "agent.log"
    if not log_path.exists():
        # Fallback: generate from cost_log
        costs = _read_jsonl(home / "unified" / "cost_log.jsonl", limit=20)
        logs = []
        for c in reversed(costs):
            level = "success" if not c.get("cache_hit") else "info"
            logs.append({
                "timestamp": time.strftime("%H:%M:%S", time.localtime(c.get("timestamp", 0))),
                "level": level,
                "module": c.get("phase", "unknown"),
                "message": f"{c.get('phase','?')} call: {c.get('total_tokens',0)} tokens" + (" (cached)" if c.get("cache_hit") else ""),
            })
        return logs
    # Read last 50 lines
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        logs = []
        for line in reversed(lines[-200:]):
            if len(logs) >= 50:
                break
            # Parse log line (format varies, extract what we can)
            level = "info"
            if "ERROR" in line or "error" in line.lower():
                level = "error"
            elif "WARN" in line or "warning" in line.lower():
                level = "warn"
            elif "success" in line.lower() or "✓" in line:
                level = "success"
            
            # Try to extract timestamp
            ts = ""
            if len(line) > 8 and line[2] == ":" and line[5] == ":":
                ts = line[:8]
                line = line[9:]
            
            logs.append({
                "timestamp": ts or time.strftime("%H:%M:%S"),
                "level": level,
                "module": "agent",
                "message": line[:200],
            })
        return logs
    except Exception:
        return []


# ─── Dashboard HTML ─────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes-Omni Control Panel</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#f5f0e8;--card:#faf6ef;--border:#e0d5c3;--text:#3d3528;--dim:#7a6f5c;--muted:#a89a82;--accent:#c8860d;--green:#5a8a3a;--red:#c44d4d;--yellow:#d4a017;--blue:#4a7ba8;--purple:#8b6bb1}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);line-height:1.5;min-height:100vh}
.mono{font-family:'SF Mono',Monaco,monospace}
.header{position:sticky;top:0;z-index:50;background:rgba(245,240,232,.95);backdrop-filter:blur(8px);border-bottom:1px solid var(--border);padding:1rem 1.5rem}
.header-c{max-width:1400px;margin:0 auto;display:flex;align-items:center;justify-content:space-between}
.logo{font-size:1.5rem;font-weight:800;background:linear-gradient(135deg,#fbbf24,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{font-size:.8rem;color:var(--dim)}
.badge{padding:.2rem .6rem;border-radius:9999px;font-size:.75rem;font-weight:600}
.badge-a{background:rgba(200,134,13,.12);color:#c8860d;border:1px solid rgba(200,134,13,.25)}
.badge-g{background:rgba(90,138,58,.12);color:var(--green);border:1px solid rgba(90,138,58,.25)}
.badge-x{background:#f0ebe2;color:var(--dim);border:1px solid var(--border)}
.dot{width:8px;height:8px;background:var(--green);border-radius:50%;display:inline-block;animation:p 2s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.4}}
.tabs{display:flex;gap:.25rem;padding:0 1.5rem;border-bottom:1px solid var(--border);overflow-x:auto;background:var(--bg);position:sticky;top:65px;z-index:40}
.tab{padding:.75rem 1.25rem;cursor:pointer;white-space:nowrap;font-size:.875rem;font-weight:500;color:var(--dim);border-bottom:2px solid transparent;background:none;border:none}
.tab:hover{color:var(--text)}.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.content{flex:1;padding:1.5rem;max-width:1400px;margin:0 auto;width:100%}
.tc{display:none}.tc.active{display:block}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.25rem;margin-bottom:1rem}
.sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1rem;margin-bottom:1.5rem}
.sc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.25rem;position:relative;overflow:hidden}
.sc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.sc.g::before{background:var(--green)}.sc.a::before{background:var(--accent)}.sc.p::before{background:var(--purple)}.sc.b::before{background:var(--blue)}
.si{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;margin-bottom:.5rem}
.si.g{background:rgba(90,138,58,.12)}.si.a{background:rgba(200,134,13,.12)}.si.p{background:rgba(139,107,177,.12)}.si.b{background:rgba(74,123,168,.12)}
.sl{font-size:.75rem;font-weight:600;text-transform:uppercase;color:var(--muted);letter-spacing:.05em}
.sv{font-size:2rem;font-weight:800;margin:.25rem 0}.ss{font-size:.8rem;color:var(--dim)}.sd{font-size:.75rem;color:var(--muted);margin-top:.25rem}
.sd.r{color:var(--red)}
.ai{display:flex;align-items:flex-start;gap:.5rem;padding:.5rem 0;border-bottom:1px solid rgba(224,213,195,.4)}
.ai:last-child{border:none}
.ad{width:8px;height:8px;border-radius:50%;margin-top:6px;flex-shrink:0}
.ad.success{background:var(--green)}.ad.info{background:var(--blue)}.ad.warn{background:var(--yellow)}.ad.error{background:var(--red)}
.al{font-size:.7rem;font-weight:700;text-transform:uppercase;padding:.1rem .4rem;border-radius:4px}
.al.success{background:rgba(90,138,58,.12);color:var(--green)}.al.info{background:rgba(74,123,168,.12);color:var(--blue)}
.al.warn{background:rgba(212,160,23,.12);color:var(--yellow)}.al.error{background:rgba(196,77,77,.12);color:var(--red)}
.am{font-size:.8rem;color:#fbbf24;font-family:monospace}.ax{font-size:.8rem;color:var(--text);flex:1}.at{font-size:.75rem;color:var(--muted);font-family:monospace}
.pc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.25rem;margin-bottom:1rem}
.ph{display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem}
.pn{font-size:1.1rem;font-weight:700}.pu{font-size:.75rem;color:var(--muted);font-family:monospace}
.mb{display:inline-block;background:rgba(200,134,13,.1);color:#c8860d;padding:.15rem .5rem;border-radius:4px;font-size:.7rem;font-family:monospace;margin-right:.25rem;margin-bottom:.25rem}
.kr{display:flex;align-items:center;gap:.5rem;padding:.5rem;border-radius:8px;background:rgba(224,213,195,.3);margin-bottom:.4rem}
.kp{font-family:monospace;font-size:.8rem;color:var(--dim);min-width:120px}
.qb{flex:1;height:8px;background:rgba(224,213,195,.5);border-radius:4px;overflow:hidden}
.qf{height:100%;border-radius:4px;transition:width .3s}
.qf.g{background:var(--green)}.qf.y{background:var(--yellow)}.qf.r{background:var(--red)}
.qt{font-size:.7rem;color:var(--muted);font-family:monospace;min-width:100px;text-align:right}
.ks{font-size:.7rem;padding:.15rem .5rem;border-radius:4px;font-weight:600}
.ks.active{background:rgba(90,138,58,.12);color:var(--green)}.ks.exhausted{background:rgba(196,77,77,.12);color:var(--red)}.ks.disabled{background:rgba(168,154,130,.15);color:var(--muted)}
.cg{margin-bottom:1.5rem}
.ct{font-size:.9rem;font-weight:700;color:#fbbf24;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem}
.cf{display:flex;align-items:center;justify-content:space-between;padding:.6rem .8rem;border-radius:8px;background:rgba(224,213,195,.2);margin-bottom:.3rem}
.cf:hover{background:rgba(224,213,195,.4)}
.cfi{flex:1}.cfp{font-family:monospace;font-size:.75rem;color:var(--muted)}.cfd{font-size:.8rem;color:var(--dim)}
.sk{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:.75rem}
.skc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem}
.skn{font-size:.9rem;font-weight:700}.skr{font-size:.7rem;color:var(--muted);font-family:monospace}
.skde{font-size:.8rem;color:var(--dim);margin:.4rem 0}
.skm{display:flex;align-items:center;gap:.4rem;margin-bottom:.5rem}
.skst{font-size:.7rem;color:var(--yellow)}.skc2{font-size:.65rem;padding:.1rem .4rem;border-radius:4px;background:rgba(200,134,13,.1);color:#c8860d}
.cc{display:flex;align-items:flex-end;gap:.5rem;height:200px;padding:1rem 0;border-bottom:1px solid var(--border);margin-bottom:1rem}
.cb{flex:1;background:linear-gradient(180deg,#c8860d,rgba(200,134,13,.2));border-radius:4px 4px 0 0;position:relative;min-height:4px;transition:height .3s}
.cbl{position:absolute;bottom:-20px;left:50%;transform:translateX(-50%);font-size:.6rem;color:var(--muted);white-space:nowrap}
.cbv{position:absolute;top:-16px;left:50%;transform:translateX(-50%);font-size:.6rem;color:var(--dim)}
.li{display:flex;align-items:flex-start;gap:.5rem;padding:.4rem .6rem;border-radius:6px;margin-bottom:.2rem;font-size:.8rem}
.li:hover{background:rgba(224,213,195,.2)}
.lt{font-family:monospace;color:var(--muted);font-size:.75rem;min-width:60px}
.ll{font-size:.65rem;font-weight:700;text-transform:uppercase;padding:.1rem .35rem;border-radius:3px;min-width:50px;text-align:center}
.ll.info{background:rgba(74,123,168,.12);color:var(--blue)}.ll.warn{background:rgba(212,160,23,.12);color:var(--yellow)}
.ll.error{background:rgba(196,77,77,.12);color:var(--red)}.ll.success{background:rgba(90,138,58,.12);color:var(--green)}
.lm{color:#fbbf24;font-family:monospace;font-size:.75rem;min-width:120px}.lx{color:var(--text);flex:1}
.footer{position:sticky;bottom:0;z-index:50;background:rgba(245,240,232,.95);backdrop-filter:blur(8px);border-top:1px solid var(--border);padding:.6rem 1.5rem;display:flex;align-items:center;justify-content:space-between;font-size:.75rem;color:var(--muted)}
.fs{display:flex;gap:1rem}.fst{display:flex;align-items:center;gap:.3rem}
.search{width:100%;padding:.6rem .8rem;border-radius:8px;background:#fffbf5;border:1px solid var(--border);color:var(--text);font-size:.875rem;margin-bottom:1rem}
.search:focus{outline:none;border-color:var(--accent)}
@media(max-width:768px){.sg{grid-template-columns:1fr}.sk{grid-template-columns:1fr}.header-c{flex-direction:column;gap:.5rem}}
</style></head><body>
<div style="min-height:100vh;display:flex;flex-direction:column">
<header class="header"><div class="header-c">
<div><div class="logo">⚡ Hermes-Omni</div><div class="sub">Control Panel — Cognitive Agent <span id="ver">v3.3</span></div></div>
<div style="display:flex;gap:.5rem">
<span class="badge badge-a">⭐ <span id="bv">v3.3</span></span>
<span class="badge badge-g"><span class="dot"></span> Live</span>
<span class="badge badge-x">⏱ <span id="up">—</span></span>
</div></div></header>
<nav class="tabs" id="tabs">
<button class="tab active" onclick="st(event,'overview')">📊 Tổng quan</button>
<button class="tab" onclick="st(event,'providers')">🔌 Provider</button>
<button class="tab" onclick="st(event,'config')">⚙️ Cấu hình</button>
<button class="tab" onclick="st(event,'skills')">📚 Kỹ năng</button>
<button class="tab" onclick="st(event,'costs')">💰 Chi phí</button>
<button class="tab" onclick="st(event,'logs')">📋 Nhật ký</button>
</nav>
<main class="content">
<div id="overview" class="tc active">
<div class="sg" id="stat-cards"></div>
<div style="display:grid;grid-template-columns:2fr 1fr;gap:1rem">
<div class="card"><div class="ph"><div style="font-weight:600">⚡ Live Activity Feed</div><span style="font-size:.75rem;color:var(--muted)">5 sự kiện gần nhất</span></div><div id="af"></div></div>
<div class="card"><div class="ph"><div style="font-weight:600">🚀 Quick Actions</div></div>
<div style="display:flex;flex-direction:column;gap:.5rem">
<button class="badge badge-g" style="cursor:pointer;border:none;padding:.5rem;border-radius:8px;font-size:.8rem">🟢 Start Gateway</button>
<button class="badge badge-x" style="cursor:pointer;border:none;padding:.5rem;border-radius:8px;font-size:.8rem">🔍 Run Evaluation</button>
<button class="badge badge-x" style="cursor:pointer;border:none;padding:.5rem;border-radius:8px;font-size:.8rem">💬 Open Chat</button>
</div></div></div></div>
<div id="providers" class="tc"><div id="pl"></div></div>
<div id="config" class="tc"><input class="search" placeholder="🔍 Tìm cấu hình..." onkeyup="fc(this.value)"><div id="cl"></div></div>
<div id="skills" class="tc"><input class="search" placeholder="🔍 Tìm kỹ năng..." onkeyup="fs(this.value)"><div id="sl" class="sk"></div></div>
<div id="costs" class="tc"><div id="cs"></div><div class="card"><div class="ph"><div style="font-weight:600">📊 Token Usage by Phase</div></div><div class="cc" id="chart"></div></div></div>
<div id="logs" class="tc"><div class="card" style="max-height:600px;overflow-y:auto"><div id="lf"></div></div></div>
</main>
<footer class="footer"><div class="fs" id="ft"></div><div id="fr">— · v3.3</div></footer>
</div>
<script>
function st(e,id){document.querySelectorAll('.tc').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.getElementById(id).classList.add('active');e.target.closest('.tab').classList.add('active')}
function fmt(n){return n>1e6?(n/1e6).toFixed(2)+'M':n>1e3?(n/1e3).toFixed(0)+'K':n}
async function api(path){try{const r=await fetch('/api/'+path);return await r.json()}catch(e){return null}}
async function refresh(){
const s=await api('status');if(!s)return;
document.getElementById('up').textContent=s.uptime||'—';
document.getElementById('stat-cards').innerHTML=`
<div class="sc g"><div class="si g">🔌</div><div class="sl">Providers</div><div class="sv">${s.providers}</div><div class="ss">${s.activeKeys}/${s.totalKeys} keys active</div><div class="sd">${s.exhaustedKeys} exhausted · ${s.strategy}</div></div>
<div class="sc a"><div class="si a">📚</div><div class="sl">Skills</div><div class="sv">${s.installedSkills}</div><div class="ss">installed</div><div class="sd">${s.totalSkills} total skills</div></div>
<div class="sc p"><div class="si p">🧠</div><div class="sl">Features</div><div class="sv">${s.enabledFeatures}/${s.totalFeatures}</div><div class="ss">enabled</div><div class="sd">${s.totalFeatures-s.enabledFeatures} feature tắt</div></div>
<div class="sc b"><div class="si b">💎</div><div class="sl">Token Usage</div><div class="sv">${fmt(s.totalCostTokens)}</div><div class="ss">${s.totalCalls} calls</div><div class="sd ${s.totalCostTokens>1e6?'r':''}">${s.totalCostTokens>1e6?'⚠ Vượt ngân sách':''}</div></div>`;
document.getElementById('ft').innerHTML=`<span class="fst">🟢 online</span><span class="fst">🔑 ${s.activeKeys}/${s.totalKeys} keys</span><span class="fst">📚 ${s.installedSkills} skills</span><span class="fst">🧠 ${s.enabledFeatures}/${s.totalFeatures} features</span><span class="fst">💰 ${fmt(s.totalCostTokens)} tokens</span>`;
}
async function refreshProviders(){const p=await api('providers');if(!p||!p.length){document.getElementById('pl').innerHTML='<div class="card">Chưa có provider. Chạy: hermes setup</div>';return}
document.getElementById('pl').innerHTML=p.map(pr=>`<div class="pc"><div class="ph"><div><div class="pn">${pr.name}</div><div class="pu">${pr.baseUrl}</div></div></div><div style="margin-bottom:.5rem">${pr.models.map(m=>`<span class="mb">${m}</span>`).join('')}</div>${pr.keys.map(k=>{const pct=k.quota>0?Math.round(k.used/k.quota*100):0;const cls=pct>80?'r':pct>50?'y':'g';const st=k.exhausted?'exhausted':k.enabled?'active':'disabled';return `<div class="kr"><span class="kp">${k.keyPreview}</span><div class="qb"><div class="qf ${cls}" style="width:${Math.max(pct,2)}%"></div></div><span class="qt">${fmt(k.used)} / ${k.quota>0?fmt(k.quota):'∞'}</span><span class="ks ${st}">${st}</span></div>`}).join('')}</div>`).join('')}
async function refreshConfig(){const c=await api('config');if(!c)return;const groups={};c.forEach(f=>{if(!groups[f.category])groups[f.category]=[];groups[f.category].push(f)});let html='';for(const[cat,fields]of Object.entries(groups)){const en=fields.filter(f=>f.enabled).length;html+=`<div class="cg"><div class="ct">${cat} (${en}/${fields.length})</div>${fields.map(f=>`<div class="cf"><div class="cfi"><div class="cfp">${f.path}</div><div class="cfd">${f.description}</div></div><span style="font-size:.8rem;color:${f.enabled?'var(--green)':'var(--muted)'}">${f.enabled?'✓ ON':'✗ OFF'}</span></div>`).join('')}</div>`}document.getElementById('cl').innerHTML=html}
async function refreshSkills(){const s=await api('skills');if(!s)return;document.getElementById('sl').innerHTML=s.map(sk=>`<div class="skc" style="${sk.installed?'border-color:rgba(34,197,94,.3)':''}"><div class="skn">${sk.id}</div><div class="skr">${sk.repo}</div><div class="skde">${sk.desc}</div><div class="skm"><span class="skst">⭐ ${sk.stars}</span><span class="skc2">${sk.category}</span></div></div>`).join('')}
async function refreshCosts(){const c=await api('costs');if(!c)return;const max=Math.max(...c.map(x=>x.tokens),1);document.getElementById('cs').innerHTML=`<div class="sg"><div class="sc a"><div class="sl">Total Tokens</div><div class="sv">${fmt(c.reduce((s,x)=>s+x.tokens,0))}</div></div><div class="sc b"><div class="sl">Total Calls</div><div class="sv">${c.reduce((s,x)=>s+x.calls,0)}</div></div></div>`;document.getElementById('chart').innerHTML=c.map(x=>{const h=Math.round(x.tokens/max*180);return `<div class="cb" style="height:${h}px"><div class="cbv">${fmt(x.tokens)}</div><div class="cbl">${x.phase}</div></div>`}).join('')}
async function refreshLogs(){const l=await api('logs');if(!l)return;document.getElementById('af').innerHTML=l.slice(0,5).map(x=>`<div class="ai"><div class="ad ${x.level}"></div><span class="al ${x.level}">${x.level}</span><span class="am">${x.module}</span><span class="ax">${x.message}</span><span class="at">${x.timestamp}</span></div>`).join('');document.getElementById('lf').innerHTML=l.map(x=>`<div class="li"><span class="lt">${x.timestamp}</span><span class="ll ${x.level}">${x.level}</span><span class="lm">${x.module}</span><span class="lx">${x.message}</span></div>`).join('')}
function fc(q){q=q.toLowerCase();document.querySelectorAll('.cf').forEach(f=>{f.style.display=f.textContent.toLowerCase().includes(q)?'':'none'})}
function fs(q){q=q.toLowerCase();document.querySelectorAll('.skc').forEach(c=>{c.style.display=c.textContent.toLowerCase().includes(q)?'':'none'})}
refresh();refreshProviders();refreshConfig();refreshSkills();refreshCosts();refreshLogs();
setInterval(refresh,10000);setInterval(refreshLogs,3000);
</script></body></html>"""


# ─── HTTP Server ────────────────────────────────────────────────────────────


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_html(DASHBOARD_HTML)
        elif path.startswith("/api/status"):
            self._send_json(_get_status())
        elif path.startswith("/api/providers"):
            self._send_json(_get_providers())
        elif path.startswith("/api/config"):
            self._send_json(_get_config())
        elif path.startswith("/api/skills"):
            self._send_json(_scan_skills())
        elif path.startswith("/api/costs"):
            self._send_json(_get_costs())
        elif path.startswith("/api/logs"):
            self._send_json(_get_logs())
        else:
            self._send_json({"error": "not found"}, 404)

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
        pass  # suppress


def run_server(port: int = 8788) -> None:
    """Run the real-time dashboard server.

    Usage:
        python -m agent.unified.dashboard_server --port 8788
    """
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"═══ Hermes-Omni Dashboard ═══")
    print(f"  URL:    http://localhost:{port}")
    print(f"  Data:   {_hermes_home()}")
    print(f"  Skills: {_repo_root() / 'skills' / 'local-repos'}")
    print(f"  Refresh: stats 10s, logs 3s")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hermes-Omni Dashboard Server")
    parser.add_argument("--port", type=int, default=8788, help="Server port")
    args = parser.parse_args()
    run_server(port=args.port)
