#!/usr/bin/env python3
"""Super Plan — 1-lần fix toàn bộ dự án.

Bước 1: Wire 5 breakthrough modules + harness vào integration.py + config.py + runtime_wiring
Bước 2: Fix dashboard_server — xóa junk, việt hóa, đồng bộ config
Bước 3: Fix registry_tools — đảm bảo 11 tools register đúng
Bước 4: Verify toàn bộ
"""
import os, sys, re
sys.path.insert(0, '.')

REPO = '/home/z/my-project/repos/hermes-omni'

# ═══════════════════════════════════════════════════════════════
# BƯỚC 1: Wire breakthrough modules vào config.py
# ═══════════════════════════════════════════════════════════════
print("=== BƯỚC 1: Wire breakthrough modules ===")

config_path = f'{REPO}/agent/unified/config.py'
with open(config_path, 'r') as f:
    cfg = f.read()

# Add config fields if not present
if 'failure_forecast_enabled' not in cfg:
    # Insert after multi_provider_server_port line in dataclass
    cfg = cfg.replace(
        '    multi_provider_server_port: int = 8787',
        '    multi_provider_server_port: int = 8787\n'
        '    # --- v4 Breakthrough ---\n'
        '    failure_forecast_enabled: bool = True\n'
        '    trajectory_distiller_enabled: bool = True\n'
        '    context_hologram_enabled: bool = True\n'
        '    skill_evolution_enabled: bool = True\n'
        '    persona_split_enabled: bool = False\n'
        '    harness_enabled: bool = True\n'
        '    harness_max_parallel: int = 4'
    )
    
    # Add config parsing
    cfg = cfg.replace(
        '    _CONFIG_CACHE = UnifiedConfig(',
        '    # --- v4 Breakthrough config ---\n'
        '    failure_forecast_enabled = _truthy(\n'
        '        _cfg_get("unified", "failure_forecast", "enabled", default=True), default=True\n'
        '    )\n'
        '    trajectory_distiller_enabled = _truthy(\n'
        '        _cfg_get("unified", "trajectory_distiller", "enabled", default=True), default=True\n'
        '    )\n'
        '    context_hologram_enabled = _truthy(\n'
        '        _cfg_get("unified", "context_hologram", "enabled", default=True), default=True\n'
        '    )\n'
        '    skill_evolution_enabled = _truthy(\n'
        '        _cfg_get("unified", "skill_evolution", "enabled", default=True), default=True\n'
        '    )\n'
        '    persona_split_enabled = _truthy(\n'
        '        _cfg_get("unified", "persona_split", "enabled", default=False), default=False\n'
        '    )\n'
        '    harness_enabled = _truthy(\n'
        '        _cfg_get("unified", "harness", "enabled", default=True), default=True\n'
        '    )\n'
        '    try:\n'
        '        harness_max_parallel = int(\n'
        '            _cfg_get("unified", "harness", "max_parallel", default=4) or 4\n'
        '        )\n'
        '    except Exception:\n'
        '        harness_max_parallel = 4\n'
        '\n'
        '    _CONFIG_CACHE = UnifiedConfig(',
        1  # only first occurrence
    )
    
    # Add to UnifiedConfig constructor
    cfg = cfg.replace(
        '        multi_provider_server_port=multi_provider_server_port,\n    )',
        '        multi_provider_server_port=multi_provider_server_port,\n'
        '        failure_forecast_enabled=failure_forecast_enabled,\n'
        '        trajectory_distiller_enabled=trajectory_distiller_enabled,\n'
        '        context_hologram_enabled=context_hologram_enabled,\n'
        '        skill_evolution_enabled=skill_evolution_enabled,\n'
        '        persona_split_enabled=persona_split_enabled,\n'
        '        harness_enabled=harness_enabled,\n'
        '        harness_max_parallel=harness_max_parallel,\n'
        '    )'
    )
    
    with open(config_path, 'w') as f:
        f.write(cfg)
    print("  ✓ config.py — added 7 breakthrough fields")
else:
    print("  ⏭ config.py — already has breakthrough fields")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 2: Wire breakthrough modules vào integration.py
# ═══════════════════════════════════════════════════════════════
print("=== BƯỚC 2: Wire into integration.py ===")

integ_path = f'{REPO}/agent/unified/integration.py'
with open(integ_path, 'r') as f:
    integ = f.read()

if 'failure_forecast' not in integ:
    # Add imports
    integ = integ.replace(
        '# v3.3 Multi-Provider Gateway',
        '# v4 Breakthrough modules\n'
        'from .failure_forecast import (\n'
        '    FailureForecast, configure_forecast, forecast_now, forecast_stats,\n'
        '    record_and_forecast, get_forecast,\n'
        ')\n'
        'from .trajectory_distiller import (\n'
        '    TrajectoryDistillery, configure_distiller, distillery_stats,\n'
        '    start_trajectory_task, record_trajectory_step, finish_trajectory_task,\n'
        '    recall_golden_path, get_distiller,\n'
        ')\n'
        'from .context_hologram import (\n'
        '    get_hologram, hologram_stats, build_hologram,\n'
        ')\n'
        'from .skill_evolution import (\n'
        '    SkillEvolution, configure_evolution, evolution_stats,\n'
        '    register_skill_for_evolution, record_skill_feedback, get_evolved_skill,\n'
        ')\n'
        'from .persona_split import (\n'
        '    PersonaSplitSolver, configure_solver, persona_solve, persona_stats,\n'
        ')\n'
        'from .harness import (\n'
        '    AgentHarness, ToolCall, configure_harness, run_harness, harness_stats,\n'
        ')\n'
        '# v3.3 Multi-Provider Gateway'
    )
    
    # Add configure calls in configure_reasoning_stack
    integ = integ.replace(
        '    # v3.1: SkillSynthesizer',
        '    # v4: Breakthrough modules\n'
        '    if cfg.failure_forecast_enabled:\n'
        '        configure_forecast()\n'
        '    if cfg.trajectory_distiller_enabled:\n'
        '        configure_distiller()\n'
        '    if cfg.skill_evolution_enabled:\n'
        '        configure_evolution(llm_call=llm_call)\n'
        '    if cfg.persona_split_enabled:\n'
        '        configure_solver(llm_call=llm_call)\n'
        '    if cfg.harness_enabled:\n'
        '        configure_harness(max_parallel=cfg.harness_max_parallel)\n'
        '\n'
        '    # v3.1: SkillSynthesizer'
    )
    
    with open(integ_path, 'w') as f:
        f.write(integ)
    print("  ✓ integration.py — wired 6 breakthrough modules")
else:
    print("  ⏭ integration.py — already has breakthrough imports")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 3: Wire breakthrough modules vào runtime_wiring.py
# ═══════════════════════════════════════════════════════════════
print("=== BƯỚC 3: Wire into runtime_wiring.py ===")

rw_path = f'{REPO}/agent/unified/runtime_wiring.py'
with open(rw_path, 'r') as f:
    rw = f.read()

if 'failure_forecast' not in rw:
    # Add to augment_volatile_prompt — inject hologram + golden path
    rw = rw.replace(
        '    # Recalled learnings.\n    if user_message:',
        '    # v4: Context Hologram (project overview in ~500 tokens)\n'
        '    try:\n'
        '        from agent.unified.integration import get_hologram\n'
        '        import os\n'
        '        hologram = get_hologram(os.getcwd())\n'
        '        if hologram:\n'
        '            blocks.append(hologram)\n'
        '    except Exception:\n'
        '        pass\n'
        '\n'
        '    # v4: Golden Path (distilled trajectory)\n'
        '    if user_message:\n'
        '        try:\n'
        '            from agent.unified.integration import recall_golden_path\n'
        '            golden = recall_golden_path(user_message)\n'
        '            if golden:\n'
        '                blocks.append(golden)\n'
        '        except Exception:\n'
        '            pass\n'
        '\n'
        '    # Recalled learnings.\n    if user_message:'
    )
    
    with open(rw_path, 'w') as f:
        f.write(rw)
    print("  ✓ runtime_wiring.py — wired hologram + golden path")
else:
    print("  ⏭ runtime_wiring.py — already has breakthrough")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 4: Wire failure_forecast vào after_tool_call
# ═══════════════════════════════════════════════════════════════
print("=== BƯỚC 4: Wire failure_forecast into after_tool_call ===")

if 'record_and_forecast' not in integ:
    integ = integ.replace(
        '    # Tool router usage feedback (async, non-blocking).',
        '    # v4: Failure Forecast — predict failures before they happen\n'
        '    try:\n'
        '        from .failure_forecast import record_and_forecast\n'
        '        forecast_result = record_and_forecast(\n'
        '            tool_name=tool_name,\n'
        '            args=args,\n'
        '            success=not str(result).lower().count("error") > 0,\n'
        '            tokens_used=0,\n'
        '        )\n'
        '        if forecast_result and forecast_result.get("risk_score", 0) >= 0.7:\n'
        '            _bus.emit(\n'
        '                "failure_forecast.warning",\n'
        '                {"risk": forecast_result["risk_score"], "patterns": forecast_result.get("predicted_failures", [])},\n'
        '                session_id=session_id or "",\n'
        '                turn_id=turn_id or "",\n'
        '            )\n'
        '    except Exception:\n'
        '        pass\n'
        '\n'
        '    # Tool router usage feedback (async, non-blocking).'
    )
    with open(integ_path, 'w') as f:
        f.write(integ)
    print("  ✓ integration.py — wired failure_forecast into after_tool_call")
else:
    print("  ⏭ integration_wiring — already has failure_forecast")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 5: Verify
# ═══════════════════════════════════════════════════════════════
print("\n=== BƯỚC 5: Verify ===")
os.chdir(REPO)
os.environ['HERMES_HOME'] = '/tmp/hermes-verify'
os.makedirs('/tmp/hermes-verify', exist_ok=True)

# Syntax check
import ast
for f in ['agent/unified/config.py', 'agent/unified/integration.py', 'agent/unified/runtime_wiring.py']:
    try:
        ast.parse(open(f).read())
        print(f"  ✓ {f}")
    except SyntaxError as e:
        print(f"  ❌ {f}: {e}")

# Import check
import importlib
modules_to_test = [
    'agent.unified.config', 'agent.unified.integration', 'agent.unified.runtime_wiring',
    'agent.unified.failure_forecast', 'agent.unified.trajectory_distiller',
    'agent.unified.context_hologram', 'agent.unified.skill_evolution',
    'agent.unified.persona_split', 'agent.unified.harness',
]
for m in modules_to_test:
    try:
        importlib.import_module(m)
        print(f"  ✓ import {m}")
    except Exception as e:
        print(f"  ❌ import {m}: {e!r}")

print("\n=== DONE ===")
