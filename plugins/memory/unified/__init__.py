"""Bundled unified reflexion memory provider."""

from agent.unified.memory_provider import UnifiedReflexionMemoryProvider


def register(ctx):
    ctx.register_memory_provider(UnifiedReflexionMemoryProvider())
