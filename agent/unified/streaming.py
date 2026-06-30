"""Streaming LLM Support — async generator yielding tokens.

WHY THIS EXISTS
---------------
Without streaming, slow_thinking (max = 8 rounds, ~15K tokens) blocks
the user 30s-2min. User thinks agent is stuck, cancels, agent dies.

With streaming, user sees tokens appear in real-time → knows agent is
working → doesn't cancel. Top-tier agents all stream.

ARCHITECTURE
------------
This module provides:
1. `llm_stream(system, user) -> AsyncGenerator[str, None]` — yields tokens
2. `stream_slow_thinking(request, level) -> AsyncGenerator[str, None]`
3. Callback-based streaming for sync callers

The actual streaming depends on the LLM provider (OpenAI, Anthropic,
GLM). This module adapts the agent's existing client to a streaming
interface. If the client doesn't support streaming, it degrades to
batch (collect all tokens, yield at once).

USAGE
-----
    # Async:
    async for token in llm_stream(system, user, agent):
        print(token, end="", flush=True)

    # Sync (via callback):
    def on_token(token: str):
        print(token, end="", flush=True)
    full_response = stream_with_callback(system, user, agent, on_token)
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, Generator


# --------------------------------------------------------------------------- #
# Streaming LLM call
# --------------------------------------------------------------------------- #


def llm_stream(
    system: str,
    user: str,
    agent: Any,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Generator[str, None, str]:
    """Stream LLM tokens as a generator.

    Yields tokens (str) as they arrive. Returns the full response when done.

    If the agent's client doesn't support streaming, falls back to batch
    (yields the full response at once).

    Args:
        system: system prompt
        user: user prompt
        agent: AIAgent instance (must have .client and .model)
        model: override agent.model
        temperature: sampling temperature
        max_tokens: max tokens to generate

    Yields:
        Token strings (may be 1 char or 1 word depending on provider)

    Returns:
        Full response string (also available via StopIteration value)
    """
    client = getattr(agent, "client", None) or getattr(agent, "_client", None)
    model_name = model or getattr(agent, "model", None)
    if client is None or model_name is None:
        # No client — yield empty and return empty.
        return ""

    try:
        # Try OpenAI-compatible streaming.
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        full_response = []
        for chunk in stream:
            try:
                # OpenAI streaming chunk format.
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None) or ""
                    if content:
                        full_response.append(content)
                        yield content
                elif isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_response.append(content)
                            yield content
            except Exception:
                continue
        return "".join(full_response)
    except Exception:
        # Fallback: batch call, yield all at once.
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            if content:
                yield content
            return content
        except Exception:
            return ""


def stream_with_callback(
    system: str,
    user: str,
    agent: Any,
    callback: Callable[[str], None],
    **kwargs: Any,
) -> str:
    """Stream LLM tokens, calling callback for each. Returns full response.

    Sync wrapper around llm_stream for callers that want callback-based
    streaming (e.g., TUI updating a progress display).
    """
    full = ""
    try:
        gen = llm_stream(system, user, agent, **kwargs)
        try:
            while True:
                token = next(gen)
                if token:
                    full += token
                    try:
                        callback(token)
                    except Exception:
                        pass
        except StopIteration as e:
            # Generator return value.
            ret = e.value
            if ret and not full:
                full = ret
                try:
                    callback(ret)
                except Exception:
                    pass
    except Exception:
        pass
    return full


# --------------------------------------------------------------------------- #
# Streaming slow thinking
# --------------------------------------------------------------------------- #


def stream_slow_thinking_round(
    *,
    phase: str,
    system: str,
    user: str,
    agent: Any,
    callback: Callable[[str, str], None] | None = None,
    # callback(phase, token)
) -> str:
    """Stream one round of slow thinking.

    Args:
        phase: "decompose" | "analyze" | "synthesize" | "critique" | "refine" | "final"
        system: system prompt for this phase
        user: user prompt for this phase
        agent: AIAgent instance
        callback: optional callable(phase, token) called per token

    Returns:
        Full response for this round.
    """
    full = ""

    def _on_token(token: str) -> None:
        nonlocal full
        full += token
        if callback is not None:
            try:
                callback(phase, token)
            except Exception:
                pass

    stream_with_callback(system, user, agent, _on_token)
    return full


# --------------------------------------------------------------------------- #
# Async wrapper (for async callers)
# --------------------------------------------------------------------------- #


async def async_llm_stream(
    system: str,
    user: str,
    agent: Any,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Async wrapper around llm_stream.

    Yields tokens as they arrive from a background thread running the
    sync stream.
    """
    import queue

    token_queue: queue.Queue[str | None] = queue.Queue()
    full_response = {"value": ""}

    def _producer() -> None:
        try:
            gen = llm_stream(system, user, agent, **kwargs)
            try:
                while True:
                    token = next(gen)
                    if token:
                        token_queue.put(token)
            except StopIteration as e:
                ret = e.value
                if ret:
                    full_response["value"] = ret
        except Exception:
            pass
        finally:
            token_queue.put(None)  # sentinel

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        try:
            # Non-blocking get with timeout so we can yield control.
            token = await asyncio.get_event_loop().run_in_executor(
                None, lambda: token_queue.get(timeout=0.1)
            )
        except Exception:
            await asyncio.sleep(0.01)
            continue
        if token is None:
            break
        yield token

    thread.join(timeout=1.0)


# --------------------------------------------------------------------------- #
# Stream status helper
# --------------------------------------------------------------------------- #


class StreamStatus:
    """Track streaming progress for UI display."""

    def __init__(self) -> None:
        self.phase: str = ""
        self.tokens_received: int = 0
        self.started_at: float = 0.0
        self.last_token_at: float = 0.0

    def start_phase(self, phase: str) -> None:
        self.phase = phase
        self.tokens_received = 0
        self.started_at = time.time()
        self.last_token_at = self.started_at

    def on_token(self, phase: str, token: str) -> None:
        if phase != self.phase:
            self.start_phase(phase)
        self.tokens_received += 1
        self.last_token_at = time.time()

    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        return time.time() - self.started_at

    def tokens_per_second(self) -> float:
        elapsed = self.elapsed_seconds()
        if elapsed <= 0:
            return 0.0
        return self.tokens_received / elapsed

    def status_line(self) -> str:
        tps = self.tokens_per_second()
        return f"[{self.phase}] {self.tokens_received} tokens, {tps:.1f} tok/s, {self.elapsed_seconds():.1f}s"
