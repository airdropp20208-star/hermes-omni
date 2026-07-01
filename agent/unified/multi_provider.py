"""Multi-Provider Gateway — aggregate many LLM APIs into 1 endpoint.

THE PROBLEM
-----------
User có nhiều API keys lẻ (mỗi account $2 free credit) từ nhiều providers
(GLM, OpenRouter, DeepSeek, Kimi, Qwen, v.v.). Mỗi key hết nhanh.

freellmapi (https://github.com/tashfeenahmed/freellmapi) giải quyết bằng
cách proxy + round-robin. Nhưng nó là standalone server, không tích hợp
vào agent.

MultiProviderGateway tích hợp trực tiếp vào Hermes-Omni:
1. **Multiple keys per provider** — round-robin giữa nhiều keys
2. **Failover** — provider A fail → tự switch sang provider B
3. **Cost tracking** — đếm tokens per key, disable key khi hết quota
4. **Load balancing** — round-robin + least-used strategies
5. **OpenAI-compatible endpoint** — agent gọi 1 endpoint, gateway route

ARCHITECTURE
------------
```
   Agent (OpenAI client)
       │
       ▼
   MultiProviderGateway (localhost:8787)
       │
       ├── Provider: glm (key1, key2, key3)  ← round-robin
       ├── Provider: openrouter (key1, key2)
       ├── Provider: deepseek (key1)
       ├── Provider: kimi (key1, key2)
       └── Provider: qwen (key1)
       │
       ▼
   Failover: glm fail → openrouter → deepseek → kimi → qwen
```

CONFIG
------
~/.hermes/multi_provider.yaml:

    providers:
      glm:
        base_url: https://open.bigmodel.cn/api/paas/v4
        keys:
          - key: "abc123..."
            quota_tokens: 2000000
          - key: "def456..."
            quota_tokens: 2000000
        models: [glm-4.6, glm-4.5-air]
      openrouter:
        base_url: https://openrouter.ai/api/v1
        keys:
          - key: "sk-or-v1-xxx"
        models: [anthropic/claude-3.5-sonnet, openai/gpt-4o]
      deepseek:
        base_url: https://api.deepseek.com/v1
        keys:
          - key: "sk-xxx"
        models: [deepseek-chat, deepseek-reasoner]

    strategy: round-robin  # round-robin | least-used | failover-only
    failover_order: [glm, openrouter, deepseek, kimi, qwen]

USAGE
-----
    # Start gateway (background)
    hermes multi-provider start

    # Use in agent (config.yaml):
    model:
      provider: custom
      base_url: http://localhost:8787/v1
      api_key: "any"  # gateway doesn't require key

    # Or directly:
    from agent.unified.multi_provider import MultiProviderGateway
    gateway = MultiProviderGateway()
    response = gateway.chat("What is 2+2?", model="glm-4.6")
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class APIKey:
    """One API key with quota tracking."""

    key: str
    quota_tokens: int = 0  # 0 = unlimited
    used_tokens: int = 0
    enabled: bool = True
    last_used: float = 0.0
    error_count: int = 0
    last_error: str = ""

    @property
    def remaining(self) -> int:
        if self.quota_tokens == 0:
            return 999_999_999
        return max(0, self.quota_tokens - self.used_tokens)

    @property
    def exhausted(self) -> bool:
        return self.quota_tokens > 0 and self.used_tokens >= self.quota_tokens


@dataclass
class Provider:
    """One LLM provider with multiple keys."""

    name: str
    base_url: str
    keys: list[APIKey] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    enabled: bool = True

    def active_keys(self) -> list[APIKey]:
        return [k for k in self.keys if k.enabled and not k.exhausted]


@dataclass
class ChatResult:
    """Result of a chat completion call."""

    success: bool
    content: str = ""
    provider: str = ""
    key_index: int = -1
    model: str = ""
    tokens_used: int = 0
    error: str = ""
    elapsed_ms: int = 0


# --------------------------------------------------------------------------- #
# MultiProviderGateway
# --------------------------------------------------------------------------- #


class MultiProviderGateway:
    """Aggregate multiple LLM providers + keys into one interface.

    Thread-safe. Round-robin per provider, failover across providers.
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        strategy: Literal["round-robin", "least-used", "failover-only"] = "round-robin",
        failover_order: list[str] | None = None,
    ) -> None:
        if config_path is None:
            from hermes_constants import get_hermes_home

            config_path = get_hermes_home() / "multi_provider.yaml"
        self._config_path = Path(config_path).expanduser()
        self._strategy = strategy
        self._failover_order = failover_order or []
        self._providers: dict[str, Provider] = {}
        self._rr_index: dict[str, int] = defaultdict(int)  # round-robin counter
        self._lock = threading.RLock()
        self._load_config()

    # ------------------------------------------------------------------ #
    # Config loading
    # ------------------------------------------------------------------ #

    def _load_config(self) -> None:
        """Load provider config from YAML."""
        if not self._config_path.exists():
            return
        try:
            import yaml

            data = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not data or not isinstance(data, dict):
            return
        # Strategy + failover order.
        self._strategy = data.get("strategy", self._strategy)
        self._failover_order = data.get("failover_order", self._failover_order)
        # Providers.
        for name, pdata in (data.get("providers") or {}).items():
            if not isinstance(pdata, dict):
                continue
            keys = []
            for kdata in pdata.get("keys") or []:
                if isinstance(kdata, dict):
                    keys.append(
                        APIKey(
                            key=str(kdata.get("key", "")),
                            quota_tokens=int(kdata.get("quota_tokens", 0) or 0),
                        )
                    )
                elif isinstance(kdata, str):
                    keys.append(APIKey(key=kdata))
            self._providers[name] = Provider(
                name=name,
                base_url=str(pdata.get("base_url", "")),
                keys=keys,
                models=[str(m) for m in pdata.get("models") or []],
                enabled=bool(pdata.get("enabled", True)),
            )

    def reload(self) -> None:
        """Reload config from disk."""
        with self._lock:
            self._providers.clear()
            self._rr_index.clear()
            self._load_config()

    # ------------------------------------------------------------------ #
    # Provider management
    # ------------------------------------------------------------------ #

    def add_provider(
        self,
        name: str,
        base_url: str,
        keys: list[str | dict[str, Any]],
        models: list[str] | None = None,
    ) -> None:
        """Add a provider programmatically."""
        with self._lock:
            api_keys = []
            for k in keys:
                if isinstance(k, str):
                    api_keys.append(APIKey(key=k))
                elif isinstance(k, dict):
                    api_keys.append(
                        APIKey(
                            key=str(k.get("key", "")),
                            quota_tokens=int(k.get("quota_tokens", 0) or 0),
                        )
                    )
            self._providers[name] = Provider(
                name=name,
                base_url=base_url,
                keys=api_keys,
                models=models or [],
            )

    def add_key(self, provider: str, key: str, quota_tokens: int = 0) -> bool:
        """Add an API key to a provider."""
        with self._lock:
            p = self._providers.get(provider)
            if p is None:
                return False
            p.keys.append(APIKey(key=key, quota_tokens=quota_tokens))
            return True

    def remove_key(self, provider: str, key: str) -> bool:
        """Remove a key from a provider."""
        with self._lock:
            p = self._providers.get(provider)
            if p is None:
                return False
            before = len(p.keys)
            p.keys = [k for k in p.keys if k.key != key]
            return len(p.keys) < before

    def list_providers(self) -> list[dict[str, Any]]:
        """List all providers with their key stats."""
        with self._lock:
            result = []
            for p in self._providers.values():
                result.append(
                    {
                        "name": p.name,
                        "base_url": p.base_url,
                        "enabled": p.enabled,
                        "models": p.models,
                        "keys": [
                            {
                                "key_preview": k.key[:8] + "..." + k.key[-4:] if len(k.key) > 12 else k.key[:4] + "***",
                                "quota": k.quota_tokens,
                                "used": k.used_tokens,
                                "remaining": k.remaining,
                                "exhausted": k.exhausted,
                                "enabled": k.enabled,
                                "error_count": k.error_count,
                            }
                            for k in p.keys
                        ],
                        "active_keys": len(p.active_keys()),
                        "total_keys": len(p.keys),
                    }
                )
            return result

    # ------------------------------------------------------------------ #
    # Key selection
    # ------------------------------------------------------------------ #

    def _select_key(self, provider: Provider) -> tuple[APIKey, int] | None:
        """Select a key from provider based on strategy."""
        active = provider.active_keys()
        if not active:
            return None
        if self._strategy == "least-used":
            # Pick key with least used_tokens.
            idx = min(range(len(active)), key=lambda i: active[i].used_tokens)
            return active[idx], idx
        elif self._strategy == "failover-only":
            # Always pick first active key.
            return active[0], 0
        else:  # round-robin (default)
            with self._lock:
                rr = self._rr_index[provider.name]
                self._rr_index[provider.name] = (rr + 1) % len(active)
                return active[rr % len(active)], rr % len(active)

    def _get_provider_order(self, preferred: str = "") -> list[Provider]:
        """Get providers in failover order."""
        with self._lock:
            all_providers = [p for p in self._providers.values() if p.enabled and p.active_keys()]
        if preferred and preferred in self._providers:
            p = self._providers[preferred]
            if p.enabled and p.active_keys():
                all_providers = [p] + [x for x in all_providers if x.name != preferred]
        elif self._failover_order:
            ordered = []
            for name in self._failover_order:
                p = self._providers.get(name)
                if p and p.enabled and p.active_keys():
                    ordered.append(p)
            # Add any not in failover_order.
            for p in all_providers:
                if p.name not in self._failover_order:
                    ordered.append(p)
            all_providers = ordered
        return all_providers

    # ------------------------------------------------------------------ #
    # Chat completion
    # ------------------------------------------------------------------ #

    def chat(
        self,
        message: str,
        *,
        model: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        preferred_provider: str = "",
        timeout: int = 60,
    ) -> ChatResult:
        """Send a chat completion request with automatic failover.

        Args:
            message: user message
            model: model name (e.g., "glm-4.6"). If empty, uses first available.
            system: optional system prompt
            temperature: sampling temperature
            max_tokens: max tokens to generate
            preferred_provider: try this provider first (e.g., "glm")
            timeout: request timeout in seconds

        Returns:
            ChatResult with content + metadata
        """
        providers = self._get_provider_order(preferred_provider)
        if not providers:
            return ChatResult(success=False, error="No providers with active keys")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        last_error = ""
        for provider in providers:
            # Determine model for this provider.
            use_model = model
            if not use_model and provider.models:
                use_model = provider.models[0]
            if not use_model:
                use_model = "gpt-3.5-turbo"  # fallback

            key_result = self._select_key(provider)
            if key_result is None:
                continue
            api_key, key_idx = key_result

            started = time.time()
            try:
                response_data = self._call_provider(
                    provider=provider,
                    api_key=api_key,
                    model=use_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                elapsed = int((time.time() - started) * 1000)

                # Extract content + usage.
                content = ""
                tokens_used = 0
                choices = response_data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                usage = response_data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)

                # Update key stats.
                with self._lock:
                    api_key.used_tokens += tokens_used
                    api_key.last_used = time.time()

                return ChatResult(
                    success=True,
                    content=content,
                    provider=provider.name,
                    key_index=key_idx,
                    model=use_model,
                    tokens_used=tokens_used,
                    elapsed_ms=elapsed,
                )
            except HTTPError as exc:
                elapsed = int((time.time() - started) * 1000)
                last_error = f"{provider.name}: HTTP {exc.code} {exc.reason}"
                with self._lock:
                    api_key.error_count += 1
                    api_key.last_error = last_error
                    # Disable key after 5 errors.
                    if api_key.error_count >= 5:
                        api_key.enabled = False
                # Continue to next provider.
                continue
            except (URLError, Exception) as exc:
                elapsed = int((time.time() - started) * 1000)
                last_error = f"{provider.name}: {exc!r}"
                with self._lock:
                    api_key.error_count += 1
                    api_key.last_error = last_error
                continue

        return ChatResult(success=False, error=f"All providers failed. Last: {last_error}")

    def _call_provider(
        self,
        *,
        provider: Provider,
        api_key: APIKey,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> dict[str, Any]:
        """Make HTTP request to provider's API (OpenAI-compatible)."""
        url = f"{provider.base_url.rstrip('/')}/chat/completions"
        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        req = Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.key}",
            },
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))

    # ------------------------------------------------------------------ #
    # Stats + management
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Get gateway stats."""
        with self._lock:
            total_keys = sum(len(p.keys) for p in self._providers.values())
            active_keys = sum(len(p.active_keys()) for p in self._providers.values())
            total_used = sum(k.used_tokens for p in self._providers.values() for k in p.keys)
            total_quota = sum(k.quota_tokens for p in self._providers.values() for k in p.keys)
            return {
                "providers": len(self._providers),
                "total_keys": total_keys,
                "active_keys": active_keys,
                "exhausted_keys": total_keys - active_keys,
                "total_used_tokens": total_used,
                "total_quota_tokens": total_quota,
                "strategy": self._strategy,
                "failover_order": self._failover_order,
                "config_path": str(self._config_path),
            }

    def reset_usage(self) -> None:
        """Reset all usage counters (e.g., monthly reset)."""
        with self._lock:
            for p in self._providers.values():
                for k in p.keys:
                    k.used_tokens = 0
                    k.error_count = 0
                    k.enabled = True
                    k.last_error = ""

    def save_config(self) -> None:
        """Save current config to YAML."""
        try:
            import yaml

            data = {
                "strategy": self._strategy,
                "failover_order": self._failover_order,
                "providers": {},
            }
            for p in self._providers.values():
                data["providers"][p.name] = {
                    "base_url": p.base_url,
                    "enabled": p.enabled,
                    "models": p.models,
                    "keys": [
                        {"key": k.key, "quota_tokens": k.quota_tokens}
                        for k in p.keys
                    ],
                }
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# OpenAI-compatible HTTP server (optional, for standalone use)
# --------------------------------------------------------------------------- #


def run_server(port: int = 8787, *, config_path: str | Path | None = None) -> None:
    """Run OpenAI-compatible HTTP server.

    This lets any OpenAI client point to localhost:8787 and get
    multi-provider failover automatically.

    Usage:
        python -m agent.unified.multi_provider --serve --port 8787

    Then in agent config:
        model:
          provider: custom
          base_url: http://localhost:8787/v1
          api_key: "any"
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    gateway = MultiProviderGateway(config_path=config_path)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if "/chat/completions" not in self.path:
                self.send_error(404, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body)
                messages = data.get("messages", [])
                user_msg = ""
                system_msg = ""
                for m in messages:
                    if m.get("role") == "user":
                        user_msg = m.get("content", "")
                    elif m.get("role") == "system":
                        system_msg = m.get("content", "")
                model = data.get("model", "")
                result = gateway.chat(
                    user_msg,
                    model=model,
                    system=system_msg,
                    temperature=data.get("temperature", 0.7),
                    max_tokens=data.get("max_tokens", 4096),
                )
                if result.success:
                    response = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": result.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": result.content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": result.tokens_used,
                            "total_tokens": result.tokens_used,
                        },
                        "x_provider": result.provider,
                        "x_key_index": result.key_index,
                    }
                    self._send_json(200, response)
                else:
                    self._send_json(502, {"error": {"message": result.error}})
            except Exception as exc:
                self._send_json(500, {"error": {"message": str(exc)}})

        def do_GET(self):
            if self.path == "/v1/models" or self.path == "/models":
                models = []
                for p in gateway._providers.values():
                    for m in p.models:
                        models.append({"id": m, "object": "model", "owned_by": p.name})
                self._send_json(200, {"object": "list", "data": models})
            elif self.path == "/stats":
                self._send_json(200, gateway.stats())
            else:
                self.send_error(404, "Not found")

        def _send_json(self, code, data):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass  # suppress logs

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Multi-Provider Gateway running on http://localhost:{port}")
    print(f"  Endpoint: http://localhost:{port}/v1/chat/completions")
    print(f"  Models:   http://localhost:{port}/v1/models")
    print(f"  Stats:    http://localhost:{port}/stats")
    print(f"  Providers: {len(gateway._providers)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_gateway: MultiProviderGateway | None = None


def get_gateway() -> MultiProviderGateway:
    global _gateway
    if _gateway is None:
        _gateway = MultiProviderGateway()
    return _gateway


def configure_gateway(
    *,
    strategy: str = "round-robin",
    failover_order: list[str] | None = None,
    config_path: str | Path | None = None,
) -> MultiProviderGateway:
    global _gateway
    _gateway = MultiProviderGateway(
        strategy=strategy,  # type: ignore[arg-type]
        failover_order=failover_order,
        config_path=config_path,
    )
    return _gateway


def multi_provider_chat(
    message: str,
    *,
    model: str = "",
    system: str = "",
    preferred_provider: str = "",
) -> dict[str, Any]:
    """Public API: chat via multi-provider gateway."""
    result = get_gateway().chat(
        message,
        model=model,
        system=system,
        preferred_provider=preferred_provider,
    )
    return {
        "success": result.success,
        "content": result.content,
        "provider": result.provider,
        "model": result.model,
        "tokens_used": result.tokens_used,
        "elapsed_ms": result.elapsed_ms,
        "error": result.error,
    }


def multi_provider_stats() -> dict[str, Any]:
    """Public API: get gateway stats."""
    return get_gateway().stats()


def multi_provider_list() -> list[dict[str, Any]]:
    """Public API: list all providers + keys."""
    return get_gateway().list_providers()


def multi_provider_add(
    provider: str,
    base_url: str,
    keys: list[str],
    models: list[str] | None = None,
) -> None:
    """Public API: add a provider."""
    get_gateway().add_provider(provider, base_url, keys, models)


def multi_provider_add_key(provider: str, key: str, quota_tokens: int = 0) -> bool:
    """Public API: add a key to a provider."""
    return get_gateway().add_key(provider, key, quota_tokens)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Provider Gateway")
    parser.add_argument("--serve", action="store_true", help="Run as HTTP server")
    parser.add_argument("--port", type=int, default=8787, help="Server port")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--list", action="store_true", help="List providers")
    args = parser.parse_args()

    if args.serve:
        run_server(port=args.port)
    elif args.stats:
        print(json.dumps(get_gateway().stats(), indent=2))
    elif args.list:
        print(json.dumps(get_gateway().list_providers(), indent=2, default=str))
    else:
        parser.print_help()
