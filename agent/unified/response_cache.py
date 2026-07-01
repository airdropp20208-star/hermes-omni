"""Response Cache — content-hash cache for LLM calls.

WHY THIS EXISTS
---------------
Slow thinking, ensemble, verifier all call LLM with similar prompts.
If the same prompt is asked twice (e.g., agent retries, or user asks
the same question), we re-pay for the same response.

This module caches (system_prompt, user_prompt) → response by content
hash. Cache hits are FREE (don't count toward budget). TTL ensures
stale entries expire. Bounded LRU prevents unbounded growth.

USAGE
-----
    from agent.unified.response_cache import cached_llm_call

    response = cached_llm_call(
        phase="plan",
        lambda: llm(system, user),
        system=system,
        user=user,
    )

    # Or check + record manually:
    from agent.unified.response_cache import get_cache
    cache = get_cache()
    cached = cache.get(system, user)
    if cached is not None:
        return cached
    response = llm(system, user)
    cache.put(system, user, response)
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class CacheEntry:
    """One cached response."""

    response: str
    created_at: float
    access_count: int = 0
    last_accessed_at: float = 0.0
    size_bytes: int = 0


# --------------------------------------------------------------------------- #
# ResponseCache
# --------------------------------------------------------------------------- #


class ResponseCache:
    """Content-hash LLM response cache with TTL + LRU eviction.

    Thread-safe. Bounded size.
    """

    def __init__(
        self,
        *,
        max_entries: int = 512,
        ttl_seconds: int = 3600,  # 1 hour
        max_entry_size_bytes: int = 100_000,  # don't cache huge responses
    ) -> None:
        self._max_entries = max(8, max_entries)
        self._ttl = max(60, ttl_seconds)
        self._max_size = max(1024, max_entry_size_bytes)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _key(self, system: str, user: str) -> str:
        """Stable content-hash key."""
        h = hashlib.sha256()
        h.update((system or "").encode("utf-8", errors="ignore"))
        h.update(b"\0---\0")
        h.update((user or "").encode("utf-8", errors="ignore"))
        return h.hexdigest()[:32]

    def get(self, system: str, user: str) -> str | None:
        """Get cached response. Returns None on miss/expired."""
        key = self._key(system, user)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            # Check TTL.
            age = time.time() - entry.created_at
            if age > self._ttl:
                # Expired — evict.
                self._cache.pop(key, None)
                self._misses += 1
                return None
            # Hit — update access stats, move to end (LRU).
            entry.access_count += 1
            entry.last_accessed_at = time.time()
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.response

    def put(self, system: str, user: str, response: str) -> bool:
        """Cache a response. Returns True if cached, False if skipped
        (e.g., response too large)."""
        if not response or not isinstance(response, str):
            return False
        size = len(response.encode("utf-8", errors="ignore"))
        if size > self._max_size:
            return False  # don't cache huge responses
        key = self._key(system, user)
        with self._lock:
            self._cache[key] = CacheEntry(
                response=response,
                created_at=time.time(),
                last_accessed_at=time.time(),
                size_bytes=size,
            )
            self._cache.move_to_end(key)
            # Evict oldest if over capacity.
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return True

    def invalidate(self, system: str, user: str) -> bool:
        """Remove a specific entry. Returns True if existed."""
        key = self._key(system, user)
        with self._lock:
            return self._cache.pop(key, None) is not None

    def clear(self) -> int:
        """Clear all entries. Returns count removed."""
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
            return n

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_size = sum(e.size_bytes for e in self._cache.values())
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(1, self._hits + self._misses),
                "total_size_bytes": total_size,
                "ttl_seconds": self._ttl,
            }


# --------------------------------------------------------------------------- #
# Cached LLM call wrapper
# --------------------------------------------------------------------------- #


def cached_llm_call(
    phase: str,
    fn: Callable[[], T],
    *,
    system: str,
    user: str,
    use_cache: bool = True,
) -> T:
    """Wrap an LLM call with response caching.

    Args:
        phase: phase name (for cost tracking integration, optional)
        fn: callable() -> response (str)
        system: system prompt (cache key part 1)
        user: user prompt (cache key part 2)
        use_cache: if False, bypass cache entirely

    Returns:
        Cached response (str) or fresh response from fn().
    """
    cache = get_cache()
    if cache is None or not use_cache:
        return fn()
    # Check cache.
    cached = cache.get(system, user)
    if cached is not None:
        return cached  # type: ignore[return-value]
    # Miss — call fn and cache result.
    response = fn()
    if isinstance(response, str) and response:
        cache.put(system, user, response)
    return response


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_cache: ResponseCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> ResponseCache | None:
    return _cache


def configure_cache(
    *,
    max_entries: int = 512,
    ttl_seconds: int = 3600,
    max_entry_size_bytes: int = 100_000,
) -> ResponseCache:
    global _cache
    with _cache_lock:
        _cache = ResponseCache(
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            max_entry_size_bytes=max_entry_size_bytes,
        )
        return _cache


def get_cache_stats() -> dict[str, Any]:
    """Public API: get cache stats."""
    if _cache is None:
        return {"enabled": False}
    return _cache.stats()


def clear_response_cache() -> int:
    """Public API: clear cache. Returns count removed."""
    if _cache is None:
        return 0
    return _cache.clear()
