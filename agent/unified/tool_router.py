"""Tool Router — automatically suggests the right tool for a task.

The problem this solves
-----------------------
A senior engineer doesn't need to be told "use grep to search files" —
they just know. A junior engineer does. The same is true for LLMs:

- Top-tier models (Claude Opus, GPT-5, GLM-4.6) rarely need help picking
  tools — they've seen enough training data to know that "search file
  contents" → grep.
- Mid-tier models (Haiku, MiniMax, smaller open-source) often pick the
  wrong tool, miss available tools, or use a generic `bash` when a
  purpose-built tool would work better.
- Even top-tier models miss tools they don't know exist — MCP servers,
  custom plugins, newly-registered skills.

The ToolRouter helps all of them by:

1. **Indexing all available tools** — name, description, parameters,
   toolset, source (built-in / plugin / MCP / skill). Uses the existing
   `tools.registry` and reuses the BM25 catalog from `tools/tool_search`.
2. **Routing by intent** — given a natural-language task description,
   returns the top-N tools most likely to be relevant. Uses BM25 first
   (cheap), then optionally an LLM reranker (accurate).
3. **Injecting suggestions into the system prompt** — when enabled, the
   router prepends a "Relevant tools for this task:" block to the
   system prompt, so the LLM sees the suggestions before generating a
   tool call.
4. **Tracking usage** — counts how often each tool is actually called.
   Tools that are frequently called for a given intent get a relevance
   boost in future routing (learning from usage).

Design goals
------------
1. **Cheap by default.** BM25 is microseconds. LLM reranking is opt-in.
2. **Non-blocking.** If the router fails, the agent proceeds without
   suggestions (fail-open).
3. **Composable with tool_search.py.** The existing tool_search system
   uses a "bridge tool" pattern to defer tool loading. The ToolRouter
   is complementary — it suggests *which* tools to load, while
   tool_search handles the lazy loading mechanics.
4. **Learns from usage.** A simple usage histogram shapes future
   rankings without requiring retraining.

Integration points
------------------
- **System prompt augmentation** — the conversation loop can call
  `router.suggest_for_task(user_message)` and inject the result into
  the system prompt before the LLM generates a tool call.
- **Tool call validation** — when the LLM picks a tool, the router can
  sanity-check: "did the LLM pick `bash` when `grep` would be better?"
  This is advisory, not blocking.
- **Tool discovery** — when the user asks "what can you do?", the
  router can produce a categorized list.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tool metadata
# --------------------------------------------------------------------------- #


@dataclass
class ToolMeta:
    """Metadata about one registered tool."""

    name: str
    description: str
    toolset: str
    parameters: list[str]  # top-level parameter names
    source: str  # "builtin" | "plugin" | "mcp" | "skill" | "other"
    emoji: str = "⚡"
    # Tokenized text for BM25 (name + description + params).
    _tokens: list[str] = field(default_factory=list, repr=False)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _build_meta(name: str, schema: dict[str, Any], toolset: str, source: str, emoji: str) -> ToolMeta:
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    description = str(fn.get("description", "") or "")
    params = list(((fn.get("parameters") or {}).get("properties") or {}).keys())
    name_words = name.replace("_", " ").replace(".", " ").replace("-", " ").replace(":", " ")
    text = f"{name_words} {description} {' '.join(params)}"
    return ToolMeta(
        name=name,
        description=description,
        toolset=toolset,
        parameters=params,
        source=source,
        emoji=emoji,
        _tokens=_tokenize(text),
    )


# --------------------------------------------------------------------------- #
# BM25 (small inline impl, matches tools/tool_search.py)
# --------------------------------------------------------------------------- #


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    *,
    doc_lengths: list[int],
    avg_dl: float,
    doc_freq: dict[str, int],
    n_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not doc_tokens:
        return 0.0
    score = 0.0
    dl = len(doc_tokens)
    doc_tf: dict[str, int] = {}
    for t in doc_tokens:
        doc_tf[t] = doc_tf.get(t, 0) + 1
    for q in query_tokens:
        df = doc_freq.get(q, 0)
        if df == 0:
            continue
        idf = float((n_docs - df + 0.5)) / float(df + 0.5)
        idf = max(idf, 0.0)  # never let IDF go negative
        import math

        idf = math.log(1.0 + idf)
        tf = doc_tf.get(q, 0)
        if tf == 0:
            continue
        norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0)))
        score += idf * norm
    return score


# --------------------------------------------------------------------------- #
# Intent → tool mapping heuristics
# --------------------------------------------------------------------------- #
# These are short-circuit patterns that bypass BM25 when the user's intent
# is unambiguous. Each entry is (regex_pattern, list of tool name substrings
# to boost). The boost is applied as a multiplier to the BM25 score.

_INTENT_PATTERNS: list[tuple[re.Pattern[str], list[tuple[str, float]]]] = [
    # File operations
    (re.compile(r"\b(read|cat|view|show|display)\s+(a\s+)?(file|contents?)\b", re.IGNORECASE),
     [("read", 3.0), ("file", 2.0), ("cat", 2.0)]),
    (re.compile(r"\b(write|create|make|touch|save)\s+(a\s+)?(file|document)\b", re.IGNORECASE),
     [("write", 3.0), ("create", 2.5), ("save", 2.0), ("file", 1.5)]),
    (re.compile(r"\b(edit|modify|update|change|patch)\s+(a\s+)?(file|line|content)", re.IGNORECASE),
     [("edit", 3.0), ("write", 2.0), ("patch", 2.5)]),
    (re.compile(r"\b(delete|remove|rm)\s+(a\s+)?(file|dir|directory)", re.IGNORECASE),
     [("delete", 3.0), ("remove", 2.5), ("rm", 2.0)]),
    # Search
    (re.compile(r"\b(search|find|grep|look\s+for|locate)\s+(in\s+)?(file|files|content|code)", re.IGNORECASE),
     [("grep", 3.0), ("search", 2.5), ("find", 2.5)]),
    (re.compile(r"\b(list|ls|dir|show)\s+(files?|directories|folder)", re.IGNORECASE),
     [("ls", 3.0), ("list", 2.5), ("glob", 2.5)]),
    (re.compile(r"\b(glob|find\s+files?\s+by\s+name|pattern\s+match\s+files)", re.IGNORECASE),
     [("glob", 3.0), ("find", 2.0)]),
    # Shell/execute
    (re.compile(r"\b(run|execute|bash|shell|command\s+line)\b", re.IGNORECASE),
     [("bash", 3.0), ("shell", 2.5), ("execute", 2.5), ("terminal", 2.0)]),
    # Web
    (re.compile(r"\b(search|google|look\s+up)\s+(the\s+)?(web|internet|online)", re.IGNORECASE),
     [("web_search", 3.0), ("search", 2.5), ("brave", 2.0), ("ddgs", 1.5), ("tavily", 1.5)]),
    (re.compile(r"\b(fetch|download|scrape|crawl)\s+(a\s+)?(url|webpage|website|page)", re.IGNORECASE),
     [("fetch", 3.0), ("web_fetch", 2.5), ("scrape", 2.0), ("crawl", 2.0)]),
    # Code execution
    (re.compile(r"\b(run|execute)\s+(python|py|code|script)", re.IGNORECASE),
     [("execute_code", 3.0), ("python", 2.5), ("code", 2.0)]),
    (re.compile(r"\b(run|execute)\s+(node|js|javascript)", re.IGNORECASE),
     [("execute_code", 2.5), ("node", 2.0), ("javascript", 2.0)]),
    # Memory / reflexion
    (re.compile(r"\b(remember|recall|memory|lesson|past\s+experience|previous\s+failure)", re.IGNORECASE),
     [("unified_recall", 3.0), ("recall", 2.5), ("memory", 2.0), ("reflexion", 2.5)]),
    # Reasoning
    (re.compile(r"\b(plan|think|reason|strategy|approach)\s+(before|about)", re.IGNORECASE),
     [("reasoning_plan", 3.0), ("reasoning", 2.5), ("plan", 2.0)]),
    # Browser
    (re.compile(r"\b(open|browse|navigate|visit)\s+(a\s+)?(website|page|url|browser)", re.IGNORECASE),
     [("browser", 3.0), ("browse", 2.5), ("navigate", 2.0)]),
    # Image / video
    (re.compile(r"\b(generate|create|make)\s+(an?\s+)?(image|picture|photo|drawing)", re.IGNORECASE),
     [("image_gen", 3.0), ("generate_image", 2.5), ("image", 2.0), ("fal", 1.5), ("openai", 1.0)]),
    (re.compile(r"\b(generate|create|make)\s+(a\s+)?(video|movie|clip)", re.IGNORECASE),
     [("video_gen", 3.0), ("generate_video", 2.5), ("video", 2.0)]),
    # Voice
    (re.compile(r"\b(speak|say|read\s+aloud|voice|tts|narrate)", re.IGNORECASE),
     [("tts", 3.0), ("speak", 2.5), ("voice", 2.0)]),
    (re.compile(r"\b(transcribe|speech\s+to\s+text|stt|listen\s+to\s+audio)", re.IGNORECASE),
     [("transcribe", 3.0), ("stt", 2.5), ("whisper", 2.0)]),
    # Cron / scheduling
    (re.compile(r"\b(schedule|cron|recurring|every\s+(day|hour|week|minute))", re.IGNORECASE),
     [("cron", 3.0), ("schedule", 2.5), ("recurring", 2.0)]),
    # Delegation / subagent
    (re.compile(r"\b(delegate|subagent|parallel|spawn|sub-task|subprocess)", re.IGNORECASE),
     [("delegate", 3.0), ("spawn", 2.5), ("subagent", 2.5)]),
    # Send message
    (re.compile(r"\b(send|message|notify|email|chat)\s+(to|a|an)", re.IGNORECASE),
     [("send_message", 3.0), ("notify", 2.0), ("email", 2.0)]),
]


# --------------------------------------------------------------------------- #
# ToolRouter
# --------------------------------------------------------------------------- #


class ToolRouter:
    """Auto-suggests the right tool for a task.

    Lifecycle:
        1. `refresh()` — (re)build the catalog from `tools.registry`.
        2. `suggest_for_task(query, top_n=5)` — returns ranked tools.
        3. `record_usage(tool_name, query)` — feedback loop.

    The router is thread-safe (read-only after refresh).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._catalog: list[ToolMeta] = []
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0
        self._doc_freq: dict[str, int] = {}
        self._n_docs: int = 0
        self._usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # ^ _usage[tool_name][intent_keyword] = count
        self._last_refresh: float = 0.0

    # ------------------------------------------------------------------ #
    # Catalog management
    # ------------------------------------------------------------------ #

    def refresh(self) -> int:
        """Rebuild the catalog from `tools.registry`. Returns the count."""
        with self._lock:
            try:
                from tools.registry import registry

                names = registry.get_all_tool_names()
                catalog: list[ToolMeta] = []
                for name in names:
                    try:
                        entry = registry.get_entry(name)
                        if entry is None:
                            continue
                        schema = registry.get_schema(name) or {}
                        source = self._classify_source(entry.toolset)
                        emoji = registry.get_emoji(name, default="⚡")
                        meta = _build_meta(
                            name=name,
                            schema=schema,
                            toolset=entry.toolset,
                            source=source,
                            emoji=emoji,
                        )
                        catalog.append(meta)
                    except Exception:
                        continue
                self._catalog = catalog
                self._doc_lengths = [len(m._tokens) for m in catalog]
                self._avg_dl = (
                    sum(self._doc_lengths) / max(len(self._doc_lengths), 1)
                )
                self._doc_freq = {}
                for meta in catalog:
                    seen = set(meta._tokens)
                    for t in seen:
                        self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
                self._n_docs = len(catalog)
                self._last_refresh = time.time()
                return self._n_docs
            except Exception as exc:
                logger.warning("tool router refresh failed: %r", exc)
                return 0

    @staticmethod
    def _classify_source(toolset: str) -> str:
        if not toolset:
            return "other"
        if toolset.startswith("mcp-"):
            return "mcp"
        if toolset in {"unified", "filesystem", "terminal", "web", "browser", "voice"}:
            return "builtin"
        if "skill" in toolset.lower():
            return "skill"
        return "plugin"

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def suggest_for_task(
        self,
        query: str,
        *,
        top_n: int = 5,
        exclude: Iterable[str] = (),
    ) -> list[ToolMeta]:
        """Return the top-N tools most likely to be relevant for `query`.

        Algorithm:
            1. BM25 score against (name + description + params).
            2. Intent pattern boost (multiplier on BM25).
            3. Usage boost (small additive bonus based on past usage).
            4. Filter out `exclude` (e.g., tools already in system prompt).
        """
        with self._lock:
            if not self._catalog:
                self.refresh()
            if not self._catalog:
                return []
            excluded = set(exclude)
            query_tokens = _tokenize(query)
            if not query_tokens:
                return []

            # Intent boosts.
            intent_boosts: dict[str, float] = {}  # substring → multiplier
            for pattern, boosts in _INTENT_PATTERNS:
                if pattern.search(query):
                    for substring, mult in boosts:
                        # Take the max if multiple patterns match.
                        if substring not in intent_boosts or mult > intent_boosts[substring]:
                            intent_boosts[substring] = mult

            # Score every tool.
            scored: list[tuple[float, ToolMeta]] = []
            for meta in self._catalog:
                if meta.name in excluded:
                    continue
                bm25 = _bm25_score(
                    query_tokens,
                    meta._tokens,
                    doc_lengths=self._doc_lengths,
                    avg_dl=self._avg_dl,
                    doc_freq=self._doc_freq,
                    n_docs=self._n_docs,
                )
                # Apply intent boosts.
                boost = 1.0
                for substring, mult in intent_boosts.items():
                    if substring in meta.name.lower():
                        boost = max(boost, mult)
                # Usage bonus (small, capped).
                usage_bonus = 0.0
                usage_map = self._usage.get(meta.name, {})
                for kw in query_tokens:
                    usage_bonus += min(usage_map.get(kw, 0), 5) * 0.1
                usage_bonus = min(usage_bonus, 1.0)
                score = bm25 * boost + usage_bonus
                if score > 0:
                    scored.append((score, meta))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [m for _, m in scored[:top_n]]

    def format_suggestions(self, query: str, *, top_n: int = 5) -> str:
        """Render suggestions as a markdown block for the system prompt."""
        tools = self.suggest_for_task(query, top_n=top_n)
        if not tools:
            return ""
        lines = ["<tool-suggestions>", f"Task: {query[:200]}", "Relevant tools:"]
        for t in tools:
            params = ", ".join(t.parameters[:5])
            params_text = f" (params: {params})" if params else ""
            lines.append(
                f"- {t.emoji} {t.name} [{t.source}/{t.toolset}]{params_text} — "
                f"{t.description[:160]}"
            )
        lines.append("</tool-suggestions>")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Usage feedback
    # ------------------------------------------------------------------ #

    def record_usage(self, tool_name: str, query: str) -> None:
        """Record that `tool_name` was used for `query`. Used to boost
        future suggestions for similar queries."""
        with self._lock:
            tokens = _tokenize(query)
            for t in tokens:
                self._usage[tool_name][t] += 1

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "catalog_size": self._n_docs,
                "last_refresh_ago_s": (
                    time.time() - self._last_refresh if self._last_refresh else None
                ),
                "usage_entries": sum(len(v) for v in self._usage.values()),
            }


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_router: ToolRouter | None = None
_router_lock = threading.Lock()


def get_router() -> ToolRouter:
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = ToolRouter()
                _router.refresh()
    return _router


def refresh_router() -> int:
    """Force a refresh of the global router. Returns the new catalog size."""
    return get_router().refresh()
