"""Long-Run Engine — keeps the agent alive across hours/days without
timeouts, sleeping, or lost work.

The problem this solves
-----------------------
Hermes' default conversation loop is **synchronous and per-turn**:
each user message triggers a turn that runs until completion (or
timeout). For long-running autonomous work — "monitor this repo for
changes and refactor when needed" — this model breaks down:

1. **Timeouts kill work.** The reasoning protocol v1 added 1-3 LLM
   calls per consequential action. With `reasoning_timeouts.py`
   flooring at 24h (unlimited mode), the agent won't be killed
   mid-plan, but a single turn can run for *hours* and the user
   has no way to checkpoint/resume.
2. **No background reflection.** v1 reflection runs synchronously
   after each tool call, adding latency. For long-running work we
   want reflection to happen *in the background* — batch multiple
   reflections together, debounce, run on a worker thread.
3. **No work queue.** If the agent wants to "do X, then Y, then Z"
   across an hour, it has to do them all in one turn. A work queue
   lets it enqueue items and process them across multiple ticks.
4. **No checkpoint/resume.** If the process crashes mid-task, all
   in-flight work is lost. A checkpoint store lets the agent resume
   from the last completed step.

Architecture
------------
```
   ┌─────────────────────────────────────────────────────────┐
   │                     LongRunEngine                       │
   │                                                          │
   │  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
   │  │  Work Queue  │ ─▶ │   Dispatcher │ ─▶ │  Worker   │ │
   │  │ (priority)   │    │  (priority)  │    │ (1 thread)│ │
   │  └──────────────┘    └──────────────┘    └───────────┘ │
   │         │                                       │       │
   │         ▼                                       ▼       │
   │  ┌──────────────┐                      ┌──────────────┐│
   │  │ Checkpoint   │ ◀── write after ────│  Result     ││
   │  │ Store (JSONL)│     each item done  │  Sink       ││
   │  └──────────────┘                      └──────────────┘│
   │                                                          │
   │  ┌──────────────────────────────────────────────────┐  │
   │  │  Background Reflection Worker (separate thread)  │  │
   │  │  - Debounced (waits 5s after last tool call)     │  │
   │  │  - Batched (groups up to 10 reflections)         │  │
   │  │  - Calls ReasoningProtocol.reflect_batch()       │  │
   │  └──────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────┘
```

Key design decisions
--------------------
1. **Single worker thread.** Avoids concurrency headaches with the
   tool registry (which is not thread-safe). Multiple workers would
   require locking around tool dispatch; one worker keeps it simple.
   If you need parallelism, spawn multiple LongRunEngines.

2. **Priority queue.** Critical work (user-initiated) jumps ahead of
   background work (reflexion, indexing). Implemented as a heap.

3. **Checkpoint after every item.** Each completed item is appended
   to a JSONL checkpoint file. On restart, the engine replays the
   checkpoint to skip already-done items.

4. **Reflection is fire-and-forget.** The reflection worker runs on
   its own thread and never blocks tool dispatch. If the LLM call
   fails, the reflection is silently dropped (fail-open).

5. **Heartbeat.** The engine emits a heartbeat event every N seconds
   so monitoring can detect a stuck worker. Configurable.

6. **No external deps.** Uses stdlib only (threading, queue, json,
   pathlib). No asyncio (would require rewriting the whole Hermes
   loop). The worker thread does blocking I/O, which is fine because
   there's only one.

Compatibility
-------------
- **Default OFF.** When disabled, behavior is identical to v1.
- **Coexists with v1.** The reasoning protocol still runs synchronously
  for consequential actions (to block before execution). Only the
  *reflection* phase moves to the background when longrun is enabled.
- **No changes to conversation_loop.py.** The engine is opt-in via
  config and wired through `agent.unified.integration`.
"""

from __future__ import annotations

import heapq
import itertools
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from queue import Empty, PriorityQueue
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


class WorkItemPriority(IntEnum):
    """Lower number = higher priority (heap semantics)."""

    CRITICAL = 0  # user-initiated, must run now
    HIGH = 10  # tool dispatch from agent
    NORMAL = 50  # background indexing
    LOW = 100  # reflection, cleanup


@dataclass(order=True)
class WorkItem:
    """One unit of work for the long-run engine.

    The sort order is (priority, sequence) so FIFO within a priority.
    """

    priority: int
    sequence: int
    item_id: str = field(compare=False, default="")
    kind: str = field(compare=False, default="")  # "tool_call" | "reflection" | "checkpoint" | "custom"
    payload: dict[str, Any] = field(compare=False, default_factory=dict)
    created_at: float = field(compare=False, default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.item_id:
            self.item_id = f"{self.kind}_{int(self.created_at * 1000)}_{self.sequence}"


@dataclass
class CheckpointEntry:
    """One completed work item, persisted to disk for resume."""

    item_id: str
    kind: str
    started_at: float
    completed_at: float
    success: bool
    result_preview: str  # first ~500 chars of result
    error: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- #
# Checkpoint store
# --------------------------------------------------------------------------- #


class CheckpointStore:
    """JSONL append-only log of completed work items.

    On engine restart, the store is replayed to populate the
    `_completed_ids` set, which the dispatcher uses to skip
    already-done items (idempotency).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed_ids: set[str] = set()
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("item_id"):
                        self._completed_ids.add(data["item_id"])
                except Exception:
                    continue
        except Exception:
            pass  # corrupt checkpoint — start fresh

    def is_done(self, item_id: str) -> bool:
        with self._lock:
            return item_id in self._completed_ids

    def mark_done(self, entry: CheckpointEntry) -> None:
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(entry.to_jsonl() + "\n")
                self._completed_ids.add(entry.item_id)
            except Exception as exc:
                logger.warning("checkpoint write failed: %r", exc)

    def count(self) -> int:
        with self._lock:
            return len(self._completed_ids)

    def reset(self) -> int:
        """Clear the checkpoint. Returns the count of removed entries."""
        with self._lock:
            n = len(self._completed_ids)
            self._completed_ids.clear()
            try:
                self.path.write_text("", encoding="utf-8")
            except Exception:
                pass
            return n


# --------------------------------------------------------------------------- #
# Background reflection worker
# --------------------------------------------------------------------------- #


class ReflectionWorker:
    """Debounced, batched background reflection.

    Collects reflection requests (plan + tool_name + args + result) and
    processes them in batches on a worker thread. Debounce: waits
    `debounce_seconds` after the last request before processing. Batch:
    processes up to `batch_size` items per LLM call.

    The LLM call is made via `reflect_batch_fn(items) -> list[lesson]`,
    supplied by the caller (typically `ReasoningProtocol.reflect_batch`).
    If the function raises or returns None for an item, that reflection
    is silently dropped (fail-open).
    """

    def __init__(
        self,
        *,
        reflect_batch_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any] | None]],
        debounce_seconds: float = 5.0,
        batch_size: int = 10,
        poll_interval: float = 0.5,
    ) -> None:
        self._reflect_batch_fn = reflect_batch_fn
        self._debounce = max(0.1, debounce_seconds)
        self._batch_size = max(1, batch_size)
        self._poll_interval = max(0.05, poll_interval)
        self._queue: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._last_enqueue_time: float = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._processed_count = 0
        self._dropped_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="hermes-longrun-reflection", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def enqueue(self, item: dict[str, Any]) -> None:
        """Add a reflection request. Non-blocking."""
        with self._cv:
            self._queue.append(item)
            self._last_enqueue_time = time.time()
            self._cv.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._queue),
                "processed": self._processed_count,
                "dropped": self._dropped_count,
                "last_enqueue_ago_s": (
                    time.time() - self._last_enqueue_time
                    if self._last_enqueue_time
                    else None
                ),
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._cv:
                    # Wait until we have items AND debounce has elapsed,
                    # or we're being stopped.
                    while not self._queue and not self._stop.is_set():
                        self._cv.wait(timeout=self._poll_interval)
                    if self._stop.is_set():
                        break
                    # Debounce: wait until no new items for `debounce_seconds`.
                    while True:
                        elapsed = time.time() - self._last_enqueue_time
                        if elapsed >= self._debounce:
                            break
                        remaining = self._debounce - elapsed
                        self._cv.wait(timeout=min(remaining, self._poll_interval))
                        if self._stop.is_set():
                            break
                    if self._stop.is_set():
                        break
                    # Pull a batch.
                    batch = self._queue[: self._batch_size]
                    self._queue = self._queue[len(batch) :]

                # Process batch outside the lock.
                if batch:
                    self._process_batch(batch)
            except Exception as exc:
                logger.warning("reflection worker error: %r", exc)
                # Don't crash the worker thread on a single batch failure.
                time.sleep(1.0)

    def _process_batch(self, batch: list[dict[str, Any]]) -> None:
        try:
            results = self._reflect_batch_fn(batch)
        except Exception as exc:
            logger.warning("reflect_batch_fn raised: %r — dropping %d items", exc, len(batch))
            with self._lock:
                self._dropped_count += len(batch)
            return

        if not isinstance(results, list) or len(results) != len(batch):
            # Function returned wrong shape — drop the whole batch.
            with self._lock:
                self._dropped_count += len(batch)
            return

        # Persist successful reflections.
        from .integration import get_store  # late import to avoid cycle
        from .reflexion import ReflexionRecord

        persisted = 0
        for item, result in zip(batch, results):
            if result is None:
                continue
            lesson = str(result.get("lesson", "")).strip()
            if not lesson:
                continue
            try:
                record = ReflexionRecord(
                    lesson=lesson,
                    source="reasoning_reflection_bg",
                    score=float(result.get("score", 1.5)),
                    tags=["reflection", "reasoning", "background"]
                    + (["outcome_mismatch"] if not result.get("outcome_matched", True) else []),
                    session_id=str(item.get("session_id", "")),
                    turn_id=str(item.get("turn_id", "")),
                    tool_name=str(item.get("tool_name", "")),
                    scope=str(item.get("scope", "global")),
                )
                if get_store().add(record):
                    persisted += 1
            except Exception:
                continue

        with self._lock:
            self._processed_count += len(batch)
        logger.debug(
            "reflection batch: %d items, %d lessons persisted", len(batch), persisted
        )


# --------------------------------------------------------------------------- #
# Long-run engine
# --------------------------------------------------------------------------- #


class LongRunEngine:
    """Persistent work queue with checkpoint/resume + background reflection.

    The engine is a singleton per process. It owns:
        - A priority queue of WorkItems
        - A single worker thread that processes items
        - A CheckpointStore for idempotency
        - A ReflectionWorker for background reflection

    The engine is **not** the agent loop. It is a *companion* that
    handles background work and long-running tasks. The main conversation
    loop still runs synchronously — but when longrun is enabled, the
    *reflection* phase of each tool call is enqueued to the background
    instead of running synchronously.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        heartbeat_seconds: float = 30.0,
        reflection_debounce_seconds: float = 5.0,
        reflection_batch_size: int = 10,
    ) -> None:
        self._checkpoint = CheckpointStore(checkpoint_path)
        self._heartbeat = max(5.0, heartbeat_seconds)
        self._queue: PriorityQueue[WorkItem] = PriorityQueue()
        self._counter = itertools.count()  # stable sequence for FIFO within priority
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_heartbeat: float = 0.0
        self._items_processed = 0
        self._items_failed = 0
        self._items_skipped = 0
        self._lock = threading.Lock()
        self._handlers: dict[str, Callable[[WorkItem], Any]] = {}
        self._reflection_worker: ReflectionWorker | None = None
        self._reflection_debounce = reflection_debounce_seconds
        self._reflection_batch_size = reflection_batch_size
        self._started = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        if self._started:
            return
        self._started = True
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run, name="hermes-longrun-worker", daemon=True
        )
        self._worker.start()
        # Reflection worker is started lazily when a reflect_fn is set.
        logger.info(
            "longrun engine started (checkpoint=%s, heartbeat=%ds)",
            self._checkpoint.path,
            self._heartbeat,
        )

    def stop(self, *, timeout: float = 10.0) -> None:
        """Signal stop and wait for the worker to drain."""
        self._stop.set()
        if self._reflection_worker is not None:
            self._reflection_worker.stop(timeout=timeout)
        if self._worker is not None:
            # Wake up the worker.
            self._queue.put(WorkItem(priority=0, sequence=0, kind="_stop"))
            self._worker.join(timeout=timeout)
            self._worker = None
        self._started = False

    def register_handler(self, kind: str, handler: Callable[[WorkItem], Any]) -> None:
        """Register a handler for a work item kind.

        The handler receives the WorkItem and returns a string (result)
        or raises. Exceptions are caught and logged; the item is marked
        failed in the checkpoint.
        """
        with self._lock:
            self._handlers[kind] = handler

    def set_reflect_batch_fn(
        self,
        fn: Callable[[list[dict[str, Any]]], list[dict[str, Any] | None]],
    ) -> None:
        """Wire the reflection batch function. Lazily starts the worker."""
        if self._reflection_worker is None:
            self._reflection_worker = ReflectionWorker(
                reflect_batch_fn=fn,
                debounce_seconds=self._reflection_debounce,
                batch_size=self._reflection_batch_size,
            )
            self._reflection_worker.start()
        else:
            # Replace the function by recreating the worker.
            self._reflection_worker.stop(timeout=2.0)
            self._reflection_worker = ReflectionWorker(
                reflect_batch_fn=fn,
                debounce_seconds=self._reflection_debounce,
                batch_size=self._reflection_batch_size,
            )
            self._reflection_worker.start()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def enqueue(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        priority: WorkItemPriority = WorkItemPriority.NORMAL,
        item_id: str = "",
    ) -> str:
        """Enqueue a work item. Returns the item_id.

        If the item_id was already completed (per checkpoint), it is
        skipped silently — this is the idempotency mechanism.
        """
        seq = next(self._counter)
        if not item_id:
            item_id = f"{kind}_{seq}"
        if self._checkpoint.is_done(item_id):
            with self._lock:
                self._items_skipped += 1
            return item_id
        item = WorkItem(
            priority=int(priority),
            sequence=seq,
            item_id=item_id,
            kind=kind,
            payload=payload,
        )
        self._queue.put(item)
        return item_id

    def enqueue_reflection(self, item: dict[str, Any]) -> None:
        """Enqueue a reflection request to the background worker."""
        if self._reflection_worker is None:
            return  # not wired — drop silently (fail-open)
        self._reflection_worker.enqueue(item)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started": self._started,
                "queue_size": self._queue.qsize(),
                "items_processed": self._items_processed,
                "items_failed": self._items_failed,
                "items_skipped_duplicate": self._items_skipped,
                "checkpoint_entries": self._checkpoint.count(),
                "last_heartbeat_ago_s": (
                    time.time() - self._last_heartbeat if self._last_heartbeat else None
                ),
                "reflection": (
                    self._reflection_worker.stats() if self._reflection_worker else None
                ),
            }

    def reset_checkpoint(self) -> int:
        """Clear the checkpoint store. Returns the count of removed entries."""
        return self._checkpoint.reset()

    # ------------------------------------------------------------------ #
    # Worker loop
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Heartbeat.
                now = time.time()
                if now - self._last_heartbeat > self._heartbeat:
                    self._last_heartbeat = now
                    # Emit heartbeat event (best-effort).
                    try:
                        from .integration import get_bus

                        get_bus().emit(
                            "longrun.heartbeat",
                            {"stats": self.stats()},
                        )
                    except Exception:
                        pass

                # Block up to 1s waiting for an item. This lets the
                # heartbeat tick even when the queue is empty.
                try:
                    item = self._queue.get(timeout=1.0)
                except Empty:
                    continue

                # Stop signal.
                if item.kind == "_stop":
                    break

                # Idempotency: skip if already done (may have been
                # re-enqueued after completion).
                if self._checkpoint.is_done(item.item_id):
                    with self._lock:
                        self._items_skipped += 1
                    self._queue.task_done()
                    continue

                # Dispatch.
                started = time.time()
                handler = self._handlers.get(item.kind)
                if handler is None:
                    logger.warning("no handler for kind=%r — dropping", item.kind)
                    self._queue.task_done()
                    continue

                success = True
                error = ""
                result_preview = ""
                try:
                    result = handler(item)
                    if isinstance(result, str):
                        result_preview = result[:500]
                    elif result is not None:
                        try:
                            result_preview = json.dumps(
                                result, ensure_ascii=False, default=str
                            )[:500]
                        except Exception:
                            result_preview = str(result)[:500]
                except Exception as exc:
                    success = False
                    error = repr(exc)
                    logger.warning(
                        "longrun handler %r failed: %r", item.kind, exc
                    )

                # Checkpoint.
                self._checkpoint.mark_done(
                    CheckpointEntry(
                        item_id=item.item_id,
                        kind=item.kind,
                        started_at=started,
                        completed_at=time.time(),
                        success=success,
                        result_preview=result_preview,
                        error=error,
                    )
                )

                with self._lock:
                    if success:
                        self._items_processed += 1
                    else:
                        self._items_failed += 1

                self._queue.task_done()

            except Exception as exc:
                # Never let the worker thread die.
                logger.error("longrun worker top-level error: %r", exc)
                time.sleep(1.0)


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_engine: LongRunEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> LongRunEngine | None:
    """Return the global engine, or None if not initialized."""
    return _engine


def configure_engine(
    *,
    checkpoint_path: str | Path | None = None,
    heartbeat_seconds: float = 30.0,
    reflection_debounce_seconds: float = 5.0,
    reflection_batch_size: int = 10,
    autostart: bool = True,
) -> LongRunEngine:
    """Initialize (or replace) the global long-run engine."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.stop(timeout=2.0)
            except Exception:
                pass
        if checkpoint_path is None:
            from hermes_constants import get_hermes_home

            checkpoint_path = get_hermes_home() / "unified" / "longrun_checkpoint.jsonl"
        _engine = LongRunEngine(
            checkpoint_path=checkpoint_path,
            heartbeat_seconds=heartbeat_seconds,
            reflection_debounce_seconds=reflection_debounce_seconds,
            reflection_batch_size=reflection_batch_size,
        )
        if autostart:
            _engine.start()
        return _engine


def shutdown_engine(*, timeout: float = 5.0) -> None:
    """Stop and clear the global engine."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.stop(timeout=timeout)
            except Exception:
                pass
            _engine = None
