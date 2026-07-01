"""In-process event bus used by the unified Hermes plugin.

Inspired by OmniAgent's channel/bus split and AgentScope's message bus, but
implemented without depending on either project so Hermes keeps its lean core.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, DefaultDict
from collections import defaultdict, deque


@dataclass(frozen=True)
class Event:
    """A typed event emitted by the unified layer."""

    topic: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    session_id: str = ""
    turn_id: str = ""
    trace_id: str = ""


Subscriber = Callable[[Event], None]


class EventBus:
    """Small thread-safe publish/subscribe bus.

    Subscribers can attach to an exact topic or to ``"*"`` for all events.
    A bounded recent-event buffer helps diagnostics without becoming a durable
    store. Durable concerns live in ReflexionStore/AuditSink.
    """

    def __init__(self, *, max_recent: int = 500) -> None:
        self._lock = RLock()
        self._subscribers: DefaultDict[str, list[Subscriber]] = defaultdict(list)
        self._recent: deque[Event] = deque(maxlen=max_recent)

    def subscribe(self, topic: str, callback: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers[topic].append(callback)

        def unsubscribe() -> None:
            with self._lock:
                callbacks = self._subscribers.get(topic, [])
                if callback in callbacks:
                    callbacks.remove(callback)

        return unsubscribe

    def publish(self, event: Event) -> Event:
        with self._lock:
            self._recent.append(event)
            callbacks = list(self._subscribers.get(event.topic, ())) + list(
                self._subscribers.get("*", ())
            )
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                # Event delivery is observational; never break an agent turn.
                continue
        return event

    def emit(self, topic: str, payload: dict[str, Any] | None = None, **meta: Any) -> Event:
        return self.publish(Event(topic=topic, payload=payload or {}, **meta))

    def recent(self, *, topic: str | None = None, limit: int = 100) -> list[Event]:
        with self._lock:
            items = list(self._recent)
        if topic is not None:
            items = [item for item in items if item.topic == topic]
        return items[-limit:]
