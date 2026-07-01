"""Optional tracing helpers.

If OpenTelemetry is installed (AgentScope uses it heavily), unified Hermes emits
spans. Otherwise the helpers degrade to no-ops.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    try:
        from opentelemetry import trace  # type: ignore

        tracer = trace.get_tracer("hermes.unified")
        with tracer.start_as_current_span(name) as active:
            for key, value in attributes.items():
                if value is not None:
                    active.set_attribute(key, str(value))
            yield
    except Exception:
        yield
