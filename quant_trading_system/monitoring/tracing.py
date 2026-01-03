"""
OpenTelemetry distributed tracing for AlphaTrade.

Provides request tracing across all system components for:
- Order lifecycle tracking
- Model inference latency
- Data pipeline monitoring
- API call tracing

Usage:
    from quant_trading_system.monitoring.tracing import get_tracer, trace_function

    # Get tracer for a component
    tracer = get_tracer("trading_engine")

    # Use as context manager
    with tracer.start_as_current_span("process_signal") as span:
        span.set_attribute("symbol", "AAPL")
        # ... do work

    # Or use decorator
    @trace_function("order_submission")
    async def submit_order(order):
        ...
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generator, TypeVar
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# Type variable for generic function wrapping
F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(str, Enum):
    """Type of span for categorization."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span completion status."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for span propagation across services."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    def to_headers(self) -> dict[str, str]:
        """Convert to W3C Trace Context headers."""
        return {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-01",
            "tracestate": "",
        }

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> SpanContext | None:
        """Parse from W3C Trace Context headers."""
        traceparent = headers.get("traceparent", "")
        if not traceparent:
            return None

        parts = traceparent.split("-")
        if len(parts) < 4:
            return None

        return cls(
            trace_id=parts[1],
            span_id=parts[2],
        )


@dataclass
class Span:
    """
    A span represents a single operation within a trace.

    Attributes:
        name: Operation name
        trace_id: Unique trace identifier
        span_id: Unique span identifier
        parent_span_id: Parent span (if any)
        kind: Type of span
        start_time: When span started
        end_time: When span ended
        attributes: Key-value pairs describing the operation
        events: Timestamped events during the span
        status: Final status of the operation
    """

    name: str
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid4().hex[:16])
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple attributes at once."""
        self.attributes.update(attributes)

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Add a timestamped event to the span."""
        self.events.append({
            "name": name,
            "timestamp": timestamp or time.time(),
            "attributes": attributes or {},
        })

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def end(self, end_time: float | None = None) -> None:
        """End the span."""
        self.end_time = end_time or time.time()

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def get_context(self) -> SpanContext:
        """Get span context for propagation."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status.value,
            "status_message": self.status_message,
        }


class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: list[Span]) -> None:
        """Export spans to backend."""
        pass

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Export spans to console/logs for debugging."""

    def __init__(self, log_level: int = logging.DEBUG) -> None:
        self._log_level = log_level

    def export(self, spans: list[Span]) -> None:
        for span in spans:
            logger.log(
                self._log_level,
                f"[TRACE] {span.name} | trace_id={span.trace_id[:8]}... | "
                f"duration={span.duration_ms:.2f}ms | status={span.status.value}",
            )


class InMemorySpanExporter(SpanExporter):
    """Store spans in memory for testing/analysis."""

    def __init__(self, max_spans: int = 10000) -> None:
        self._spans: list[Span] = []
        self._max_spans = max_spans
        self._lock = threading.Lock()

    def export(self, spans: list[Span]) -> None:
        with self._lock:
            self._spans.extend(spans)
            # Trim if over capacity
            if len(self._spans) > self._max_spans:
                self._spans = self._spans[-self._max_spans:]

    def get_spans(self) -> list[Span]:
        """Get all stored spans."""
        with self._lock:
            return list(self._spans)

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return [s for s in self._spans if s.trace_id == trace_id]

    def clear(self) -> None:
        """Clear all stored spans."""
        with self._lock:
            self._spans.clear()


class OTLPSpanExporter(SpanExporter):
    """
    Export spans via OTLP (OpenTelemetry Protocol).

    Sends spans to an OpenTelemetry collector or compatible backend
    like Jaeger, Zipkin, or cloud providers.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._endpoint = endpoint
        self._headers = headers or {}
        self._timeout = timeout
        self._batch: list[Span] = []
        self._batch_size = 100
        self._lock = threading.Lock()

    def export(self, spans: list[Span]) -> None:
        """Export spans via HTTP to OTLP endpoint."""
        if not spans:
            return

        try:
            import requests

            # Convert spans to OTLP format
            otlp_spans = [self._to_otlp_span(s) for s in spans]
            payload = {
                "resourceSpans": [{
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "alphatrade"}},
                        ]
                    },
                    "scopeSpans": [{
                        "scope": {"name": "alphatrade.tracing"},
                        "spans": otlp_spans,
                    }]
                }]
            }

            response = requests.post(
                self._endpoint,
                json=payload,
                headers={**self._headers, "Content-Type": "application/json"},
                timeout=self._timeout,
            )
            response.raise_for_status()

        except ImportError:
            logger.warning("requests library not available for OTLP export")
        except Exception as e:
            logger.error(f"Failed to export spans to OTLP: {e}")

    def _to_otlp_span(self, span: Span) -> dict[str, Any]:
        """Convert span to OTLP format."""
        return {
            "traceId": span.trace_id,
            "spanId": span.span_id,
            "parentSpanId": span.parent_span_id or "",
            "name": span.name,
            "kind": self._kind_to_otlp(span.kind),
            "startTimeUnixNano": int(span.start_time * 1e9),
            "endTimeUnixNano": int((span.end_time or time.time()) * 1e9),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in span.attributes.items()
            ],
            "events": [
                {
                    "name": e["name"],
                    "timeUnixNano": int(e["timestamp"] * 1e9),
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in e.get("attributes", {}).items()
                    ],
                }
                for e in span.events
            ],
            "status": {
                "code": 1 if span.status == SpanStatus.OK else 2 if span.status == SpanStatus.ERROR else 0,
                "message": span.status_message,
            },
        }

    def _kind_to_otlp(self, kind: SpanKind) -> int:
        """Convert SpanKind to OTLP numeric value."""
        mapping = {
            SpanKind.INTERNAL: 1,
            SpanKind.SERVER: 2,
            SpanKind.CLIENT: 3,
            SpanKind.PRODUCER: 4,
            SpanKind.CONSUMER: 5,
        }
        return mapping.get(kind, 0)


class Tracer:
    """
    Tracer for creating and managing spans.

    Each component should have its own tracer with a unique name.
    """

    def __init__(
        self,
        name: str,
        exporter: SpanExporter | None = None,
        parent_context: SpanContext | None = None,
    ) -> None:
        self._name = name
        self._exporter = exporter or ConsoleSpanExporter()
        self._parent_context = parent_context
        self._current_span: Span | None = None
        self._span_stack: list[Span] = []

    @property
    def name(self) -> str:
        return self._name

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """
        Start a new span as the current span.

        Usage:
            with tracer.start_as_current_span("operation") as span:
                span.set_attribute("key", "value")
                # ... do work
        """
        span = self.start_span(name, kind, attributes)
        self._span_stack.append(span)
        old_current = self._current_span
        self._current_span = span

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            span.end()
            self._export_span(span)
            self._span_stack.pop()
            self._current_span = old_current

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span."""
        # Determine parent
        parent_span_id = None
        trace_id = uuid4().hex

        if self._current_span:
            parent_span_id = self._current_span.span_id
            trace_id = self._current_span.trace_id
        elif self._parent_context:
            parent_span_id = self._parent_context.span_id
            trace_id = self._parent_context.trace_id

        span = Span(
            name=f"{self._name}.{name}",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            kind=kind,
            attributes=attributes or {},
        )

        return span

    def get_current_span(self) -> Span | None:
        """Get the current active span."""
        return self._current_span

    def _export_span(self, span: Span) -> None:
        """Export a completed span."""
        if self._exporter:
            self._exporter.export([span])


class TracerProvider:
    """
    Singleton provider for tracers.

    Manages tracer instances and global configuration.
    """

    _instance: TracerProvider | None = None
    _lock = threading.Lock()

    def __new__(cls) -> TracerProvider:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._tracers: dict[str, Tracer] = {}
        self._default_exporter: SpanExporter | None = None
        self._enabled = True
        self._initialized = True

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def set_exporter(self, exporter: SpanExporter) -> None:
        """Set the default exporter for new tracers."""
        self._default_exporter = exporter

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable tracing globally."""
        self._enabled = enabled

    def get_tracer(
        self,
        name: str,
        exporter: SpanExporter | None = None,
    ) -> Tracer:
        """Get or create a tracer with the given name."""
        if name not in self._tracers:
            self._tracers[name] = Tracer(
                name=name,
                exporter=exporter or self._default_exporter,
            )
        return self._tracers[name]


# Global functions for convenience


def get_tracer(name: str) -> Tracer:
    """Get a tracer for the given component name."""
    return TracerProvider().get_tracer(name)


def configure_tracing(
    exporter: SpanExporter | None = None,
    enabled: bool = True,
) -> None:
    """Configure global tracing settings."""
    provider = TracerProvider()
    if exporter:
        provider.set_exporter(exporter)
    provider.set_enabled(enabled)


def trace_function(
    span_name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Callable[[F], F]:
    """
    Decorator to trace a function.

    Usage:
        @trace_function("process_order")
        def process_order(order):
            ...

        @trace_function()  # Uses function name as span name
        async def async_operation():
            ...
    """
    def decorator(func: F) -> F:
        name = span_name or func.__name__
        tracer = get_tracer(func.__module__ or "unknown")

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(name, kind) as span:
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(name, kind) as span:
                return await func(*args, **kwargs)

        if asyncio_available():
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def asyncio_available() -> bool:
    """Check if asyncio is available."""
    try:
        import asyncio
        return True
    except ImportError:
        return False


# Pre-defined tracers for common components
def get_trading_tracer() -> Tracer:
    """Get tracer for trading engine."""
    return get_tracer("trading_engine")


def get_execution_tracer() -> Tracer:
    """Get tracer for order execution."""
    return get_tracer("execution")


def get_model_tracer() -> Tracer:
    """Get tracer for model inference."""
    return get_tracer("model")


def get_data_tracer() -> Tracer:
    """Get tracer for data pipeline."""
    return get_tracer("data")
