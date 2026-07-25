"""
server/otel_setup.py
OpenTelemetry trace and metric setup for the FL Platform.

Activated when OTEL_EXPORTER_OTLP_ENDPOINT is set.
Falls back to a no-op tracer so callers never crash.

Usage
-----
    from otel_setup import get_tracer, get_meter

    tracer = get_tracer()
    with tracer.start_as_current_span("round_lifecycle") as span:
        span.set_attribute("fl.proj_id", proj_id)
        span.set_attribute("fl.round", round_num)
        ...

Environment variables
---------------------
OTEL_EXPORTER_OTLP_ENDPOINT  : e.g. http://jaeger:4318
OTEL_SERVICE_NAME             : default "fl-platform-server"
"""
from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "fl-platform-server")
_ENDPOINT     = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

_tracer = None
_meter  = None
_otel_initialised = False


def _try_init() -> bool:
    """Attempt to initialise OTel SDK. Returns True on success."""
    global _tracer, _meter, _otel_initialised

    if _otel_initialised:
        return _tracer is not None

    _otel_initialised = True

    if not _ENDPOINT:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — OTel tracing disabled.")
        return False

    try:
        from opentelemetry import trace, metrics as otel_metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": _SERVICE_NAME})

        # ── Traces ────────────────────────────────────────────────────────────
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=f"{_ENDPOINT}/v1/traces")
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer(_SERVICE_NAME)

        # ── Metrics ───────────────────────────────────────────────────────────
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=f"{_ENDPOINT}/v1/metrics"),
            export_interval_millis=60_000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        otel_metrics.set_meter_provider(meter_provider)
        _meter = otel_metrics.get_meter(_SERVICE_NAME)

        logger.info("OpenTelemetry initialised → %s", _ENDPOINT)
        return True

    except ImportError as e:
        logger.warning(
            "opentelemetry SDK not installed — tracing disabled. "
            "Install: pip install opentelemetry-sdk opentelemetry-exporter-otlp. "
            "Error: %s", e
        )
        return False
    except Exception as e:
        logger.error("OTel initialisation failed: %s", e)
        return False


def get_tracer():
    """
    Return an OpenTelemetry Tracer, or a no-op tracer if OTel is not configured.
    Safe to call from any module — never raises.
    """
    _try_init()
    if _tracer is not None:
        return _tracer
    # Return a no-op tracer
    return _NoOpTracer()


def get_meter():
    """
    Return an OpenTelemetry Meter, or a no-op meter if OTel is not configured.
    Safe to call from any module — never raises.
    """
    _try_init()
    if _meter is not None:
        return _meter
    return _NoOpMeter()


# ─── No-op stubs ─────────────────────────────────────────────────────────────

class _NoOpSpan:
    def set_attribute(self, *a, **kw): pass
    def record_exception(self, *a, **kw): pass
    def set_status(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kw):
        return _NoOpSpan()
    def start_span(self, name: str, **kw):
        return _NoOpSpan()


class _NoOpCounter:
    def add(self, *a, **kw): pass


class _NoOpHistogram:
    def record(self, *a, **kw): pass


class _NoOpMeter:
    def create_counter(self, *a, **kw): return _NoOpCounter()
    def create_histogram(self, *a, **kw): return _NoOpHistogram()


def instrument_round_lifecycle(proj_id: str, round_num: int):
    """
    Context manager that wraps round_lifecycle() with an OTel span.

    Usage::

        with instrument_round_lifecycle(proj_id, round_num) as span:
            span.set_attribute("fl.num_clients", len(updates))
            round_lifecycle(proj_id, updates, db)
    """
    tracer = get_tracer()
    span_ctx = tracer.start_as_current_span("fl.round_lifecycle")
    try:
        span = span_ctx.__enter__()
        span.set_attribute("fl.proj_id", proj_id)
        span.set_attribute("fl.round", round_num)
        return span_ctx
    except Exception:
        return _NoOpSpan()
