"""
server/metrics.py
Prometheus metrics for the FL Platform.

Exposes all metrics via prometheus_client. The `/metrics` endpoint
in main.py serves these.

Metrics
-------
fl_rounds_completed_total{proj_id, status}   Counter
fl_round_duration_seconds{proj_id}           Histogram
fl_active_clients{proj_id}                   Gauge
fl_pending_updates{proj_id}                  Gauge
fl_api_requests_total{endpoint, method, status_code}  Counter
fl_api_latency_seconds{endpoint}             Histogram
fl_global_val_rmse{proj_id}                  Gauge
fl_global_tox_accuracy{proj_id}              Gauge
fl_global_auc{proj_id}                       Gauge
"""
from __future__ import annotations

import functools
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)

_PROM_AVAILABLE = False
try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


def _noop(*args, **kwargs):
    """No-op stub returned when prometheus_client is not installed."""
    class _Stub:
        def labels(self, **kw): return self
        def inc(self, *a, **kw): pass
        def dec(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def observe(self, *a, **kw): pass
        def time(self): return _CtxStub()
    class _CtxStub:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Stub()


if _PROM_AVAILABLE:
    # ── Round metrics ─────────────────────────────────────────────────────────
    ROUNDS_COMPLETED = Counter(
        "fl_rounds_completed_total",
        "Number of completed federated rounds",
        ["proj_id", "status"],          # status: success | error
    )
    ROUND_DURATION = Histogram(
        "fl_round_duration_seconds",
        "Wall-clock time for a complete federated round",
        ["proj_id"],
        buckets=[30, 60, 120, 300, 600, 900, 1800],
    )

    # ── Client metrics ────────────────────────────────────────────────────────
    ACTIVE_CLIENTS = Gauge(
        "fl_active_clients",
        "Number of clients currently in connected_clients",
        ["proj_id"],
    )
    PENDING_UPDATES = Gauge(
        "fl_pending_updates",
        "Number of client updates buffered and awaiting aggregation",
        ["proj_id"],
    )

    # ── API metrics ───────────────────────────────────────────────────────────
    API_REQUESTS = Counter(
        "fl_api_requests_total",
        "Total HTTP requests handled",
        ["endpoint", "method", "status_code"],
    )
    API_LATENCY = Histogram(
        "fl_api_latency_seconds",
        "HTTP request latency",
        ["endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    # ── Model quality metrics ─────────────────────────────────────────────────
    GLOBAL_VAL_RMSE = Gauge(
        "fl_global_val_rmse",
        "Latest global model validation RMSE",
        ["proj_id"],
    )
    GLOBAL_TOX_ACCURACY = Gauge(
        "fl_global_tox_accuracy",
        "Latest global model toxicity classification accuracy",
        ["proj_id"],
    )
    GLOBAL_AUC = Gauge(
        "fl_global_auc",
        "Latest global model binary AUC-ROC",
        ["proj_id"],
    )

else:
    # Stub all metrics with no-ops
    ROUNDS_COMPLETED    = _noop()
    ROUND_DURATION      = _noop()
    ACTIVE_CLIENTS      = _noop()
    PENDING_UPDATES     = _noop()
    API_REQUESTS        = _noop()
    API_LATENCY         = _noop()
    GLOBAL_VAL_RMSE     = _noop()
    GLOBAL_TOX_ACCURACY = _noop()
    GLOBAL_AUC          = _noop()


def record_round_metrics(proj_id: str, metrics: dict) -> None:
    """
    Update all model-quality gauges from a validate_global_model() result dict.
    Called at the end of round_lifecycle (or from tasks.py).
    """
    try:
        if "global_val_rmse" in metrics and metrics["global_val_rmse"] is not None:
            GLOBAL_VAL_RMSE.labels(proj_id=proj_id).set(metrics["global_val_rmse"])
        if "global_tox_accuracy" in metrics and metrics["global_tox_accuracy"] is not None:
            GLOBAL_TOX_ACCURACY.labels(proj_id=proj_id).set(metrics["global_tox_accuracy"])
        if "global_auc" in metrics and metrics["global_auc"] is not None:
            GLOBAL_AUC.labels(proj_id=proj_id).set(metrics["global_auc"])
    except Exception as e:
        logger.warning("Failed to record round metrics to Prometheus: %s", e)


def prometheus_response() -> tuple[bytes, str]:
    """
    Return (body_bytes, content_type) for the /metrics endpoint.
    If prometheus_client is not installed, returns a 501 placeholder.
    """
    if not _PROM_AVAILABLE:
        return b"# prometheus_client not installed\n", "text/plain"
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def track_api(endpoint: str) -> Callable:
    """
    Decorator that records API_REQUESTS and API_LATENCY for a FastAPI route.

    Usage::

        @router.get("/api/projects")
        @track_api("/api/projects")
        async def list_projects(...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            status = "200"
            try:
                result = await fn(*args, **kwargs)
                if hasattr(result, "status_code"):
                    status = str(result.status_code)
                return result
            except Exception as exc:
                status = "500"
                raise
            finally:
                elapsed = time.perf_counter() - t0
                try:
                    API_REQUESTS.labels(
                        endpoint=endpoint,
                        method="GET",
                        status_code=status,
                    ).inc()
                    API_LATENCY.labels(endpoint=endpoint).observe(elapsed)
                except Exception:
                    pass
        return wrapper
    return decorator
