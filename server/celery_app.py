"""
server/celery_app.py
Celery application instance for the FL Platform.

Activated when CELERY_BROKER_URL is set (typically pointing to Redis).
When not set, the app is still importable but tasks run synchronously
via task_always_eager=True so development and testing work without a broker.

Usage (production)
------------------
    # Start a Celery worker:
    celery -A celery_app worker --loglevel=info --concurrency=2

    # Inspect registered tasks:
    celery -A celery_app inspect registered

Environment variables
---------------------
CELERY_BROKER_URL   : Redis URL, e.g. redis://localhost:6379/0
                      Falls back to "memory://" (in-process) if not set.
CELERY_RESULT_URL   : Backend URL for task results.
                      Defaults to CELERY_BROKER_URL.
"""
from __future__ import annotations

import os
import logging

from celery import Celery
from celery.utils.log import get_task_logger

logger = logging.getLogger(__name__)

_BROKER = os.getenv("CELERY_BROKER_URL", "memory://")
_BACKEND = os.getenv("CELERY_RESULT_URL", os.getenv("CELERY_BROKER_URL", "cache+memory://"))

app = Celery(
    "fl_platform",
    broker=_BROKER,
    backend=_BACKEND,
    include=["tasks"],      # auto-discover tasks module
)

app.conf.update(
    # ── Serialisation ─────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Reliability ───────────────────────────────────────────────────────────
    task_acks_late=True,            # ack after completion, not on receipt
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,   # one task at a time per worker slot

    # ── Timeouts ─────────────────────────────────────────────────────────────
    task_soft_time_limit=1800,      # 30 min soft limit
    task_time_limit=2100,           # 35 min hard kill

    # ── Dev/test mode ─────────────────────────────────────────────────────────
    # task_always_eager is controlled dynamically in tasks.py based on CELERY_BROKER_URL
    task_always_eager=not bool(os.getenv("CELERY_BROKER_URL")),
    task_eager_propagates=True,

    # ── Result expiry ─────────────────────────────────────────────────────────
    result_expires=86400,           # 24 h

    # ── Timezone ──────────────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,
)

task_logger = get_task_logger(__name__)
