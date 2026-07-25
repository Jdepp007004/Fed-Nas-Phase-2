"""
server/tasks.py
Celery task definitions for the FL Platform.

Tasks
-----
dispatch_round(proj_id, updates_buffer, db_snapshot)
    Wraps project_router.round_lifecycle() as a Celery task.
    Adds state-machine transitions and Prometheus/OTel instrumentation.

retry_stalled_rounds()
    Periodic task (beat): finds projects stuck in "aggregating" state
    longer than STALL_TIMEOUT_SECONDS and transitions them to "error".
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from celery_app import app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)
STALL_TIMEOUT_SECONDS = int(os.getenv("ROUND_STALL_TIMEOUT_S", "7200"))  # 2 h default


@app.task(
    bind=True,
    name="fl_platform.dispatch_round",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def dispatch_round(
    self,
    proj_id: str,
    updates_buffer: list[dict],
    db_snapshot: dict,
) -> dict[str, Any]:
    """
    Execute one federated round as a Celery task.

    Parameters
    ----------
    proj_id       : Project UUID
    updates_buffer: List of decrypted client update dicts
    db_snapshot   : DB snapshot taken at trigger time (same as round_lifecycle signature)

    Returns
    -------
    dict with keys: proj_id, round, success, elapsed_s
    """
    import sys
    import os as _os
    _os.makedirs(_os.path.join(_os.path.dirname(__file__), "models"), exist_ok=True)
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))

    from project_router import round_lifecycle
    from round_state import RoundState

    # ── State machine: IDLE/COLLECTING → AGGREGATING ──────────────────────────
    _set_project_round_state(proj_id, RoundState.AGGREGATING)

    t0 = time.monotonic()
    try:
        round_lifecycle(proj_id, updates_buffer, db_snapshot)
        elapsed = time.monotonic() - t0

        # ── State machine: AGGREGATING → DONE ─────────────────────────────────
        _set_project_round_state(proj_id, RoundState.DONE)
        _set_project_round_state(proj_id, RoundState.IDLE)  # immediately ready

        logger.info("Round complete for %s in %.1fs", proj_id, elapsed)

        # ── Prometheus instrumentation ─────────────────────────────────────────
        try:
            from metrics import ROUNDS_COMPLETED, ROUND_DURATION
            ROUNDS_COMPLETED.labels(proj_id=proj_id, status="success").inc()
            ROUND_DURATION.labels(proj_id=proj_id).observe(elapsed)
        except Exception:
            pass

        return {"proj_id": proj_id, "success": True, "elapsed_s": elapsed}

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.exception("Round failed for %s after %.1fs: %s", proj_id, elapsed, exc)

        # ── State machine: AGGREGATING → ERROR ────────────────────────────────
        _set_project_round_state(proj_id, RoundState.ERROR)

        try:
            from metrics import ROUNDS_COMPLETED
            ROUNDS_COMPLETED.labels(proj_id=proj_id, status="error").inc()
        except Exception:
            pass

        # Retry up to max_retries times
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.error("Max retries exceeded for project %s — round marked ERROR", proj_id)
            return {"proj_id": proj_id, "success": False, "elapsed_s": elapsed,
                    "error": str(exc)}


def _set_project_round_state(proj_id: str, state) -> None:
    """Best-effort state transition — never raises."""
    try:
        from db_handler import update_project
        update_project(proj_id, {"round_state": state.value})
    except Exception as e:
        logger.warning("Could not update round_state for %s: %s", proj_id, e)


@app.task(
    name="fl_platform.retry_stalled_rounds",
)
def retry_stalled_rounds() -> dict:
    """
    Periodic Celery Beat task — finds projects stuck in AGGREGATING and
    transitions them to ERROR so they do not block future rounds indefinitely.
    """
    try:
        from db_handler import read_db, update_project
        import datetime

        db = read_db()
        stalled = 0
        for proj in db.get("projects", []):
            if proj.get("round_state") == "aggregating":
                # We don't have a start-time for the aggregation in JSON mode,
                # so we conservatively mark any aggregating project as stalled.
                update_project(proj["proj_id"], {"round_state": "error"})
                logger.warning(
                    "Marked project %s as ERROR (stalled in aggregating)",
                    proj["proj_id"],
                )
                stalled += 1
        return {"stalled_reset": stalled}
    except Exception as e:
        logger.exception("retry_stalled_rounds failed: %s", e)
        return {"stalled_reset": 0, "error": str(e)}
