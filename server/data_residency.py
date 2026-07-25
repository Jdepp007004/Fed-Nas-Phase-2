"""
server/data_residency.py
Data residency controls for the _pending_updates lifecycle (Phase 3 — C6).

HIPAA-compliant FL platforms must ensure that:
  1. Client model weights (which may encode patient-level signal) are held
     in memory only for the duration of a single aggregation round.
  2. A hard TTL (time-to-live) forces deletion of stale updates so that
     a crashed round does not leak data across rounds.
  3. An audit log records every write/delete of pending updates.

This module wraps the in-memory `_pending_updates` buffer in
`project_router.py` with TTL-aware, audited operations.

Architecture
------------
    ResidencyManager  ←  called by project_router when adding/clearing updates
        │
        ├── _store   : {proj_id: {user_id: {"weight_hash": str, "inserted_at": float}}}
        │              (note: actual weight tensors stay in project_router's buffer)
        │
        ├── TTL enforcement via background watchdog thread
        │
        └── audit()  appends to rounds_history with event="residency_purge"

Public API
----------
    from data_residency import ResidencyManager
    rm = ResidencyManager(ttl_seconds=300)

    # When a client submits an update:
    rm.record_arrival(proj_id, user_id, weight_hash)

    # After a round completes / on explicit clear:
    rm.purge_round(proj_id)

    # Called by the watchdog to expire stale pending updates:
    stale = rm.find_stale()
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = int(os.getenv("PENDING_UPDATE_TTL_SECONDS", "300"))  # 5 minutes


class ResidencyManager:
    """
    TTL-aware residency tracker for in-flight pending model updates.

    All recorded arrivals are evicted automatically after `ttl_seconds`.
    Purges are logged to the rounds_history audit trail.

    Parameters
    ----------
    ttl_seconds   : int — maximum age (seconds) of a pending update
    purge_callback: callable(proj_id, expired_user_ids) — called on TTL eviction
                    Use this to remove entries from the in-memory buffer.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        purge_callback: Optional[Callable] = None,
    ) -> None:
        self.ttl = ttl_seconds
        self._purge_callback = purge_callback
        self._store: dict[str, dict] = {}   # proj_id → {user_id → {"hash", "ts"}}
        self._lock = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_arrival(
        self, proj_id: str, user_id: str, weights: dict
    ) -> str:
        """
        Record that `user_id` submitted an update for `proj_id`.

        Parameters
        ----------
        proj_id  : project UUID
        user_id  : hospital UUID
        weights  : dict of weight arrays — a SHA-256 fingerprint is stored (not the weights)

        Returns
        -------
        str — SHA-256 fingerprint of the weights (for audit)
        """
        wh = _weight_hash(weights)
        with self._lock:
            self._store.setdefault(proj_id, {})[user_id] = {
                "hash": wh,
                "inserted_at": time.monotonic(),
                "inserted_utc": _utcnow(),
            }
        logger.debug("Residency: recorded update proj=%s user=%s hash=%s", proj_id, user_id, wh[:8])
        return wh

    def purge_round(self, proj_id: str) -> int:
        """
        Explicitly purge all residency records for `proj_id` after a round.
        Returns the count of records purged.
        """
        with self._lock:
            records = self._store.pop(proj_id, {})
        n = len(records)
        if n:
            self._audit_purge(proj_id, list(records.keys()), reason="round_complete")
            logger.info("Residency: purged %d records for proj=%s (round complete)", n, proj_id)
        return n

    def find_stale(self) -> dict[str, list[str]]:
        """
        Return a mapping of proj_id → [stale user_ids] where
        `inserted_at` is older than `ttl_seconds`.
        """
        cutoff = time.monotonic() - self.ttl
        stale: dict[str, list[str]] = {}
        with self._lock:
            for proj_id, users in self._store.items():
                old = [uid for uid, rec in users.items() if rec["inserted_at"] < cutoff]
                if old:
                    stale[proj_id] = old
        return stale

    def evict_stale(self) -> int:
        """
        Evict all stale records (older than TTL). Calls purge_callback if set.
        Returns total number of records evicted.
        """
        stale = self.find_stale()
        total = 0
        for proj_id, user_ids in stale.items():
            with self._lock:
                for uid in user_ids:
                    self._store.get(proj_id, {}).pop(uid, None)
            total += len(user_ids)
            self._audit_purge(proj_id, user_ids, reason="ttl_expired")
            if self._purge_callback:
                try:
                    self._purge_callback(proj_id, user_ids)
                except Exception as e:
                    logger.warning("Residency purge_callback failed: %s", e)
        if total:
            logger.warning("Residency: evicted %d stale update(s) due to TTL", total)
        return total

    def pending_count(self, proj_id: str) -> int:
        """Return the number of pending updates tracked for this project."""
        with self._lock:
            return len(self._store.get(proj_id, {}))

    def start_watchdog(self, interval_seconds: int = 60) -> threading.Thread:
        """
        Start a daemon thread that periodically calls `evict_stale()`.
        Returns the thread (already started).
        """
        def _loop():
            while True:
                time.sleep(interval_seconds)
                try:
                    self.evict_stale()
                except Exception as e:
                    logger.error("Residency watchdog error: %s", e)

        t = threading.Thread(target=_loop, daemon=True, name="residency-watchdog")
        t.start()
        logger.info("Residency watchdog started (interval=%ds, TTL=%ds)", interval_seconds, self.ttl)
        return t

    # ── Internal ──────────────────────────────────────────────────────────────

    def _audit_purge(self, proj_id: str, user_ids: list, reason: str) -> None:
        """Append a residency_purge record to rounds_history."""
        try:
            from db_handler import append_round_history
            append_round_history({
                "proj_id":       proj_id,
                "event":         "residency_purge",
                "reason":        reason,
                "purged_users":  user_ids,
                "timestamp":     _utcnow(),
                "round":         None,
            })
        except Exception as e:
            logger.warning("Residency audit write failed: %s", e)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _weight_hash(weights: dict) -> str:
    """SHA-256 fingerprint of all weight values (for audit, not security)."""
    h = hashlib.sha256()
    for key in sorted(weights.keys()):
        val = weights[key]
        if hasattr(val, "tobytes"):
            h.update(key.encode())
            h.update(np.array(val, dtype=np.float32).tobytes())
        else:
            h.update(str(val).encode())
    return h.hexdigest()


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
