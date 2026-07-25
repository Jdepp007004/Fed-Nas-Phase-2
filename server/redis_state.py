"""
server/redis_state.py
Redis-backed persistent state for _pending_updates and _velocity_state.

Activated when REDIS_URL is set in the environment.
When Redis is unavailable the system falls back to in-memory dicts
(identical to the existing project_router.py behaviour) so development
and testing work without a Redis instance.

Public API mirrors the dict-interface used in project_router.py:

    state = RedisState()

    # Pending updates buffer
    state.push_update("proj-1", update_dict)
    updates = state.pop_all_updates("proj-1")   # atomic flush
    count   = state.count_updates("proj-1")

    # Velocity (momentum) state
    state.set_velocity("proj-1", velocity_dict)
    vel = state.get_velocity("proj-1")           # {} if not set
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

_REDIS_AVAILABLE = False
try:
    import redis as _redis_mod
    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


class RedisState:
    """
    Thread-safe state manager for round_lifecycle buffers.

    Uses Redis RPUSH/LPOP for the pending-updates queue so that multiple
    Celery workers can safely read and flush the buffer atomically via a
    Lua script. Falls back to an in-memory threading.Lock-based dict when
    REDIS_URL is not set or redis-py is not installed.
    """

    _PREFIX_UPDATES  = "fl:updates:"
    _PREFIX_VELOCITY = "fl:velocity:"
    _FLUSH_SCRIPT = """
        local key   = KEYS[1]
        local items = redis.call('LRANGE', key, 0, -1)
        redis.call('DEL', key)
        return items
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        url = redis_url or os.getenv("REDIS_URL")
        self._use_redis = _REDIS_AVAILABLE and bool(url)

        if self._use_redis:
            self._r = _redis_mod.from_url(url, decode_responses=True)
            self._flush_script = self._r.register_script(self._FLUSH_SCRIPT)
        else:
            # In-memory fallback — compatible with existing project_router behaviour
            self._lock = threading.Lock()
            self._updates: dict[str, list] = {}
            self._velocity: dict[str, dict] = {}

    # ── Pending updates ───────────────────────────────────────────────────────

    def push_update(self, proj_id: str, update: dict) -> int:
        """
        Append an update dict to the buffer.
        Returns the new queue length.
        """
        if self._use_redis:
            return self._r.rpush(
                self._PREFIX_UPDATES + proj_id,
                json.dumps(update, default=_json_default),
            )
        with self._lock:
            self._updates.setdefault(proj_id, []).append(update)
            return len(self._updates[proj_id])

    def count_updates(self, proj_id: str) -> int:
        """Return current number of buffered updates for proj_id."""
        if self._use_redis:
            return self._r.llen(self._PREFIX_UPDATES + proj_id)
        with self._lock:
            return len(self._updates.get(proj_id, []))

    def pop_all_updates(self, proj_id: str) -> list[dict]:
        """
        Atomically drain and return all buffered updates for proj_id.
        After this call the buffer for proj_id is empty.
        """
        if self._use_redis:
            raw_items = self._flush_script(
                keys=[self._PREFIX_UPDATES + proj_id], args=[]
            )
            return [json.loads(item) for item in (raw_items or [])]
        with self._lock:
            items = list(self._updates.pop(proj_id, []))
            return items

    # ── Velocity state ────────────────────────────────────────────────────────

    def set_velocity(self, proj_id: str, velocity: dict) -> None:
        """Persist momentum velocity dict for proj_id."""
        if self._use_redis:
            self._r.set(
                self._PREFIX_VELOCITY + proj_id,
                json.dumps(velocity, default=_json_default),
            )
        else:
            with self._lock:
                self._velocity[proj_id] = velocity

    def get_velocity(self, proj_id: str) -> dict:
        """Return persisted velocity dict, or {} if not set."""
        if self._use_redis:
            raw = self._r.get(self._PREFIX_VELOCITY + proj_id)
            return json.loads(raw) if raw else {}
        with self._lock:
            return dict(self._velocity.get(proj_id, {}))

    def delete_velocity(self, proj_id: str) -> None:
        """Remove velocity state (used when a project is reset)."""
        if self._use_redis:
            self._r.delete(self._PREFIX_VELOCITY + proj_id)
        else:
            with self._lock:
                self._velocity.pop(proj_id, None)

    # ── Health ────────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if the Redis connection is alive (or in-memory mode)."""
        if self._use_redis:
            try:
                return self._r.ping()
            except Exception:  # pragma: no cover
                return False
        return True  # in-memory always "alive"

    @property
    def backend(self) -> str:
        """Return 'redis' or 'memory' for observability."""
        return "redis" if self._use_redis else "memory"


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy arrays (stored as nested lists)."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# Module-level singleton — shared across all project_router invocations
_default_state: Optional[RedisState] = None


def get_state() -> RedisState:
    """Return (or lazily create) the module-level RedisState singleton."""
    global _default_state
    if _default_state is None:
        _default_state = RedisState()
    return _default_state
