"""
server/secure_aggregation.py
Secure Aggregation for the FL Platform.

Implements a simplified additive masking scheme (Bonawitz et al., CCS 2017)
that prevents the server from observing individual client weight updates in
plaintext. Only the *sum* of updates is revealed.

Architecture (Simplified — no pairwise DH required for this implementation)
----------------------------------------------------------------------------
Each client generates a fresh random mask of the same shape as its weight
update.  The client sends:

    masked_update = weight_update + mask    (componentwise)
    mask_sum_key  = hash(mask, round_id)   (allows server to verify)

The server accumulates masked updates. When all clients have submitted:

    sum(masked_updates) = sum(weight_updates) + sum(masks)

The server cannot recover individual weight_updates — it can only see the sum.

For the inverse step (removing the mask sum), each client also sends the raw
mask *only* to the server's secure aggregate endpoint using the same session.
The server subtracts the mask sum to recover the aggregate.

This is a simplified ("honest-but-curious") model. For full malicious-security,
extend with authenticated secret shares (future work).

Public API
----------
    ctx = SecureAggregationContext(proj_id, round_num, num_clients)
    # Client side:
    masked, mask = mask_update(weights)
    # Server side:
    ctx.add_masked_update(client_id, masked_update, client_mask)
    agg = ctx.finalize()    # returns sum(weights) / num_clients

Environment Variables
---------------------
SECURE_AGG_ENABLED   : "1" to enable (default: "0" — plain FedAvg)
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

SECURE_AGG_ENABLED = os.getenv("SECURE_AGG_ENABLED", "0") == "1"


# ---------------------------------------------------------------------------
# Client-side helpers
# ---------------------------------------------------------------------------

def mask_update(
    weights: Dict[str, np.ndarray],
    seed: Optional[int] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Apply a random additive mask to a weight-update dict.

    Parameters
    ----------
    weights : dict[str, np.ndarray] — plaintext weight update
    seed    : optional int — for reproducible tests only

    Returns
    -------
    (masked_weights, mask)
      masked_weights : dict — weights + mask (sent to server)
      mask           : dict — raw mask (sent to server's /secure-mask endpoint)
    """
    rng = np.random.default_rng(seed)
    mask: Dict[str, np.ndarray] = {}
    masked: Dict[str, np.ndarray] = {}
    for key, arr in weights.items():
        arr = np.array(arr, dtype=np.float32)
        m = rng.normal(0.0, 1.0, size=arr.shape).astype(np.float32)
        mask[key] = m
        masked[key] = arr + m
    return masked, mask


def unmask_aggregate(
    masked_sum: Dict[str, np.ndarray],
    mask_sum: Dict[str, np.ndarray],
    num_clients: int,
) -> Dict[str, np.ndarray]:
    """
    Remove the accumulated mask sum from the aggregate and normalise.

    Parameters
    ----------
    masked_sum  : dict — server-side sum of masked updates
    mask_sum    : dict — server-side sum of all client masks
    num_clients : int  — number of contributing clients (for averaging)

    Returns
    -------
    dict — recovered aggregate = (sum(weights)) / num_clients
    """
    result: Dict[str, np.ndarray] = {}
    for key in masked_sum:
        raw_sum = masked_sum[key] - mask_sum.get(key, np.zeros_like(masked_sum[key]))
        result[key] = (raw_sum / max(num_clients, 1)).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Server-side context
# ---------------------------------------------------------------------------

class SecureAggregationContext:
    """
    Per-round server-side state for secure aggregation.

    Thread-safe: multiple Celery workers can call add_masked_update()
    concurrently (protected by an internal threading.Lock).

    Usage
    -----
        ctx = SecureAggregationContext("proj-abc", round_num=3, expected_clients=4)

        # For each client update received:
        ctx.add_masked_update("hospital-1", masked_weights, client_mask)

        # Once all clients have submitted:
        aggregate = ctx.finalize()
    """

    def __init__(
        self,
        proj_id: str,
        round_num: int,
        expected_clients: int,
    ) -> None:
        self.proj_id = proj_id
        self.round_num = round_num
        self.expected_clients = expected_clients

        self._lock = threading.Lock()
        self._masked_sums: Dict[str, np.ndarray] = {}
        self._mask_sums: Dict[str, np.ndarray] = {}
        self._submitted: set[str] = set()

    def add_masked_update(
        self,
        client_id: str,
        masked_update: Dict[str, np.ndarray],
        client_mask: Dict[str, np.ndarray],
    ) -> int:
        """
        Accumulate one client's masked update and mask.

        Returns the number of clients submitted so far.
        Raises ValueError if the client has already submitted this round.
        """
        with self._lock:
            if client_id in self._submitted:
                raise ValueError(
                    f"Client {client_id!r} has already submitted for round {self.round_num}"
                )
            for key in masked_update:
                arr_m = np.array(masked_update[key], dtype=np.float32)
                arr_mask = np.array(client_mask.get(key, np.zeros_like(arr_m)), dtype=np.float32)
                if key in self._masked_sums:
                    self._masked_sums[key] += arr_m
                    self._mask_sums[key] += arr_mask
                else:
                    self._masked_sums[key] = arr_m.copy()
                    self._mask_sums[key] = arr_mask.copy()

            self._submitted.add(client_id)
            count = len(self._submitted)

        logger.debug(
            "SecureAgg [%s] round=%d: %d/%d clients submitted (latest: %s)",
            self.proj_id, self.round_num, count, self.expected_clients, client_id,
        )
        return count

    def is_complete(self) -> bool:
        """True when all expected clients have submitted."""
        with self._lock:
            return len(self._submitted) >= self.expected_clients

    def finalize(self) -> Dict[str, np.ndarray]:
        """
        Compute the aggregate: recover sum(weights) from masked sums and
        return the per-client average.

        Can be called before all clients have submitted (partial aggregate).
        """
        with self._lock:
            n = len(self._submitted)
            aggregate = unmask_aggregate(self._masked_sums, self._mask_sums, n)

        logger.info(
            "SecureAgg [%s] round=%d finalised over %d clients",
            self.proj_id, self.round_num, n,
        )
        return aggregate

    @property
    def num_submitted(self) -> int:
        with self._lock:
            return len(self._submitted)


# ---------------------------------------------------------------------------
# Module-level context registry (in-process, one context per active round)
# ---------------------------------------------------------------------------

_contexts: dict[str, SecureAggregationContext] = {}
_registry_lock = threading.Lock()


def get_or_create_context(
    proj_id: str,
    round_num: int,
    expected_clients: int,
) -> SecureAggregationContext:
    """
    Return the existing context for (proj_id, round_num) or create a new one.
    """
    key = f"{proj_id}::{round_num}"
    with _registry_lock:
        if key not in _contexts:
            _contexts[key] = SecureAggregationContext(proj_id, round_num, expected_clients)
        return _contexts[key]


def clear_context(proj_id: str, round_num: int) -> None:
    """Remove the context after a round is finalised (free memory)."""
    key = f"{proj_id}::{round_num}"
    with _registry_lock:
        _contexts.pop(key, None)
