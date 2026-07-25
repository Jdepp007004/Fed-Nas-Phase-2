"""
server/reputation.py
Per-client reputation tracking for the FL Platform (Phase 3 — B5).

Each round a client submits an update, its reputation score is updated based
on how well its update aligns with the global aggregated result. Clients with
consistently divergent updates (potential Byzantines) accumulate a low score
and are eventually suspended from participation.

Reputation Model
----------------
    score_new = alpha * alignment + (1 - alpha) * score_old

where:
    alignment = cosine_similarity(client_update, global_aggregate)
    alpha     = REPUTATION_ALPHA (smoothing factor, default 0.3)

A client is suspended when its score drops below REPUTATION_THRESHOLD.

Scores are persisted in db["client_reputations"] as:
    [{"client_id": str, "proj_id": str, "score": float, "rounds": int,
      "suspended": bool, "last_updated": str}]

Public API
----------
    from reputation import ReputationManager
    rm = ReputationManager()

    # After a round:
    rm.update_scores(proj_id, client_updates, global_aggregate)

    # Before accepting an update:
    if rm.is_suspended(user_id, proj_id):
        raise HTTPException(403, "Client suspended due to low reputation")
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default hyperparameters (can be overridden via env vars)
REPUTATION_ALPHA     = float(os.getenv("REPUTATION_ALPHA", "0.3"))
REPUTATION_THRESHOLD = float(os.getenv("REPUTATION_THRESHOLD", "0.2"))
REPUTATION_INIT      = float(os.getenv("REPUTATION_INIT", "1.0"))   # new clients start at 1.0


class ReputationManager:
    """
    Tracks per-client reputation scores across rounds.
    Backed by db_handler (transparent JSON/PG).
    """

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _read_reputations(self) -> list[dict]:
        from db_handler import read_db
        return read_db().get("client_reputations", [])

    def _write_reputations(self, records: list[dict]) -> None:
        from db_handler import read_db, write_db
        db = read_db()
        db["client_reputations"] = records
        write_db(db)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_score(self, client_id: str, proj_id: str) -> float:
        """Return current reputation score for (client_id, proj_id). New clients get INIT."""
        for r in self._read_reputations():
            if r["client_id"] == client_id and r["proj_id"] == proj_id:
                return float(r["score"])
        return REPUTATION_INIT

    def is_suspended(self, client_id: str, proj_id: str) -> bool:
        """Return True if the client's reputation is below the threshold."""
        for r in self._read_reputations():
            if r["client_id"] == client_id and r["proj_id"] == proj_id:
                return bool(r.get("suspended", False))
        return False

    def update_scores(
        self,
        proj_id: str,
        client_updates: list[dict],
        client_ids: list[str],
        global_aggregate: dict,
        alpha: float = REPUTATION_ALPHA,
        threshold: float = REPUTATION_THRESHOLD,
    ) -> dict[str, float]:
        """
        Update reputation scores for all clients that submitted this round.

        Parameters
        ----------
        proj_id          : project UUID
        client_updates   : list of weight dicts (same order as client_ids)
        client_ids       : list of user_id strings
        global_aggregate : aggregated weight dict for this round
        alpha            : smoothing factor for EMA score update
        threshold        : suspension threshold

        Returns
        -------
        dict[client_id → new_score]
        """
        if len(client_updates) != len(client_ids):
            raise ValueError("client_updates and client_ids must have the same length")

        # Flatten global aggregate into a reference vector
        global_flat = _flatten(global_aggregate)
        records = self._read_reputations()
        record_index = {(r["client_id"], r["proj_id"]): i for i, r in enumerate(records)}

        new_scores: dict[str, float] = {}
        now = _utcnow()

        for client_id, upd in zip(client_ids, client_updates):
            # Compute alignment: cosine similarity with global aggregate
            upd_flat = _flatten(upd)
            alignment = float(_cosine_sim(upd_flat, global_flat))

            # EMA update
            old_score = self.get_score(client_id, proj_id)
            new_score = float(alpha * alignment + (1.0 - alpha) * old_score)
            new_score = max(0.0, min(1.0, new_score))
            suspended = new_score < threshold

            if suspended and not self.is_suspended(client_id, proj_id):
                logger.warning(
                    "Reputation: client=%s proj=%s score=%.3f → SUSPENDED",
                    client_id, proj_id, new_score,
                )

            key = (client_id, proj_id)
            entry = {
                "client_id":    client_id,
                "proj_id":      proj_id,
                "score":        new_score,
                "rounds":       old_score,  # overwritten below
                "suspended":    suspended,
                "last_updated": now,
            }

            if key in record_index:
                old = records[record_index[key]]
                entry["rounds"] = old.get("rounds", 0) + 1
                records[record_index[key]] = entry
            else:
                entry["rounds"] = 1
                records.append(entry)
                record_index[key] = len(records) - 1

            new_scores[client_id] = new_score

        self._write_reputations(records)
        return new_scores

    def reinstate(self, client_id: str, proj_id: str) -> bool:
        """Manually reinstate a suspended client (admin action). Returns True if found."""
        records = self._read_reputations()
        found = False
        for r in records:
            if r["client_id"] == client_id and r["proj_id"] == proj_id:
                r["suspended"] = False
                r["score"] = REPUTATION_INIT
                found = True
        if found:
            self._write_reputations(records)
            logger.info("Reputation: client=%s proj=%s reinstated", client_id, proj_id)
        return found

    def list_scores(self, proj_id: str) -> list[dict]:
        """Return all reputation records for a project."""
        return [r for r in self._read_reputations() if r["proj_id"] == proj_id]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _flatten(weights: dict) -> np.ndarray:
    parts = [np.array(v, dtype=np.float32).flatten() for v in weights.values()]
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat vectors. Returns 0 if either is zero."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
