"""
server/reconnection.py
Client reconnection protocol for the FL Platform (Phase 3 — R4).

When a client disconnects mid-round (network drop, crash, timeout), the
server must:
  1. Detect the gap (client did not submit within the round TTL).
  2. Allow the client to reconnect and resume from the current round number.
  3. Not duplicate the client's contribution if it already submitted.
  4. Optionally carry forward the client's last known update if the round
     completed without them (via `last_known_weights` field in DB).

Reconnection State Machine
---------------------------
    PENDING   → client joined but not yet submitted in the current round
    SUBMITTED → client submitted for the current round
    MISSED    → round closed without the client submitting
    RECONNECTED → client reconnected after a MISSED round

Public API
----------
    from reconnection import ReconnectionManager

    rm = ReconnectionManager()
    rm.mark_submitted(proj_id, user_id, round_num)
    rm.close_round(proj_id, round_num, submitted_clients)
    state = rm.get_state(proj_id, user_id)  # → "submitted" | "missed" | "pending"
    ctx = rm.reconnect(proj_id, user_id)    # → {"round": int, "action": str}
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ReconnectionManager:
    """
    Tracks per-client round participation and handles reconnections.

    State is stored in db["client_round_states"] as:
        [{"proj_id", "user_id", "round", "state", "updated_at"}]
    """

    _STATES = {"pending", "submitted", "missed", "reconnected"}

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _read_states(self) -> list[dict]:
        from db_handler import read_db
        return read_db().get("client_round_states", [])

    def _write_states(self, states: list[dict]) -> None:
        from db_handler import read_db, write_db
        db = read_db()
        db["client_round_states"] = states
        write_db(db)

    def _upsert(self, proj_id: str, user_id: str, round_num: int, state: str) -> None:
        states = self._read_states()
        for s in states:
            if s["proj_id"] == proj_id and s["user_id"] == user_id and s["round"] == round_num:
                s["state"] = state
                s["updated_at"] = _utcnow()
                self._write_states(states)
                return
        states.append({
            "proj_id":    proj_id,
            "user_id":    user_id,
            "round":      round_num,
            "state":      state,
            "updated_at": _utcnow(),
        })
        self._write_states(states)

    # ── Public API ────────────────────────────────────────────────────────────

    def mark_pending(self, proj_id: str, user_id: str, round_num: int) -> None:
        """Record that a client is expected to submit for this round."""
        self._upsert(proj_id, user_id, round_num, "pending")
        logger.debug("Reconnection: %s/%s round=%d → pending", proj_id, user_id, round_num)

    def mark_submitted(self, proj_id: str, user_id: str, round_num: int) -> None:
        """Record that a client successfully submitted its update."""
        self._upsert(proj_id, user_id, round_num, "submitted")
        logger.debug("Reconnection: %s/%s round=%d → submitted", proj_id, user_id, round_num)

    def close_round(
        self, proj_id: str, round_num: int, submitted_clients: list[str]
    ) -> list[str]:
        """
        Mark clients that did NOT submit as "missed".

        Parameters
        ----------
        proj_id           : project UUID
        round_num         : round number that just closed
        submitted_clients : list of user_ids that submitted

        Returns
        -------
        list[str] — user_ids marked as "missed"
        """
        states = self._read_states()
        missed = []
        submitted_set = set(submitted_clients)
        for s in states:
            if (s["proj_id"] == proj_id and s["round"] == round_num
                    and s["state"] == "pending"):
                if s["user_id"] not in submitted_set:
                    s["state"] = "missed"
                    s["updated_at"] = _utcnow()
                    missed.append(s["user_id"])
        self._write_states(states)
        if missed:
            logger.warning(
                "Reconnection: %d client(s) missed round %d for proj=%s: %s",
                len(missed), round_num, proj_id, missed,
            )
        return missed

    def get_state(self, proj_id: str, user_id: str, round_num: Optional[int] = None) -> str:
        """
        Return the most recent state for a (proj_id, user_id) pair.

        If round_num is given, looks up that specific round.
        Returns "unknown" if no record found.
        """
        states = self._read_states()
        matches = [
            s for s in states
            if s["proj_id"] == proj_id and s["user_id"] == user_id
            and (round_num is None or s["round"] == round_num)
        ]
        if not matches:
            return "unknown"
        return sorted(matches, key=lambda s: s["round"])[-1]["state"]

    def reconnect(self, proj_id: str, user_id: str) -> dict:
        """
        Handle a client reconnecting after a missed round or disconnect.

        Determines the current project round and returns instructions:
        - If client missed a round: "catch_up" — download latest global model
        - If client is pending:    "submit"    — proceed with current round
        - If client already submitted: "wait"  — wait for next round

        Parameters
        ----------
        proj_id : project UUID
        user_id : hospital UUID

        Returns
        -------
        dict with keys:
            action       : "catch_up" | "submit" | "wait" | "unknown"
            current_round: int
            last_state   : str
        """
        from db_handler import get_project
        proj = get_project(proj_id)
        if proj is None:
            return {"action": "unknown", "current_round": 0, "last_state": "unknown"}

        current_round = proj.get("current_round", 0)
        last_state = self.get_state(proj_id, user_id)

        if last_state in ("missed", "unknown"):
            # Client missed a round — needs to download latest model
            self._upsert(proj_id, user_id, current_round, "reconnected")
            action = "catch_up"
        elif last_state == "pending":
            action = "submit"
        elif last_state in ("submitted", "reconnected"):
            action = "wait"
        else:
            action = "catch_up"

        logger.info(
            "Reconnection: proj=%s user=%s round=%d action=%s",
            proj_id, user_id, current_round, action,
        )
        return {
            "action":        action,
            "current_round": current_round,
            "last_state":    last_state,
        }

    def list_missed(self, proj_id: str) -> list[dict]:
        """Return all missed-state records for a project."""
        return [
            s for s in self._read_states()
            if s["proj_id"] == proj_id and s["state"] == "missed"
        ]


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
