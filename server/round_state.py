"""
server/round_state.py
Explicit round state machine for the federated learning lifecycle.

States
------
IDLE        → No round in progress. Accepting client updates.
COLLECTING  → Waiting for enough client updates to trigger aggregation.
AGGREGATING → FedAvg + momentum + NAS running (BackgroundTask or Celery worker).
DONE        → Round complete; global model saved; metrics recorded.
ERROR       → Aggregation failed; requires operator intervention.

Transitions
-----------
IDLE        → COLLECTING  (first update received for the round)
COLLECTING  → AGGREGATING (min_clients_per_round threshold crossed)
AGGREGATING → DONE        (round_lifecycle completed successfully)
AGGREGATING → ERROR       (unhandled exception in round_lifecycle)
DONE        → IDLE        (server increments round counter, resets state)
ERROR       → IDLE        (operator manually resets)

Usage
-----
    from round_state import RoundState, RoundStateMachine

    machine = RoundStateMachine(proj_id="proj-uuid")
    machine.transition(RoundState.COLLECTING)
    assert machine.state == RoundState.COLLECTING
"""
from __future__ import annotations

import enum
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RoundState(str, enum.Enum):
    """Enumeration of all valid round states."""
    IDLE        = "idle"
    COLLECTING  = "collecting"
    AGGREGATING = "aggregating"
    DONE        = "done"
    ERROR       = "error"


# Valid state transitions: {from_state: {allowed to_states}}
_TRANSITIONS: dict[RoundState, set[RoundState]] = {
    RoundState.IDLE:        {RoundState.COLLECTING},
    RoundState.COLLECTING:  {RoundState.AGGREGATING, RoundState.IDLE},
    RoundState.AGGREGATING: {RoundState.DONE, RoundState.ERROR},
    RoundState.DONE:        {RoundState.IDLE},
    RoundState.ERROR:       {RoundState.IDLE},
}


class InvalidTransitionError(Exception):
    """Raised when a state transition is not in the allowed set."""


class RoundStateMachine:
    """
    In-process state machine for a single project's round lifecycle.

    This class is intentionally stateless with respect to persistence —
    callers are responsible for reading the `round_state` field from the
    DB and constructing a new machine per request.  This avoids any
    consistency issues between processes or Celery workers.

    Example
    -------
        state = RoundStateMachine(proj_id, current_state="collecting")
        state.transition(RoundState.AGGREGATING)
        # persist state.state to DB here
    """

    def __init__(
        self,
        proj_id: str,
        current_state: str | RoundState = RoundState.IDLE,
    ) -> None:
        self.proj_id = proj_id
        self._state = RoundState(current_state)

    @property
    def state(self) -> RoundState:
        return self._state

    def can_transition(self, to: RoundState) -> bool:
        """Return True if the requested transition is legal from current state."""
        return to in _TRANSITIONS.get(self._state, set())

    def transition(self, to: RoundState | str) -> RoundState:
        """
        Perform a state transition.

        Parameters
        ----------
        to : RoundState or str — target state

        Returns
        -------
        The new state.

        Raises
        ------
        InvalidTransitionError if the transition is not allowed.
        """
        target = RoundState(to)
        if not self.can_transition(target):
            raise InvalidTransitionError(
                f"Project {self.proj_id!r}: cannot transition from "
                f"{self._state.value!r} → {target.value!r}. "
                f"Allowed: {[s.value for s in _TRANSITIONS.get(self._state, set())]}"
            )
        logger.info(
            "Round state transition [%s]: %s → %s",
            self.proj_id, self._state.value, target.value,
        )
        self._state = target
        return self._state

    def reset_to_idle(self) -> None:
        """Force-reset to IDLE regardless of current state (operator action)."""
        logger.warning(
            "Force-resetting round state for %s from %s to idle",
            self.proj_id, self._state.value,
        )
        self._state = RoundState.IDLE

    def __repr__(self) -> str:
        return f"RoundStateMachine(proj_id={self.proj_id!r}, state={self._state.value!r})"
