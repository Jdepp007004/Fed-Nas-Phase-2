"""
server/unlearning.py
Federated Unlearning — Right to Deletion (Phase 3 — C5).

When a hospital exercises its right to deletion, all model checkpoints that
incorporated that hospital's data must be rolled back to the last clean state
(the global model checkpoint from the round *before* the hospital joined).

Algorithm
---------
1. Identify the first round in which the hospital (`user_id`) submitted an
   update (from `rounds_history`).
2. Find the global model checkpoint from the round *before* that round
   (the "last clean checkpoint").
3. Restore the project's `global_model_path` to that checkpoint.
4. Reset `current_round` to the clean round number.
5. Revoke all consents for the hospital.
6. Remove the hospital from `connected_clients` and `approved_projects`.
7. Append an audit record to `rounds_history` documenting the unlearning.

Note: This implementation resets rounds and restores the checkpoint file path.
Full re-training from the clean checkpoint is left to the operators; this
module handles the database and filesystem accounting.

Public API
----------
    from unlearning import UnlearningManager
    um = UnlearningManager()
    result = um.forget_hospital(proj_id, user_id)
    # result = {"status": "ok", "clean_round": 3, "reverted_rounds": 2}
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class UnlearningError(Exception):
    """Raised when federated unlearning cannot proceed."""


class UnlearningManager:
    """
    Implements the right-to-deletion via checkpoint rollback.

    All DB operations go through db_handler (JSON or PG, same interface).
    """

    def forget_hospital(
        self,
        proj_id: str,
        user_id: str,
        models_dir: Optional[str] = None,
    ) -> dict:
        """
        Erase a hospital's contribution from the global model.

        Parameters
        ----------
        proj_id   : str — project UUID
        user_id   : str — hospital (user) UUID to forget
        models_dir: str — override path to model checkpoints

        Returns
        -------
        dict with keys:
            status          : "ok" | "no_contribution"
            clean_round     : int — round restored to (0 = initial)
            reverted_rounds : int — number of rounds rolled back
            checkpoint_path : str — path of restored global model
        """
        from db_handler import read_db, write_db, update_project

        db = read_db()
        proj = next((p for p in db.get("projects", []) if p["proj_id"] == proj_id), None)
        if proj is None:
            raise UnlearningError(f"Project {proj_id!r} not found")

        # ── Step 1: identify first round this hospital contributed ────────────
        history = sorted(
            [r for r in db.get("rounds_history", [])
             if r.get("proj_id") == proj_id],
            key=lambda r: r.get("round", 0),
        )
        hospital_rounds = [
            r for r in history
            if user_id in r.get("contributing_clients", [])
        ]

        if not hospital_rounds:
            logger.info(
                "UnlearningManager: user=%s has no recorded contributions to proj=%s",
                user_id, proj_id,
            )
            # Still revoke consent and remove from project
            self._remove_hospital(db, proj_id, user_id)
            write_db(db)
            return {
                "status": "no_contribution",
                "clean_round": proj.get("current_round", 0),
                "reverted_rounds": 0,
                "checkpoint_path": proj.get("global_model_path", ""),
            }

        first_contaminated_round = min(r["round"] for r in hospital_rounds)
        clean_round = max(0, first_contaminated_round - 1)
        reverted_rounds = proj.get("current_round", 0) - clean_round

        # ── Step 2: find clean checkpoint path ───────────────────────────────
        clean_checkpoint = self._find_checkpoint(
            proj_id, clean_round, models_dir or _default_models_dir(),
        )

        # ── Step 3: update project record ────────────────────────────────────
        update_project(proj_id, {
            "current_round":    clean_round,
            "global_model_path": clean_checkpoint,
            "round_state":      "idle",
        })

        # ── Step 4: revoke consent + remove hospital from project ─────────────
        db = read_db()  # re-read after update_project
        self._remove_hospital(db, proj_id, user_id)

        # ── Step 5: audit record ─────────────────────────────────────────────
        audit = {
            "proj_id":              proj_id,
            "round":                clean_round,
            "event":                "unlearning",
            "forgotten_user_id":    user_id,
            "reverted_rounds":      reverted_rounds,
            "restored_checkpoint":  clean_checkpoint,
            "timestamp":            _utcnow(),
            "contributing_clients": [],
        }
        db.setdefault("rounds_history", []).append(audit)
        write_db(db)

        logger.info(
            "Federated unlearning complete: proj=%s user=%s clean_round=%d reverted=%d",
            proj_id, user_id, clean_round, reverted_rounds,
        )
        return {
            "status":           "ok",
            "clean_round":      clean_round,
            "reverted_rounds":  reverted_rounds,
            "checkpoint_path":  clean_checkpoint,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_checkpoint(
        self, proj_id: str, round_num: int, models_dir: str,
    ) -> str:
        """Return path to checkpoint for given round, or empty string."""
        if round_num <= 0:
            return ""
        candidate = os.path.join(models_dir, f"{proj_id}_round{round_num}.pt")
        if os.path.isfile(candidate):
            return candidate
        # Walk backwards to find the nearest available checkpoint
        for r in range(round_num - 1, 0, -1):
            fallback = os.path.join(models_dir, f"{proj_id}_round{r}.pt")
            if os.path.isfile(fallback):
                logger.warning(
                    "Checkpoint for round %d not found; using round %d", round_num, r,
                )
                return fallback
        return ""

    def _remove_hospital(self, db: dict, proj_id: str, user_id: str) -> None:
        """
        Remove user from project's connected_clients list.
        Revoke all consents (best-effort — consent module may not be active).
        Remove proj_id from user's approved_projects.
        """
        for proj in db.get("projects", []):
            if proj["proj_id"] == proj_id:
                proj["connected_clients"] = [
                    c for c in proj.get("connected_clients", []) if c != user_id
                ]
                break

        for user in db.get("users", []):
            if user["user_id"] == user_id:
                user["approved_projects"] = [
                    p for p in user.get("approved_projects", []) if p != proj_id
                ]
                break

        # Revoke consents (best-effort)
        try:
            from consent import ConsentManager
            ConsentManager().revoke_all_for_user(user_id)
        except Exception as e:
            logger.warning("Could not revoke consents for %s: %s", user_id, e)


def _default_models_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "models")


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
