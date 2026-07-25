"""
server/consent.py
Per-hospital, per-project consent management for the FL Platform (Phase 3 — C4).

A hospital must have an active consent record for a project before the server
will accept its model updates.  Consent records are stored in the database
(JSON flat-file or PostgreSQL via db_handler).

Consent Schema (stored in db["consents"])
-----------------------------------------
{
  "consent_id":   str (uuid4),
  "user_id":      str,
  "proj_id":      str,
  "scope":        list[str],   # allowed feature columns (empty = all columns)
  "granted_at":   str (ISO-8601 UTC),
  "revoked_at":   str | None,
  "active":       bool,
}

Public API
----------
    from consent import ConsentManager
    cm = ConsentManager()

    # Grant consent (called by hospital admin)
    cm.grant(user_id, proj_id, scope=[])

    # Check before accepting update
    if not cm.has_active_consent(user_id, proj_id):
        raise HTTPException(403, "Consent not granted")

    # Revoke (right-to-deletion trigger)
    cm.revoke(user_id, proj_id)
"""
from __future__ import annotations

import datetime
import logging
import threading
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


class ConsentError(Exception):
    """Raised when a consent operation fails."""


class ConsentManager:
    """
    In-process consent management backed by db_handler.

    Thread-safe: all DB reads/writes go through db_handler's own RLock.
    """

    # ── Read helpers ──────────────────────────────────────────────────────────

    def _read_consents(self) -> list[dict]:
        from db_handler import read_db
        db = read_db()
        return db.get("consents", [])

    def _write_consents(self, consents: list[dict]) -> None:
        from db_handler import read_db, write_db
        db = read_db()
        db["consents"] = consents
        write_db(db)

    # ── Public API ────────────────────────────────────────────────────────────

    def grant(
        self,
        user_id: str,
        proj_id: str,
        scope: Optional[list[str]] = None,
    ) -> dict:
        """
        Record that hospital `user_id` has granted consent to participate in
        project `proj_id` with the given data scope.

        If an active consent already exists for this (user_id, proj_id) pair,
        it is superseded by the new record.

        Returns the new consent record.
        """
        self.revoke(user_id, proj_id, _silent=True)  # revoke any prior consent first

        record = {
            "consent_id": str(uuid.uuid4()),
            "user_id":    user_id,
            "proj_id":    proj_id,
            "scope":      scope or [],
            "granted_at": _utcnow(),
            "revoked_at": None,
            "active":     True,
        }
        consents = self._read_consents()
        consents.append(record)
        self._write_consents(consents)
        logger.info(
            "Consent granted: user=%s proj=%s scope=%s",
            user_id, proj_id, scope or "all",
        )
        return record

    def revoke(
        self,
        user_id: str,
        proj_id: str,
        _silent: bool = False,
    ) -> int:
        """
        Revoke all active consents for (user_id, proj_id).
        Returns the number of records revoked.
        """
        consents = self._read_consents()
        revoked = 0
        now = _utcnow()
        for c in consents:
            if c["user_id"] == user_id and c["proj_id"] == proj_id and c["active"]:
                c["active"] = False
                c["revoked_at"] = now
                revoked += 1
        if revoked > 0 or not _silent:
            self._write_consents(consents)
            if revoked > 0:
                logger.info("Consent revoked: user=%s proj=%s (%d records)", user_id, proj_id, revoked)
        return revoked

    def has_active_consent(self, user_id: str, proj_id: str) -> bool:
        """
        Return True if there is at least one active (non-revoked) consent for
        the given (user_id, proj_id) pair.
        """
        for c in self._read_consents():
            if c["user_id"] == user_id and c["proj_id"] == proj_id and c["active"]:
                return True
        return False

    def get_consent(self, user_id: str, proj_id: str) -> Optional[dict]:
        """Return the most recent active consent record, or None."""
        matches = [
            c for c in self._read_consents()
            if c["user_id"] == user_id and c["proj_id"] == proj_id and c["active"]
        ]
        return matches[-1] if matches else None

    def list_consents(
        self,
        user_id: Optional[str] = None,
        proj_id: Optional[str] = None,
        active_only: bool = True,
    ) -> list[dict]:
        """Return consent records filtered by user_id and/or proj_id."""
        records = self._read_consents()
        if user_id:
            records = [c for c in records if c["user_id"] == user_id]
        if proj_id:
            records = [c for c in records if c["proj_id"] == proj_id]
        if active_only:
            records = [c for c in records if c["active"]]
        return records

    def revoke_all_for_user(self, user_id: str) -> int:
        """
        Revoke ALL consents for a user across all projects.
        Called during right-to-deletion / federated unlearning.
        """
        consents = self._read_consents()
        revoked = 0
        now = _utcnow()
        for c in consents:
            if c["user_id"] == user_id and c["active"]:
                c["active"] = False
                c["revoked_at"] = now
                revoked += 1
        self._write_consents(consents)
        logger.info("All consents revoked for user=%s (%d records)", user_id, revoked)
        return revoked


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
