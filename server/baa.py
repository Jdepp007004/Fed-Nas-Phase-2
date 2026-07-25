"""
server/baa.py
Business Associate Agreement (BAA) enforcement gate (Phase 3 — C8).

Under HIPAA, any entity that processes Protected Health Information (PHI)
on behalf of a covered entity must sign a BAA.  In this FL platform, every
hospital (client) that sends model updates is a Business Associate.

Enforcement model
-----------------
  1. The user record in the DB has a `baa_signed` boolean field.
  2. Before a client is allowed to submit a model update, the server checks
     `baa_signed == True` for that user.
  3. Hospitals sign the BAA via the admin dashboard or API endpoint.
     The signing event is recorded with a timestamp and admin signature.

This module provides:
  - `baa_required()` — FastAPI dependency that raises 403 if BAA not signed
  - `BAAManager`     — record + query BAA signatures in the DB

DB Schema (stored in db["baa_records"])
----------------------------------------
  [{"user_id": str, "proj_id": str | None, "signed_at": str (UTC),
    "signed_by": str, "document_hash": str}]

The user record gets `baa_signed: True` set when the first BAA record
is added.  It is set to False on BAA revocation (hospital exit / offboarding).
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# Standard BAA template hash (sha256 of the text to track version)
_DEFAULT_BAA_HASH = hashlib.sha256(b"FL-Platform-BAA-v1.0").hexdigest()


class BAAViolationError(Exception):
    """Raised when a hospital attempts an action without a signed BAA."""


class BAAManager:
    """
    Records and enforces BAA signature state for each hospital user.

    Backed by db_handler (flat-file or PostgreSQL, same interface).
    """

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _read_records(self) -> list[dict]:
        from db_handler import read_db
        return read_db().get("baa_records", [])

    def _write_records(self, records: list[dict]) -> None:
        from db_handler import read_db, write_db
        db = read_db()
        db["baa_records"] = records
        write_db(db)

    # ── Public API ────────────────────────────────────────────────────────────

    def sign(
        self,
        user_id: str,
        signed_by: str,
        proj_id: Optional[str] = None,
        document_hash: str = _DEFAULT_BAA_HASH,
    ) -> dict:
        """
        Record that a hospital administrator (`signed_by`) has signed the BAA
        for hospital `user_id`.

        Also sets `baa_signed = True` on the user record.

        Returns the BAA record dict.
        """
        record = {
            "baa_id":       str(uuid.uuid4()),
            "user_id":      user_id,
            "proj_id":      proj_id,
            "signed_at":    _utcnow(),
            "signed_by":    signed_by,
            "document_hash": document_hash,
            "active":       True,
        }
        records = self._read_records()
        records.append(record)
        self._write_records(records)
        self._set_user_baa_flag(user_id, True)
        logger.info("BAA signed: user=%s by=%s proj=%s", user_id, signed_by, proj_id)
        return record

    def has_signed(self, user_id: str) -> bool:
        """Return True if the user has at least one active BAA record."""
        for r in self._read_records():
            if r["user_id"] == user_id and r.get("active", True):
                return True
        return False

    def revoke(self, user_id: str) -> int:
        """Revoke all BAA records for a user. Returns number revoked."""
        records = self._read_records()
        revoked = 0
        for r in records:
            if r["user_id"] == user_id and r.get("active", True):
                r["active"] = False
                revoked += 1
        if revoked:
            self._write_records(records)
            self._set_user_baa_flag(user_id, False)
            logger.info("BAA revoked: user=%s (%d records)", user_id, revoked)
        return revoked

    def get_record(self, user_id: str) -> Optional[dict]:
        """Return the most recent active BAA record for a user."""
        matches = [
            r for r in self._read_records()
            if r["user_id"] == user_id and r.get("active", True)
        ]
        return matches[-1] if matches else None

    def check_or_raise(self, user_id: str) -> None:
        """
        Raise BAAViolationError if the user does not have a signed BAA.
        Call this before accepting model updates.
        """
        if not self.has_signed(user_id):
            raise BAAViolationError(
                f"Hospital user_id={user_id} has not signed the Business Associate Agreement. "
                "Please complete the BAA before submitting model updates."
            )

    def list_records(self, active_only: bool = True) -> list[dict]:
        records = self._read_records()
        if active_only:
            records = [r for r in records if r.get("active", True)]
        return records

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_user_baa_flag(self, user_id: str, signed: bool) -> None:
        """Update baa_signed field on the user record in the DB."""
        from db_handler import read_db, write_db
        db = read_db()
        for user in db.get("users", []):
            if user.get("user_id") == user_id:
                user["baa_signed"] = signed
        write_db(db)


# ── FastAPI dependency ─────────────────────────────────────────────────────────

def baa_required(current_user: dict) -> None:
    """
    FastAPI dependency: raise HTTP 403 if the requesting hospital has not
    signed the BAA.  Inject into endpoint with:

        from fastapi import Depends
        from baa import baa_required

        @router.post("/update")
        async def post_update(
            ...,
            _baa: None = Depends(lambda cu=Depends(_get_current_user): baa_required(cu)),
        ): ...
    """
    from fastapi import HTTPException
    user_id = current_user.get("sub", "")
    mgr = BAAManager()
    if not mgr.has_signed(user_id):
        raise HTTPException(
            status_code=403,
            detail="Business Associate Agreement not signed. "
                   "Contact your administrator to complete onboarding.",
        )


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
