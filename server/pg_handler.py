"""
server/pg_handler.py
PostgreSQL-backed storage handler — same public interface as db_handler.py.

Activated when DATABASE_URL is set in the environment.
Falls back gracefully: if SQLAlchemy/psycopg2 is not installed, raises
ImportError with a clear message rather than crashing at import time.

Usage
-----
    from pg_handler import PgHandler
    db = PgHandler()          # uses DATABASE_URL env var
    project = db.get_project("proj-uuid")
    db.update_project("proj-uuid", {"current_round": 1})
"""
from __future__ import annotations

import os
from typing import Optional

try:
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session
    _SA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SA_AVAILABLE = False

from models import Base
from models.user import User
from models.project import Project
from models.round_history import RoundHistory


class PgHandlerError(Exception):
    """Raised for PostgreSQL-layer errors."""


class PgHandler:
    """
    Drop-in alternative to db_handler.py backed by PostgreSQL via SQLAlchemy.

    All public methods mirror the function signatures of db_handler so that
    callers can switch implementations without code changes.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        if not _SA_AVAILABLE:
            raise ImportError(
                "sqlalchemy is not installed. "
                "Run: pip install sqlalchemy psycopg2-binary"
            )
        url = database_url or os.getenv("DATABASE_URL")
        if not url:
            raise ValueError(
                "DATABASE_URL environment variable is not set. "
                "Set it to a PostgreSQL connection string, e.g. "
                "postgresql+psycopg2://user:pass@localhost/fl_platform"
            )
        self._engine = create_engine(url, pool_pre_ping=True, future=True)
        # Create tables if they don't exist (Alembic handles migrations in prod)
        Base.metadata.create_all(self._engine)

    # ── Project ──────────────────────────────────────────────────────────────

    def get_project(self, proj_id: str) -> Optional[dict]:
        """Return project dict or None if not found."""
        with Session(self._engine) as s:
            row = s.get(Project, proj_id)
            return row.to_dict() if row else None

    def update_project(self, proj_id: str, updates: dict) -> None:
        """Shallow-merge updates into the project record. Raises KeyError if missing."""
        with Session(self._engine) as s:
            row = s.get(Project, proj_id)
            if row is None:
                raise KeyError(f"Project {proj_id!r} not found")
            for key, val in updates.items():
                if hasattr(row, key):
                    setattr(row, key, val)
            s.commit()

    def create_project(self, proj_dict: dict) -> None:
        """Insert a new project record."""
        with Session(self._engine) as s:
            s.add(Project.from_dict(proj_dict))
            s.commit()

    def list_projects(self) -> list[dict]:
        """Return all projects as list of dicts."""
        with Session(self._engine) as s:
            rows = s.scalars(select(Project)).all()
            return [r.to_dict() for r in rows]

    # ── User ─────────────────────────────────────────────────────────────────

    def get_user(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Optional[dict]:
        """Return user dict by user_id or username, or None."""
        with Session(self._engine) as s:
            if user_id:
                row = s.get(User, user_id)
            elif username:
                row = s.scalars(
                    select(User).where(User.username == username)
                ).first()
            else:
                return None
            return row.to_dict() if row else None

    def create_user(self, user_dict: dict) -> None:
        """Insert a new user record."""
        with Session(self._engine) as s:
            s.add(User.from_dict(user_dict))
            s.commit()

    def update_user(self, user_id: str, updates: dict) -> None:
        """Shallow-merge updates into the user record."""
        with Session(self._engine) as s:
            row = s.get(User, user_id)
            if row is None:
                raise KeyError(f"User {user_id!r} not found")
            for key, val in updates.items():
                col_map = {
                    "approved_projects": "approved_projects",
                    "pending_projects":  "pending_projects",
                    "last_active":       "last_active",
                    "role":              "role",
                }
                attr = col_map.get(key, key)
                if hasattr(row, attr):
                    setattr(row, attr, val)
            s.commit()

    def list_users(self) -> list[dict]:
        """Return all users as list of dicts."""
        with Session(self._engine) as s:
            rows = s.scalars(select(User)).all()
            return [r.to_dict() for r in rows]

    # ── Round history ─────────────────────────────────────────────────────────

    def append_round_history(self, record: dict) -> None:
        """Append a round-metric record."""
        with Session(self._engine) as s:
            s.add(RoundHistory.from_dict(record))
            s.commit()

    def get_round_history(self, proj_id: str) -> list[dict]:
        """Return all round records for proj_id, ordered by round number."""
        with Session(self._engine) as s:
            rows = s.scalars(
                select(RoundHistory)
                .where(RoundHistory.proj_id == proj_id)
                .order_by(RoundHistory.round)
            ).all()
            return [r.to_dict() for r in rows]

    # ── Full DB snapshot (for round_lifecycle compatibility) ──────────────────

    def read_db(self) -> dict:
        """
        Return a db_handler-compatible dict snapshot.
        Used by round_lifecycle() which receives a db_snapshot argument.
        """
        return {
            "users":         self.list_users(),
            "projects":      self.list_projects(),
            "rounds_history": [],  # Not included — use get_round_history()
        }
