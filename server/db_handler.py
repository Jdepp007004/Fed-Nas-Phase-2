from __future__ import annotations
"""
server/db_handler.py
M3: Thread-safe JSON flat-file database operations.
Owner: Sunishka Sarkar
"""

import json
import os
import threading

# ─── Database path ────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "database.json")

# Re-entrant lock — allows the same thread to acquire the lock multiple times
# (needed when write_db() calls read_db() internally via update_project())
_db_lock = threading.RLock()


# ─── Read / Write ─────────────────────────────────────────────────────────────

def read_db() -> dict:
    """
    Thread-safe read of the entire database.json.

    Returns
    -------
    dict — full DB state: {'users': [], 'projects': [], 'rounds_history': []}
    """
    with _db_lock:
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return an empty but valid DB structure
            return {"users": [], "projects": [], "rounds_history": []}


def write_db(data: dict) -> None:
    """
    Thread-safe atomic write of the full database state.

    Writes to a .tmp file first, then os.replace() to prevent corruption
    if the process is killed mid-write.

    Parameters
    ----------
    data : dict — complete new database state

    Raises
    ------
    ServerStorageError  if the write fails (disk full, permissions, etc.)
    """
    tmp_path = DB_PATH + ".tmp"
    with _db_lock:
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, DB_PATH)
        except OSError as exc:
            raise ServerStorageError(f"Failed to write database: {exc}") from exc


class ServerStorageError(Exception):
    """Raised when disk write fails."""


# ─── Project Helpers ──────────────────────────────────────────────────────────

def get_project(proj_id: str) -> dict | None:
    """
    Return the project record matching proj_id, or None if not found.

    Parameters
    ----------
    proj_id : str — project UUID

    Returns
    -------
    dict | None
    """
    db = read_db()
    for proj in db.get("projects", []):
        if proj.get("proj_id") == proj_id:
            return proj
    return None


def update_project(proj_id: str, updates: dict) -> None:
    """
    Shallow-merge `updates` into the project record and persist.

    Parameters
    ----------
    proj_id : str  — target project UUID
    updates : dict — key-value pairs to merge

    Raises
    ------
    KeyError if proj_id is not found
    """
    with _db_lock:
        db = read_db()
        for proj in db["projects"]:
            if proj.get("proj_id") == proj_id:
                proj.update(updates)
                write_db(db)
                return
        raise KeyError(f"Project {proj_id} not found in database.")


# ─── User Helpers ─────────────────────────────────────────────────────────────

def get_user(user_id: str = None, username: str = None) -> dict | None:
    """
    Lookup a user by user_id or username.  Returns None if not found.
    """
    db = read_db()
    for user in db.get("users", []):
        if user_id and user.get("user_id") == user_id:
            return user
        if username and user.get("username") == username:
            return user
    return None


def append_round_history(record: dict) -> None:
    """Append a round metric record to the rounds_history list."""
    with _db_lock:
        db = read_db()
        db.setdefault("rounds_history", []).append(record)
        write_db(db)
