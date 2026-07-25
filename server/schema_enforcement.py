"""
server/schema_enforcement.py
Required-column schema enforcement for uploaded client data (Phase 3 — C3).

The server defines a canonical schema of required feature columns. Before
accepting any model update from a client, the server checks that the client
has declared (and validated) all required columns in its submission metadata.

Clients are expected to pass their column list as part of the join request
(already stored in `hardware_profile["columns"]`) or as part of the update
metadata (`metrics["columns"]`). This module provides the server-side
validator that rejects missing or disallowed columns.

Public API
----------
    from schema_enforcement import SchemaEnforcer

    enforcer = SchemaEnforcer(required_columns=SERVER_SCHEMA)

    # Raises SchemaViolationError if columns are missing or disallowed
    enforcer.validate(client_columns)

    # Returns a clean set of allowed columns (intersection with server schema)
    allowed = enforcer.allowed_columns(client_columns)
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SchemaViolationError(Exception):
    """Raised when a client's declared columns do not satisfy the server schema."""


class SchemaEnforcer:
    """
    Server-side column schema enforcer.

    Parameters
    ----------
    required_columns : list[str]
        Columns that every participating client must have in their dataset.
        Updated via the project's `data_schema` field.
    strict : bool
        If True, clients may not submit extra columns beyond required_columns.
        If False (default), extra columns are silently ignored.
    """

    def __init__(
        self,
        required_columns: list[str],
        strict: bool = False,
    ) -> None:
        if not required_columns:
            raise ValueError("required_columns must be non-empty.")
        self.required = set(required_columns)
        self.strict = strict

    def validate(self, client_columns: list[str]) -> None:
        """
        Assert that `client_columns` satisfies the server schema.

        Parameters
        ----------
        client_columns : list[str] — columns the client reports having

        Raises
        ------
        SchemaViolationError — if any required column is missing,
                               or (in strict mode) if extra columns are present
        """
        client_set = set(client_columns)
        missing = self.required - client_set
        if missing:
            raise SchemaViolationError(
                f"Client is missing required columns: {sorted(missing)}. "
                f"Required: {sorted(self.required)}"
            )
        if self.strict:
            extra = client_set - self.required
            if extra:
                raise SchemaViolationError(
                    f"Client submitted disallowed columns (strict mode): {sorted(extra)}"
                )
        logger.debug(
            "Schema validation passed: client has %d/%d required columns.",
            len(client_set & self.required), len(self.required),
        )

    def allowed_columns(self, client_columns: list[str]) -> list[str]:
        """
        Return the intersection of client_columns and required_columns.

        Use to produce a sanitised column list for downstream processing.
        """
        return sorted(set(client_columns) & self.required)

    def is_valid(self, client_columns: list[str]) -> bool:
        """Return True if validate() would pass (no exception)."""
        try:
            self.validate(client_columns)
            return True
        except SchemaViolationError:
            return False


def build_enforcer_from_project(proj: dict, strict: bool = False) -> SchemaEnforcer:
    """
    Convenience: build a SchemaEnforcer from a project DB record.

    Falls back to SERVER_SCHEMA if the project has no data_schema.
    """
    from shared.model_schema import SERVER_SCHEMA
    schema = proj.get("data_schema") or SERVER_SCHEMA
    return SchemaEnforcer(required_columns=schema, strict=strict)
