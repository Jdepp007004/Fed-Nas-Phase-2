"""
shared/key_manager.py
AES-256-GCM encryption key versioning and rotation for the FL Platform.

Supports:
  - Multiple named key versions stored in environment variables
  - Transparent key rotation: new keys encrypt, old keys still decrypt
  - Key fingerprinting for audit logs

Environment Variable Convention
--------------------------------
  FL_ENCRYPTION_KEY          Active key (version "v1")
  FL_ENCRYPTION_KEY_V2       Optional rotated key (version "v2")
  FL_ENCRYPTION_KEY_ACTIVE   Which version is used for NEW encryptions (default "v1")

When decrypting, the manager tries the version stored in the payload's
`key_version` field first, then falls back through all known versions.

Public API
----------
    from shared.key_manager import get_key_manager
    km = get_key_manager()

    # Encrypt (always uses the active key version)
    payload = km.encrypt(weights)      # dict with ciphertext, nonce, key_version

    # Decrypt (auto-selects key by payload["key_version"])
    weights = km.decrypt(payload)

    # Rotate active key version
    km.set_active_version("v2")
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Dict, Optional

from shared.encryption import encrypt_weights, decrypt_weights, generate_key_b64

logger = logging.getLogger(__name__)

# Maximum number of key versions supported
_MAX_VERSIONS = 10


class KeyVersionError(Exception):
    """Raised when a requested key version is not found."""


class KeyManager:
    """
    Manages multiple AES-256-GCM key versions for transparent key rotation.

    Parameters
    ----------
    keys         : dict[str, bytes] — mapping of version label → raw 32-byte key
    active_version : str — which version to use for new encryptions
    """

    def __init__(
        self,
        keys: Dict[str, bytes],
        active_version: str = "v1",
    ) -> None:
        if not keys:
            raise ValueError("At least one key version must be provided")
        if active_version not in keys:
            raise KeyVersionError(
                f"Active version {active_version!r} not found in provided keys. "
                f"Available: {list(keys.keys())}"
            )
        self._keys = {v: k for v, k in keys.items()}   # defensive copy
        self._active = active_version

    # ── Key introspection ────────────────────────────────────────────────────

    @property
    def active_version(self) -> str:
        """Currently active key version label."""
        return self._active

    @property
    def known_versions(self) -> list[str]:
        """All registered key version labels."""
        return list(self._keys.keys())

    def fingerprint(self, version: Optional[str] = None) -> str:
        """
        Return a short hex fingerprint (SHA-256[:8]) of the specified key version.
        Useful for audit logs — does NOT expose the key material.
        """
        v = version or self._active
        if v not in self._keys:
            raise KeyVersionError(f"Key version {v!r} not found")
        digest = hashlib.sha256(self._keys[v]).hexdigest()
        return digest[:16]   # 8 bytes = 64 bits — enough for audit, not reversible

    # ── Key rotation ─────────────────────────────────────────────────────────

    def set_active_version(self, version: str) -> None:
        """Promote `version` to active (used for new encryptions). Old version still decrypts."""
        if version not in self._keys:
            raise KeyVersionError(
                f"Cannot activate version {version!r} — not registered. "
                f"Call register_version() first."
            )
        logger.info(
            "KeyManager: rotating active key %r → %r (fingerprint: %s)",
            self._active, version, self.fingerprint(version),
        )
        self._active = version

    def register_version(self, version: str, key_b64: str) -> None:
        """
        Register a new key version.

        Parameters
        ----------
        version : str — version label (e.g. "v2")
        key_b64 : str — base64-encoded 32-byte AES key
        """
        raw = base64.b64decode(key_b64)
        if len(raw) != 32:
            raise ValueError(f"Key for version {version!r} must be 32 bytes, got {len(raw)}")
        self._keys[version] = raw
        logger.info("KeyManager: registered key version %r (fingerprint: %s)",
                    version, self.fingerprint(version))

    # ── Encrypt / Decrypt ────────────────────────────────────────────────────

    def encrypt(self, weights: dict) -> dict:
        """
        Encrypt a weight dict using the active key version.
        Adds `key_version` field to the payload for transparent decryption.
        """
        key_b64 = base64.b64encode(self._keys[self._active]).decode()
        payload = encrypt_weights(weights, key_b64=key_b64)
        payload["key_version"] = self._active
        return payload

    def decrypt(self, payload: dict) -> dict:
        """
        Decrypt a weight payload.
        Uses the `key_version` field to select the correct key.
        Falls back through all known versions if `key_version` is absent.
        """
        version = payload.get("key_version")

        if version:
            if version not in self._keys:
                raise KeyVersionError(
                    f"Payload has key_version={version!r} but that version is not registered. "
                    f"Known versions: {list(self._keys.keys())}"
                )
            key_b64 = base64.b64encode(self._keys[version]).decode()
            return decrypt_weights(payload, key_b64=key_b64)

        # Fallback: try all versions in order
        last_exc: Optional[Exception] = None
        for v, raw_key in self._keys.items():
            try:
                key_b64 = base64.b64encode(raw_key).decode()
                result = decrypt_weights(payload, key_b64=key_b64)
                logger.debug("KeyManager: decrypted using fallback version %r", v)
                return result
            except Exception as e:
                last_exc = e
        raise ValueError(
            f"Failed to decrypt payload with any known key version. "
            f"Last error: {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# Factory — singleton backed by environment variables
# ---------------------------------------------------------------------------

_default_manager: Optional[KeyManager] = None
_manager_lock_import = None


def get_key_manager() -> KeyManager:
    """
    Return the module-level KeyManager singleton.

    Reads key versions from environment variables:
      FL_ENCRYPTION_KEY      → "v1" (required)
      FL_ENCRYPTION_KEY_V2   → "v2" (optional)
      ...
      FL_ENCRYPTION_KEY_ACTIVE → which version is active (default: "v1")
    """
    global _default_manager
    if _default_manager is not None:
        return _default_manager

    keys: Dict[str, bytes] = {}

    # Always load v1 (required)
    v1_b64 = os.getenv("FL_ENCRYPTION_KEY", "")
    if not v1_b64:
        raise EnvironmentError(
            "FL_ENCRYPTION_KEY environment variable is not set. "
            "Generate with: python -c \"from shared.encryption import generate_key_b64; print(generate_key_b64())\""
        )
    keys["v1"] = base64.b64decode(v1_b64)

    # Load additional versioned keys (v2, v3, …)
    for i in range(2, _MAX_VERSIONS + 1):
        env_name = f"FL_ENCRYPTION_KEY_V{i}"
        b64_val = os.getenv(env_name, "")
        if b64_val:
            keys[f"v{i}"] = base64.b64decode(b64_val)

    active = os.getenv("FL_ENCRYPTION_KEY_ACTIVE", "v1")
    if active not in keys:
        logger.warning(
            "FL_ENCRYPTION_KEY_ACTIVE=%r is not a registered version — falling back to v1",
            active,
        )
        active = "v1"

    _default_manager = KeyManager(keys, active_version=active)
    logger.info(
        "KeyManager initialised: active=%r, known=%r, fingerprint=%s",
        active, list(keys.keys()), _default_manager.fingerprint(),
    )
    return _default_manager


def reset_key_manager(manager: Optional[KeyManager] = None) -> None:
    """Replace the singleton (used in tests to inject a test KeyManager)."""
    global _default_manager
    _default_manager = manager
