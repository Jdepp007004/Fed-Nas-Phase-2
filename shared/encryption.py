"""
shared/encryption.py
Utility: AES-256-GCM weight encryption/decryption.
Used by client (api_client.py) to encrypt weights before sending,
and by server (project_router.py) to decrypt received weights.
"""

import json
import os
import base64
import numpy as np

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ─── Key Management ──────────────────────────────────────────────────────────

def _get_key() -> bytes:
    """
    Load or derive the AES-256 key (32 bytes) from the FL_ENCRYPTION_KEY
    environment variable.  The env var must be a base64-encoded 32-byte key.
    Falls back to an insecure all-zeros key (only suitable for development).
    """
    raw = os.environ.get("FL_ENCRYPTION_KEY", "")
    if raw:
        key = base64.b64decode(raw)
        if len(key) != 32:
            raise ValueError(
                "FL_ENCRYPTION_KEY must decode to exactly 32 bytes (AES-256)."
            )
        return key
    # Development fallback — logs a warning
    import warnings
    warnings.warn(
        "FL_ENCRYPTION_KEY not set – using insecure all-zeros key. "
        "Set this env variable in production.",
        RuntimeWarning,
        stacklevel=2,
    )
    return b"\x00" * 32


# ─── Serialisation Helpers ───────────────────────────────────────────────────

def _weights_to_bytes(weights: dict) -> bytes:
    """Serialise a dict of {str: np.ndarray} to JSON bytes (via lists)."""
    serialisable = {k: v.tolist() for k, v in weights.items()}
    return json.dumps(serialisable).encode("utf-8")


def _bytes_to_weights(data: bytes) -> dict:
    """Deserialise JSON bytes back to {str: np.ndarray}."""
    loaded = json.loads(data.decode("utf-8"))
    return {k: np.array(v, dtype=np.float32) for k, v in loaded.items()}


# ─── Public API ──────────────────────────────────────────────────────────────

def encrypt_weights(weights: dict) -> dict:
    """
    Encrypt a weight dict using AES-256-GCM.

    Parameters
    ----------
    weights : dict
        {param_name: np.ndarray} as produced by get_subnet_weights().

    Returns
    -------
    dict
        {
          'ciphertext': str  (base64),
          'nonce':      str  (base64, 12 bytes),
          'tag':        str  (included in GCM ciphertext automatically),
        }
    """
    key = _get_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)           # 96-bit nonce recommended for GCM
    plaintext = _weights_to_bytes(weights)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)  # no additional data
    return {
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
    }


def decrypt_weights(encrypted: dict) -> dict:
    """
    Decrypt a weight payload produced by encrypt_weights().

    Parameters
    ----------
    encrypted : dict
        Must have 'ciphertext' and 'nonce' keys (base64 strings).

    Returns
    -------
    dict
        {param_name: np.ndarray}

    Raises
    ------
    ValueError
        If decryption fails (wrong key, tampered ciphertext).
    """
    key = _get_key()
    aesgcm = AESGCM(key)
    try:
        nonce = base64.b64decode(encrypted["nonce"])
        ciphertext = base64.b64decode(encrypted["ciphertext"])
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise ValueError(f"Weight decryption failed: {exc}") from exc
    return _bytes_to_weights(plaintext)


def generate_key_b64() -> str:
    """
    Utility: generate a fresh random AES-256 key and return its base64
    encoding.  Call this once at server setup and set FL_ENCRYPTION_KEY.
    """
    return base64.b64encode(os.urandom(32)).decode("ascii")
