"""
server/storage.py
Pluggable storage backend for model checkpoints (.pt files).

Backends
--------
LOCAL   Active when S3_BUCKET is not set (default — existing behaviour).
S3      Active when S3_BUCKET + AWS credentials are configured.

Both backends expose the same interface:

    store = get_storage()
    path = store.save(proj_id, round_num, model_state_dict)  # returns URI
    data = store.load(path)                                    # returns state dict
    store.delete(path)

Path / URI format
-----------------
Local : absolute filesystem path, e.g.
        /app/server/models/proj-abc_round3.pt
S3    : s3 URI, e.g.
        s3://my-bucket/fl-models/proj-abc_round3.pt
"""
from __future__ import annotations

import io
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ─── Models directory (same default as project_router.py) ────────────────────
_DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# ─── Abstract interface ───────────────────────────────────────────────────────

class StorageBackend(ABC):
    """Abstract base for all storage backends."""

    @abstractmethod
    def save(self, proj_id: str, round_num: int, state_dict: dict) -> str:
        """
        Serialise and persist state_dict.
        Returns the URI / path string to be stored in the DB as global_model_path.
        """

    @abstractmethod
    def load(self, uri: str) -> dict:
        """Load and deserialise a state_dict from the given URI."""

    @abstractmethod
    def delete(self, uri: str) -> None:
        """Delete the checkpoint at uri (best-effort; does not raise on missing)."""

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Return True if the checkpoint exists."""


# ─── Local filesystem ─────────────────────────────────────────────────────────

class LocalStorage(StorageBackend):
    """
    Stores model checkpoints as .pt files in a local directory.
    This is the default and mirrors the existing project_router.py behaviour.
    """

    def __init__(self, models_dir: Optional[str] = None) -> None:
        self._dir = models_dir or os.getenv("MODELS_DIR", _DEFAULT_MODELS_DIR)
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, proj_id: str, round_num: int) -> str:
        return os.path.join(self._dir, f"{proj_id}_round{round_num}.pt")

    def save(self, proj_id: str, round_num: int, state_dict: dict) -> str:
        path = self._path(proj_id, round_num)
        torch.save(state_dict, path)
        logger.info("Saved model checkpoint: %s", path)
        return path

    def load(self, uri: str) -> dict:
        if not os.path.exists(uri):
            raise FileNotFoundError(f"Model checkpoint not found: {uri}")
        return torch.load(uri, map_location="cpu", weights_only=True)

    def delete(self, uri: str) -> None:
        try:
            os.remove(uri)
            logger.info("Deleted model checkpoint: %s", uri)
        except FileNotFoundError:
            pass

    def exists(self, uri: str) -> bool:
        return os.path.isfile(uri)


# ─── AWS S3 ───────────────────────────────────────────────────────────────────

class S3Storage(StorageBackend):
    """
    Stores model checkpoints in an S3 bucket.

    Requires:
        pip install boto3
        Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or IAM role),
                               S3_BUCKET, S3_PREFIX (optional, default "fl-models/")
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> None:
        try:
            import boto3
            self._s3 = boto3.client("s3")
        except ImportError as exc:
            raise ImportError(
                "boto3 is not installed. Run: pip install boto3"
            ) from exc

        self._bucket = bucket or os.getenv("S3_BUCKET", "")
        if not self._bucket:
            raise ValueError("S3_BUCKET environment variable is not set.")
        self._prefix = prefix or os.getenv("S3_PREFIX", "fl-models/")

    def _key(self, proj_id: str, round_num: int) -> str:
        return f"{self._prefix}{proj_id}_round{round_num}.pt"

    def _uri(self, key: str) -> str:
        return f"s3://{self._bucket}/{key}"

    def _parse_uri(self, uri: str) -> str:
        """Strip s3://bucket/ prefix to get the S3 key."""
        prefix = f"s3://{self._bucket}/"
        if uri.startswith(prefix):
            return uri[len(prefix):]
        return uri

    def save(self, proj_id: str, round_num: int, state_dict: dict) -> str:
        key = self._key(proj_id, round_num)
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        buf.seek(0)
        self._s3.upload_fileobj(buf, self._bucket, key)
        uri = self._uri(key)
        logger.info("Saved model checkpoint to S3: %s", uri)
        return uri

    def load(self, uri: str) -> dict:
        key = self._parse_uri(uri)
        buf = io.BytesIO()
        self._s3.download_fileobj(self._bucket, key, buf)
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=True)

    def delete(self, uri: str) -> None:
        try:
            key = self._parse_uri(uri)
            self._s3.delete_object(Bucket=self._bucket, Key=key)
            logger.info("Deleted S3 model checkpoint: %s", uri)
        except Exception:
            pass

    def exists(self, uri: str) -> bool:
        try:
            key = self._parse_uri(uri)
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False


# ─── Factory ─────────────────────────────────────────────────────────────────

_default_backend: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    """
    Return the appropriate storage backend based on environment variables.

    If S3_BUCKET is set → S3Storage.
    Otherwise            → LocalStorage (default, existing behaviour).
    """
    global _default_backend
    if _default_backend is None:
        if os.getenv("S3_BUCKET"):
            _default_backend = S3Storage()
        else:
            _default_backend = LocalStorage()
    return _default_backend


def reset_storage(backend: Optional[StorageBackend] = None) -> None:
    """Replace the singleton (used in tests to inject a LocalStorage with tmp_path)."""
    global _default_backend
    _default_backend = backend
