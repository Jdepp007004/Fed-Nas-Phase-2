"""
server/mtls.py
Mutual TLS configuration helper for the FL Platform (Phase 3 — P6).

Provides a function to configure uvicorn/SSL context for mTLS so that:
  - The server presents its TLS certificate to clients
  - The server requires a valid client certificate signed by the same CA
  - Self-signed or institutional CA certificates are supported

This is loaded in main.py when MTLS_CA_CERT, MTLS_SERVER_CERT, and
MTLS_SERVER_KEY environment variables are all set.

Environment Variables
---------------------
MTLS_CA_CERT     : Path to CA certificate (PEM) that signed client certs
MTLS_SERVER_CERT : Path to server certificate (PEM)
MTLS_SERVER_KEY  : Path to server private key (PEM)

Usage (in main.py / uvicorn launch)
------------------------------------
    from mtls import get_ssl_context, is_mtls_configured

    if is_mtls_configured():
        ssl_ctx = get_ssl_context()
        uvicorn.run("main:app", ssl=ssl_ctx, ...)

Certificate Generation (development self-signed)
-------------------------------------------------
Run the companion script: scripts/gen_dev_certs.sh

    bash scripts/gen_dev_certs.sh

This creates certs/ca.pem, certs/server.pem, certs/server.key.
Set MTLS_CA_CERT, MTLS_SERVER_CERT, MTLS_SERVER_KEY to those paths.
"""
from __future__ import annotations

import logging
import os
import ssl
from typing import Optional

logger = logging.getLogger(__name__)


def is_mtls_configured() -> bool:
    """
    Return True if all three mTLS environment variables are set and the
    referenced files exist.
    """
    ca   = os.getenv("MTLS_CA_CERT", "")
    cert = os.getenv("MTLS_SERVER_CERT", "")
    key  = os.getenv("MTLS_SERVER_KEY", "")

    if not (ca and cert and key):
        return False

    for path, label in [(ca, "MTLS_CA_CERT"), (cert, "MTLS_SERVER_CERT"), (key, "MTLS_SERVER_KEY")]:
        if not os.path.isfile(path):
            logger.warning("mTLS: %s points to non-existent file %r — mTLS disabled", label, path)
            return False

    return True


def get_ssl_context() -> ssl.SSLContext:
    """
    Build and return an SSLContext configured for mutual TLS.

    Returns
    -------
    ssl.SSLContext configured with:
      - PROTOCOL_TLS_SERVER
      - Client authentication required (CERT_REQUIRED)
      - CA cert loaded for client cert verification

    Raises
    ------
    EnvironmentError  if required environment variables are not set
    ssl.SSLError      if the provided certificates cannot be loaded
    """
    ca   = os.environ.get("MTLS_CA_CERT", "")
    cert = os.environ.get("MTLS_SERVER_CERT", "")
    key  = os.environ.get("MTLS_SERVER_KEY", "")

    if not (ca and cert and key):
        raise EnvironmentError(
            "mTLS requires MTLS_CA_CERT, MTLS_SERVER_CERT, and MTLS_SERVER_KEY "
            "to all be set."
        )

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.verify_mode = ssl.CERT_REQUIRED         # require client certificate
    ctx.load_verify_locations(cafile=ca)        # trust client certs signed by this CA
    ctx.load_cert_chain(certfile=cert, keyfile=key)

    # Disable old TLS versions
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    logger.info(
        "mTLS configured: CA=%r CERT=%r KEY=%r",
        os.path.basename(ca),
        os.path.basename(cert),
        os.path.basename(key),
    )
    return ctx


def apply_to_uvicorn_config(config: dict, ssl_context: Optional[ssl.SSLContext] = None) -> dict:
    """
    Update a uvicorn.Config kwargs dict with mTLS SSL settings.

    Parameters
    ----------
    config      : dict — existing uvicorn kwargs
    ssl_context : optional pre-built SSLContext; if None, calls get_ssl_context()

    Returns
    -------
    Updated config dict with ssl_context key set
    """
    if ssl_context is None:
        ssl_context = get_ssl_context()
    config["ssl"] = ssl_context
    return config
