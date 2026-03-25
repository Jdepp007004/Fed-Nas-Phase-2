"""
server/ngrok_tunnel.py
M3: ngrok tunnel initialization and URL management.
Owner: Sunishka Sarkar
"""
from __future__ import annotations

import atexit
import os
import time

# Module-level singleton for the active public URL
_tunnel_url: str | None = None
_tunnel = None


def start_ngrok_tunnel(local_port: int, auth_token: str) -> str:
    """
    Initialize pyngrok, open an HTTPS tunnel to `local_port`, and return the
    public URL.

    Parameters
    ----------
    local_port : int — localhost port FastAPI is bound to (typically 8000)
    auth_token : str — ngrok auth token from env NGROK_AUTH_TOKEN

    Returns
    -------
    str — public HTTPS URL, e.g. 'https://abc123.ngrok-free.app'
    """
    global _tunnel_url, _tunnel

    try:
        from pyngrok import ngrok, conf  # correct import for pyngrok package
    except ImportError:
        raise ImportError(
            "pyngrok is not installed. Run: pip install pyngrok"
        )

    # Set auth token via pyngrok config
    conf.get_default().auth_token = auth_token

    # Open tunnel — pyngrok returns an NgrokTunnel object
    _tunnel = ngrok.connect(local_port, "http")

    # pyngrok v5+ provides public_url directly on the tunnel object
    url = getattr(_tunnel, "public_url", None)

    # Fallback: poll a moment in case of async init
    if not url:
        start = time.time()
        while time.time() - start < 15:
            url = getattr(_tunnel, "public_url", None)
            if url:
                break
            time.sleep(0.5)

    if not url:
        raise RuntimeError("ngrok tunnel did not provide a public URL within timeout.")

    # Normalise http:// → https://
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    _tunnel_url = url

    # Graceful cleanup on process exit
    atexit.register(_close_tunnel)

    return _tunnel_url


def get_tunnel_url() -> str:
    """
    Return the currently active ngrok public URL.

    Raises
    ------
    RuntimeError  if the tunnel has not been initialized yet.
    """
    if _tunnel_url is None:
        raise RuntimeError(
            "ngrok tunnel has not been initialized. "
            "Call start_ngrok_tunnel() first."
        )
    return _tunnel_url


def _close_tunnel():
    """atexit handler: gracefully disconnect the ngrok tunnel."""
    global _tunnel, _tunnel_url
    if _tunnel is not None:
        try:
            from pyngrok import ngrok
            public_url = getattr(_tunnel, "public_url", "")
            if public_url:
                ngrok.disconnect(public_url)
        except Exception:
            pass
    _tunnel = None
    _tunnel_url = None
