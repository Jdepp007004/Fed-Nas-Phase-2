"""
client/api_client.py
M4/M3: HTTP wrapper for all server REST API calls.
Owner: Nikhil Garuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests  # noqa: E402
from requests.adapters import HTTPAdapter  # noqa: E402
from urllib3.util.retry import Retry  # noqa: E402

from shared.encryption import encrypt_weights  # noqa: E402


# ─── Custom Exceptions ────────────────────────────────────────────────────────

class ServerUnreachableError(Exception):
    """Raised when the server is not reachable after all retries."""


class AuthError(Exception):
    """Raised on 401/403 HTTP responses."""


class SchemaError(Exception):
    """Raised on 422 Unprocessable Entity (schema mismatch)."""


class RoundConflictError(Exception):
    """Raised on 409 Conflict (wrong round_id submitted)."""


# ─── APIClient ───────────────────────────────────────────────────────────────

class APIClient:
    """
    Thin wrapper around `requests` that encapsulates all HTTP communication
    with the federated learning server.

    Parameters
    ----------
    server_url : str  — The ngrok HTTPS URL entered by the client operator
    token      : str  — JWT Bearer token (populated after login(), None before)
    """

    TIMEOUT = 30  # seconds per request

    def __init__(self, server_url: str, token: str = None):
        self.server_url = server_url.rstrip("/")
        self.token = token

        # Shared session with retry logic (3 retries, exponential backoff)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.5,      # waits 1.5s, 3s, 4.5s between retries
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = self._url(path)
        try:
            resp = self.session.request(
                method,
                url,
                headers=self._headers(),
                timeout=self.TIMEOUT,
                **kwargs,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ServerUnreachableError(f"Cannot reach server at {url}: {exc}") from exc
        except requests.exceptions.Timeout as exc:
            raise ServerUnreachableError(f"Request to {url} timed out") from exc

        if resp.status_code in (401, 403):
            raise AuthError(f"Authentication error [{resp.status_code}]: {resp.text}")
        if resp.status_code == 409:
            raise RoundConflictError(f"Round conflict: {resp.text}")
        if resp.status_code == 422:
            raise SchemaError(f"Schema mismatch: {resp.text}")
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text}")

        return resp.json()

    # ── Auth endpoints ────────────────────────────────────────────────────────

    def check_status(self) -> dict:
        """GET /api/status — verify connectivity and retrieve ngrok URL."""
        return self._request("GET", "/api/status")

    def register(self, username: str, password: str, hospital_name: str, contact_email: str) -> dict:
        """POST /api/auth/register"""
        payload = {
            "username": username,
            "password": password,
            "hospital_name": hospital_name,
            "contact_email": contact_email,
        }
        return self._request("POST", "/api/auth/register", json=payload)

    def login(self, username: str, password: str) -> dict:
        """POST /api/auth/login — sets self.token on success."""
        payload = {"username": username, "password": password}
        result = self._request("POST", "/api/auth/login", json=payload)
        self.token = result.get("access_token")
        return result

    # ── Project endpoints ─────────────────────────────────────────────────────

    def list_projects(self) -> list:
        """GET /api/projects"""
        return self._request("GET", "/api/projects")

    def join_project(self, proj_id: str, hardware_profile: dict) -> dict:
        """POST /api/projects/{proj_id}/join"""
        return self._request(
            "POST",
            f"/api/projects/{proj_id}/join",
            json={"hardware_profile": hardware_profile},
        )

    def fetch_global_model(self, proj_id: str) -> dict:
        """
        GET /api/projects/{proj_id}/model

        Returns
        -------
        dict: {'round': int, 'active_depth': int, 'weights': {param: np.ndarray}}
        """
        result = self._request("GET", f"/api/projects/{proj_id}/model")
        # Deserialise weights from lists → numpy arrays
        import numpy as np
        if "weights" in result:
            result["weights"] = {
                k: np.array(v, dtype="float32")
                for k, v in result["weights"].items()
            }
        return result

    def post_local_update(
        self,
        proj_id: str,
        weights: dict,
        num_samples: int,
        metrics: dict,
        round_id: int,
        active_depth: int,
    ) -> dict:
        """
        POST /api/projects/{proj_id}/update

        Encrypts weights before sending.
        """
        encrypted = encrypt_weights(weights)
        payload = {
            "round_id": round_id,
            "active_depth": active_depth,
            "weights": encrypted,
            "num_samples": num_samples,
            "metrics": metrics,
        }
        return self._request("POST", f"/api/projects/{proj_id}/update", json=payload)

    def get_round_history(self, proj_id: str) -> list:
        """GET /api/projects/{proj_id}/history"""
        return self._request("GET", f"/api/projects/{proj_id}/history")

    def approve_client(self, proj_id: str, user_id: str) -> dict:
        """GET /api/projects/{proj_id}/approve/{user_id}  (admin action)"""
        return self._request("GET", f"/api/projects/{proj_id}/approve/{user_id}")
