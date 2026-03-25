"""
server/main.py
M3: FastAPI app entry point — mounts all routers, starts ngrok.
Owner: Sunishka Sarkar
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from auth_router import router as auth_router
from project_router import router as project_router, set_val_dataloader
from ngrok_tunnel import start_ngrok_tunnel, get_tunnel_url
from db_handler import read_db, write_db
from shared.model_schema import MODEL_CONFIG, SERVER_SCHEMA

# ─── Configuration ────────────────────────────────────────────────────────────
SERVER_HOST     = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT     = int(os.environ.get("SERVER_PORT", "8000"))
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")
SERVER_VERSION  = "1.0.0"

# ─── Server-side validation data ─────────────────────────────────────────────
VAL_CSV_PATH = os.environ.get("VAL_CSV_PATH", "")  # optional held-out CSV

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR    = os.path.join(os.path.dirname(__file__), "static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start ngrok tunnel and prepare validation dataloader on startup."""

    # ── Start ngrok ──────────────────────────────────────────────────────────
    if NGROK_AUTH_TOKEN:
        try:
            url = start_ngrok_tunnel(SERVER_PORT, NGROK_AUTH_TOKEN)
            print(f"\n[+] ngrok tunnel active: {url}\n    Share this URL with your clients.\n")
        except Exception as e:
            print(f"[!] ngrok tunnel failed to start: {e}")
    else:
        print("[!] NGROK_AUTH_TOKEN not set — running without public tunnel.")

    # ── Optionally load server-side validation DataLoader ────────────────────
    if VAL_CSV_PATH and os.path.exists(VAL_CSV_PATH):
        try:
            client_path = os.path.join(os.path.dirname(__file__), '..', 'client')
            sys.path.insert(0, client_path)
            from data_loader import build_dataloaders_from_csv
            _, val_loader = build_dataloaders_from_csv(VAL_CSV_PATH, SERVER_SCHEMA)
            set_val_dataloader(val_loader)
            print(f"[+] Server-side val DataLoader ready from {VAL_CSV_PATH}")
        except Exception as e:
            print(f"[!] Could not create server val DataLoader: {e}")

    # ── Ensure default project exists ─────────────────────────────────────────
    _ensure_default_project()

    yield  # Application runs

    print("[*] Server shutting down.")


def _ensure_default_project():
    """Create a default demo project if no projects exist."""
    import uuid, datetime
    db = read_db()
    if not db.get("projects"):
        proj = {
            "proj_id":              str(uuid.uuid4()),
            "name":                 "TCGA Federated Learning Demo",
            "admin_id":             "server",
            "data_schema":          SERVER_SCHEMA,
            "schema_version":       "1.0.0",
            "current_round":        0,
            "max_rounds":           20,
            "min_clients_per_round": 1,
            "connected_clients":    [],
            "pending_clients":      [],
            "global_model_path":    "",
            "fedprox_mu":           0.01,
            "momentum_beta":        0.9,
            "recommended_depth":    MODEL_CONFIG["max_depth"],
            "accepting_clients":    True,
            "created_at":           datetime.datetime.utcnow().isoformat() + "Z",
        }
        db["projects"].append(proj)
        write_db(db)
        print(f"[+] Created default project: {proj['proj_id']}")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FL Platform Server",
    version=SERVER_VERSION,
    description="E2E Federated Learning Platform for Heterogeneous Systems",
    lifespan=lifespan,
)

# Mount static files
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount routers
app.include_router(auth_router)
app.include_router(project_router)


# ─── Utility Endpoints ────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    """GET /api/status — health check."""
    try:
        tunnel_url = get_tunnel_url()
    except RuntimeError:
        tunnel_url = f"http://localhost:{SERVER_PORT}"
    return JSONResponse(status_code=200, content={
        "status":         "ok",
        "ngrok_url":      tunnel_url,
        "server_version": SERVER_VERSION,
    })


# ─── Server Dashboard ─────────────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """GET /dashboard — Jinja2 server operator dashboard."""
    db = read_db()
    try:
        tunnel_url = get_tunnel_url()
    except RuntimeError:
        tunnel_url = f"http://localhost:{SERVER_PORT}"

    return templates.TemplateResponse("dashboard.html", {
        "request":        request,
        "projects":       db.get("projects", []),
        "rounds_history": db.get("rounds_history", []),
        "users":          db.get("users", []),
        "ngrok_url":      tunnel_url,
        "server_version": SERVER_VERSION,
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
    )
