"""
server/main.py
M3: FastAPI app entry point — mounts all routers, starts ngrok.
Owner: Sunishka Sarkar
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

from auth_router import router as auth_router  # noqa: E402
from project_router import router as project_router, set_val_dataloader  # noqa: E402
from demo_router import router as demo_router, sim_router  # noqa: E402
from ngrok_tunnel import start_ngrok_tunnel, get_tunnel_url  # noqa: E402
from db_handler import read_db, write_db  # noqa: E402
from shared.model_schema import MODEL_CONFIG, SERVER_SCHEMA  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import Response  # noqa: E402

# Phase 2C: observability (graceful degradation if not installed)
try:
    from otel_setup import _try_init as _otel_init
    _HAVE_OTEL = True
except Exception:  # noqa: BLE001
    _HAVE_OTEL = False

try:
    from metrics import prometheus_response
    _HAVE_METRICS = True
except Exception:  # noqa: BLE001
    _HAVE_METRICS = False

# ─── Configuration ────────────────────────────────────────────────────────────
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")
SERVER_VERSION = "1.0.0"

# ─── Server-side validation data ─────────────────────────────────────────────
VAL_CSV_PATH = os.environ.get("VAL_CSV_PATH", "")  # optional held-out CSV

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start ngrok tunnel and prepare validation dataloader on startup."""

    # ── OTel initialisation ──────────────────────────────────────────────────
    if _HAVE_OTEL:
        _otel_init()

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
    """Create a default project if no projects exist."""
    import uuid, datetime  # noqa: E401
    db = read_db()
    if not db.get("projects"):
        proj = {
            "proj_id":              str(uuid.uuid4()),
            "name":                 "TCGA Federated Learning",
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
            "created_at":           datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin since clients might be on localhost or ngrok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount routers
app.include_router(auth_router)
app.include_router(project_router)
app.include_router(sim_router)   # /api/sim/*  — used by simulation page
app.include_router(demo_router)  # /api/demo/* — backward-compat alias


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


@app.get("/metrics")
async def metrics_endpoint():
    """
    GET /metrics — Prometheus scrape endpoint.
    Returns metrics in prometheus_client exposition format.
    If prometheus_client is not installed, returns a 200 with a note.
    """
    if _HAVE_METRICS:
        body, content_type = prometheus_response()
        return Response(content=body, media_type=content_type)
    return Response(
        content=b"# prometheus-client not installed\n",
        media_type="text/plain",
        status_code=200,
    )

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
        "jwt_secret":     os.environ.get("JWT_SECRET", "dev_secret_change_in_production"),
    })


@app.get("/simulation", response_class=HTMLResponse)
async def simulation(request: Request):
    """Live federated session viewer — 4 hospitals join and train round-by-round."""
    return templates.TemplateResponse("simulation.html", {"request": request})


@app.get("/demo", response_class=HTMLResponse)
async def demo_redirect(request: Request):
    """Backward-compat redirect to /simulation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/simulation", status_code=301)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
    )
