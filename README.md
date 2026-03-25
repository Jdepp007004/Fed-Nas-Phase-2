<div align="center">

<img src="https://img.shields.io/badge/Federated%20Learning-Platform-1A73E8?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" />

# FL Platform — End-to-End Federated Learning for Clinical Data

**Privacy-preserving, distributed model training across hospital networks**  
*No patient data ever leaves the client machine.*

[Getting Started](#getting-started) · [Architecture](#architecture) · [Dataset](#dataset) · [Running](#running) · [Testing](#testing) · [Contributing](#contributing)

</div>

---

## Overview

FL Platform enables hospitals to collaboratively train a shared neural network on TCGA clinical data **without sharing raw patient records**. Each hospital trains locally on its own silo, then sends only encrypted model weight updates to a central coordinator.

### Key Features

| Feature | Description |
|---------|-------------|
| 🔒 **AES-256-GCM Encryption** | Model weights encrypted before transmission |
| 🧠 **NAS-Adaptive Subnets** | Depth auto-selected per client hardware profile |
| ⚖️ **FedProx Regularization** | Handles heterogeneous local datasets |
| 🌐 **ngrok Tunneling** | Server reachable publicly without port-forwarding |
| 📊 **Live Dashboard** | Google Material UI admin dashboard |
| 🖥️ **Browser Client** | Single HTML file — no install needed for clients |
| ✅ **CI/CD Pipeline** | GitHub Actions: lint → test → Docker build → integration |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FL Platform Server                     │
│  FastAPI + ngrok · aggregation · NAS · JWT auth · DB    │
└────────────────────────┬────────────────────────────────┘
                         │  HTTPS (ngrok tunnel)
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ Client 1 │    │ Client 2 │    │ Client 3 │  ...
   │ Hospital │    │ Hospital │    │ Hospital │
   │  Supernet│    │  Supernet│    │  Supernet│
   └──────────┘    └──────────┘    └──────────┘
   Local TCGA CSV  Local TCGA CSV  Local TCGA CSV
```

### Federated Round Sequence

```
Client                             Server
  │── GET /model ─────────────────▶│  Fetch global weights
  │◀─ {round, depth, weights} ─────│
  │   [Local training · FedProx]    │
  │── POST /update (encrypted) ────▶│
  │                                 │  FedAvg → Momentum → NAS → Validate
  │── GET /history ────────────────▶│
  │◀─ [RMSE, AUC, Tox Accuracy] ───│
      ↑ repeats until max_rounds
```

### Multi-Task Learning Targets

| Task | Target | Loss |
|------|--------|------|
| 🏥 Regression | `overall_survival` (days) | MSE |
| 💊 Toxicity Classification | `treatment_outcome` (4 classes) | CrossEntropy |
| ⚕️ Binary | `vital_status` (alive/dead) | BCE |

---

## Project Structure

```
fl_platform/
├── shared/                     # Shared by server + client
│   ├── model_schema.py         # TCGA column definitions & constants
│   └── encryption.py           # AES-256-GCM weight encryption
│
├── client/                     # Runs on each hospital machine
│   ├── supernet.py             # Depth-adaptive PyTorch Supernet (M1)
│   ├── train_loop.py           # FedProx local training loop
│   ├── data_loader.py          # TCGA data pipeline → DataLoader
│   ├── schema_validator.py     # CSV schema validation
│   ├── api_client.py           # HTTP client (retry + JWT + encryption)
│   ├── client_app.py           # CLI entry point
│   ├── client_ui.html          # Browser UI (no install required)
│   └── requirements.txt
│
├── server/                     # Runs on the central coordinator
│   ├── main.py                 # FastAPI app + lifespan + ngrok
│   ├── aggregation.py          # FedAvg + Momentum + Validation (M2)
│   ├── nas_controller.py       # NAS depth selection (M2)
│   ├── auth_router.py          # /api/auth/* (bcrypt + JWT)
│   ├── project_router.py       # /api/projects/* + round lifecycle
│   ├── db_handler.py           # Thread-safe JSON database (M3)
│   ├── ngrok_tunnel.py         # pyngrok tunnel management
│   ├── templates/dashboard.html # Server operator dashboard
│   ├── database.json           # Flat-file DB (auto-created)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_unit.py            # 12 test classes, all modules
│   └── requirements-test.txt
│
├── download_and_split.py       # Download TCGA + split for N clients
├── HOW_TO_RUN.md               # Full step-by-step guide
├── pytest.ini
├── .coveragerc
└── .github/workflows/ci.yml    # CI: lint → test → docker → integration
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Free [ngrok account](https://ngrok.com) (for public server URL)
- Internet connection (to download TCGA dataset)

### Quick Install

```bash
git clone https://github.com/YOUR_USERNAME/fl_platform.git
cd fl_platform
pip install -r server/requirements.txt
pip install -r client/requirements.txt
```

---

## Dataset

Clinical data is downloaded automatically from the [NIH GDC API](https://api.gdc.cancer.gov) — no login required.

```bash
# Download 10,000 TCGA cases and split into 4 client silos
python download_and_split.py --max-cases 10000 --n-clients 4
```

Output:
```
data/
  full_dataset.csv   ← all records
  client_1.csv       ← Teammate 1
  client_2.csv       ← Teammate 2
  client_3.csv       ← Teammate 3
  client_4.csv       ← Teammate 4
```

---

## Running

### Server (coordinator machine)

```powershell
# 1. Generate encryption key
python -c "import base64,os; print(base64.b64encode(os.urandom(32)).decode())"

# 2. Set environment variables
$env:FL_ENCRYPTION_KEY = "PASTE_KEY_HERE"
$env:JWT_SECRET        = "any_dev_secret"
$env:NGROK_AUTH_TOKEN  = "your_ngrok_token"

# 3. Install & start
pip install -r server/requirements.txt
cd server
python main.py
```

Console output:
```
[+] ngrok tunnel active: https://abc123.ngrok-free.app
INFO:     Uvicorn running on http://0.0.0.0:8000
```

- **Dashboard** → http://localhost:8000/dashboard
- **API Docs** → http://localhost:8000/docs

### Client — CLI

```powershell
pip install -r client/requirements.txt

python client/client_app.py `
  --server   https://abc123.ngrok-free.app `
  --username hospital_1 `
  --password secret1 `
  --hospital "City General Hospital" `
  --email    admin@citygeneral.org `
  --csv      data/client_1.csv `
  --proj     YOUR_PROJECT_ID
```

### Client — Browser UI

1. Open `client/client_ui.html` in any browser
2. Follow the 4-step Material Design wizard:
   - Connect → Auth → Upload CSV → Join Project
3. Wait for server admin to approve → click **Start Training**

### Docker (production server)

```bash
docker compose -f server/docker-compose.yml up --build -d
docker compose -f server/docker-compose.yml logs -f
```

Create a `.env` file at repo root:
```
FL_ENCRYPTION_KEY=your_base64_32byte_key
JWT_SECRET=your_long_random_secret
NGROK_AUTH_TOKEN=your_ngrok_token
```

---

## Testing

```bash
pip install -r tests/requirements-test.txt

# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Test coverage includes:**
- AES-256-GCM encrypt/decrypt roundtrip
- Supernet forward shapes + weight extraction
- FedAvg weighted averaging with missing subnet keys
- Nesterov momentum convergence
- NAS depth lookup bounds
- Thread-safe database operations
- Schema validation (passing + failing cases)
- TCGA data pipeline shapes
- JWT creation + tamper detection

---

## CI/CD

GitHub Actions runs on every push/PR to `main`:

| Job | What it does |
|-----|-------------|
| **Lint** | `flake8` on all Python files |
| **Test** | `pytest` with 60% coverage gate |
| **Docker Build** | Builds server image (verifies Dockerfile) |
| **Integration** | Starts real server, hits `/api/status`, registers + logs in |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FL_ENCRYPTION_KEY` | ✅ Server + Client | Base64-encoded 32-byte AES key |
| `NGROK_AUTH_TOKEN` | ✅ Server | From [ngrok dashboard](https://dashboard.ngrok.com) |
| `JWT_SECRET` | ❌ (has default) | JWT signing secret — change for production |
| `VAL_CSV_PATH` | ❌ | Path to server-side validation CSV |
| `SERVER_PORT` | ❌ (default 8000) | Port FastAPI listens on |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | PyTorch 2.0 — depth-adaptive Supernet |
| Server | FastAPI + Uvicorn |
| Tunneling | pyngrok |
| Auth | bcrypt + PyJWT (HS256) |
| Encryption | cryptography (AES-256-GCM) |
| Database | Thread-safe JSON flat-file |
| Frontend | Vanilla HTML/CSS/JS — Google Material Design |
| CI/CD | GitHub Actions |
| Container | Docker + Docker Compose |
