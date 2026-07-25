FROM python:3.11-slim AS base

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY shared/ /app/shared/
COPY server/ /app/server/

# Create directories that must exist at runtime
RUN mkdir -p /app/server/models /app/server/static

# ── Runtime configuration ──────────────────────────────────────────────────────
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

WORKDIR /app/server

# Default: run the FastAPI server.
# Override CMD in docker-compose for Celery worker / beat / migration.
CMD ["python", "main.py"]
