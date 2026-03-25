FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy shared module first (needed by server code)
COPY shared/ /app/shared/

# Copy server code
COPY server/ /app/server/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Environment variables (override via docker-compose or -e flags)
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV PYTHONUNBUFFERED=1

WORKDIR /app/server

CMD ["python", "main.py"]
