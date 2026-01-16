# Fotbollspredictor v7 - Dockerfile
# Multi-stage build för optimal bildstorlek

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Installera build-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Kopiera requirements och installera dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Kopiera installerade paket från builder
COPY --from=builder /root/.local /root/.local

# Lägg till lokala paket i PATH
ENV PATH=/root/.local/bin:$PATH

# Kopiera applikationskod
COPY . .

# Skapa nödvändiga kataloger
RUN mkdir -p data models

# Sätt miljövariabler
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Exponera Streamlit-porten
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Installera curl för healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Träna modellen vid container start (om den inte finns)
# Detta kan ta några minuter första gången
RUN python main.py || echo "Initial training skipped (data may not be available)"

# Starta Streamlit-appen
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
