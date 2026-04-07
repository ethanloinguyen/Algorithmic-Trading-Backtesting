FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY jobs/ ./jobs/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY workflows/ ./workflows/

# Default command (overridden per job in Cloud Run)
CMD ["python", "-m", "jobs.pair_job"]
