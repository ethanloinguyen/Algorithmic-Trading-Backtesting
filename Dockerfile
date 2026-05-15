FROM python:3.11-slim

# Set WORKDIR to /app/backend so that:
#   - `from app.xxx import ...` resolves correctly for uvicorn
#   - pipeline_service.py's parents[3] resolves to /app (the repo root),
#     which is where model/ and monte-carlo/ are copied
WORKDIR /app/backend

COPY Backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI application
COPY Backend/ .

# Model and simulation code imported at runtime by pipeline_service.py
COPY Model/ /app/model/
COPY monte-carlo/ /app/monte-carlo/

EXPOSE 8080

# GCP credentials are injected at runtime via Secret Manager or env vars.
# Do NOT bake the service account JSON into this image.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]