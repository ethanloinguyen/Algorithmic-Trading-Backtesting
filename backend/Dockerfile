FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=4000
EXPOSE 4000

# GCP credentials are injected at runtime via Secret Manager or env vars.
# Do NOT bake the service account JSON into this image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4000"]
