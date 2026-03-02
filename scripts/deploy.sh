#!/bin/bash
# ============================================================
# deploy.sh — GCP Infrastructure Deployment Script
# ============================================================
# Run once to set up all GCP resources.
# Edit PROJECT_ID and REGION before running.
# Usage: bash scripts/deploy.sh
# ============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID="capstone-487001"
REGION="us-central1"
BUCKET_NAME="capstone-bucket-487001"
BQ_DATASET="output_results"
SERVICE_ACCOUNT="jt-capstone"
SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE_NAME="us-central1-docker.pkg.dev/capstone-487001/analysis-pipeline/analysis-pipeline"

echo "================================================="
echo "Deploying Quant Lead-Lag Pipeline to GCP"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "================================================="

# ── APIs ──────────────────────────────────────────────────────────────────────
echo "[1/8] Enabling required GCP APIs..."
gcloud services enable \
    bigquery.googleapis.com \
    run.googleapis.com \
    cloudscheduler.googleapis.com \
    workflows.googleapis.com \
    pubsub.googleapis.com \
    storage.googleapis.com \
    --project="${PROJECT_ID}"
echo "  ✓ APIs enabled"

# ── Service Account ───────────────────────────────────────────────────────────
echo "[2/8] Creating service account..."
gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
    --display-name="Pipeline Runner" \
    --project="${PROJECT_ID}" \
    || echo "  Service account already exists, skipping creation"

# Grant required roles
ROLES=(
    "roles/bigquery.dataEditor"
    "roles/bigquery.jobUser"
    "roles/storage.objectAdmin"
    "roles/run.invoker"
    "roles/workflows.invoker"
)

for ROLE in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --quiet
done
echo "  ✓ Service account configured with roles"

# ── Cloud Storage Bucket ──────────────────────────────────────────────────────
echo "[3/8] Creating Cloud Storage bucket..."
gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${BUCKET_NAME}" \
    || echo "  Bucket already exists, skipping creation"
echo "  ✓ Bucket ready: gs://${BUCKET_NAME}"

# ── Build and Push Docker Image ───────────────────────────────────────────────
echo "[4/8] Building and pushing Docker image..."
gcloud builds submit \
    --tag "${IMAGE_NAME}:latest" \
    --project="${PROJECT_ID}" \
    .
echo "  ✓ Image built: ${IMAGE_NAME}:latest"

# ── Cloud Run Jobs ────────────────────────────────────────────────────────────
echo "[5/8] Creating Cloud Run Jobs..."

# Pair computation job
gcloud run jobs create "pair-job" \
    --image="${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --service-account="${SA_EMAIL}" \
    --memory="4Gi" \
    --cpu="2" \
    --task-timeout="3600s" \
    --max-retries="2" \
    --parallelism="50" \
    --tasks="50" \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET=${BUCKET_NAME},BQ_DATASET=${BQ_DATASET}" \
    --command="python" \
    --args="-m,jobs.pair_job" \
    --project="${PROJECT_ID}" \
    || gcloud run jobs update "pair-job" \
        --image="${IMAGE_NAME}:latest" \
        --region="${REGION}" \
        --project="${PROJECT_ID}"

# Aggregation job
gcloud run jobs create "aggregation-job" \
    --image="${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --service-account="${SA_EMAIL}" \
    --memory="8Gi" \
    --cpu="4" \
    --task-timeout="7200s" \
    --max-retries="1" \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET=${BUCKET_NAME},BQ_DATASET=${BQ_DATASET}" \
    --command="python" \
    --args="-m,jobs.aggregation_job" \
    --project="${PROJECT_ID}" \
    || gcloud run jobs update "aggregation-job" \
        --image="${IMAGE_NAME}:latest" \
        --region="${REGION}" \
        --project="${PROJECT_ID}"

# Residuals job
gcloud run jobs create "residuals-job" \
    --image="${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --service-account="${SA_EMAIL}" \
    --memory="4Gi" \
    --cpu="2" \
    --task-timeout="3600s" \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET=${BUCKET_NAME},BQ_DATASET=${BQ_DATASET}" \
    --command="python" \
    --args="-m,src.residuals" \
    --project="${PROJECT_ID}" \
    || echo "  residuals-job already exists, skipping"

echo "  ✓ Cloud Run Jobs created"

# ── Pub/Sub Topics ────────────────────────────────────────────────────────────
echo "[6/8] Creating Pub/Sub topics..."
gcloud pubsub topics create "pipeline-trigger" --project="${PROJECT_ID}" || true
gcloud pubsub topics create "pipeline-notifications" --project="${PROJECT_ID}" || true
echo "  ✓ Pub/Sub topics ready"

# ── Cloud Workflows ───────────────────────────────────────────────────────────
echo "[7/8] Deploying Cloud Workflow..."
gcloud workflows deploy "monthly-pipeline" \
    --location="${REGION}" \
    --source="workflows/monthly_pipeline.yaml" \
    --service-account="${SA_EMAIL}" \
    --project="${PROJECT_ID}"
echo "  ✓ Workflow deployed: monthly-pipeline"

# ── Cloud Scheduler ───────────────────────────────────────────────────────────
echo "[8/8] Creating Cloud Scheduler jobs..."

# Monthly pipeline trigger (1st of every month at 2am UTC)
gcloud scheduler jobs create http "monthly-pipeline-trigger" \
    --location="${REGION}" \
    --schedule="0 2 1 1, 4, 7, 10" \
    --uri="https://workflowexecutions.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/workflows/monthly-pipeline/executions" \
    --message-body='{"argument": "{\"trigger\": \"scheduler\"}"}' \
    --oauth-service-account-email="${SA_EMAIL}" \
    --time-zone="UTC" \
    --project="${PROJECT_ID}" \
    || gcloud scheduler jobs update http "monthly-pipeline-trigger" \
        --location="${REGION}" \
        --schedule="0 2 1 * *" \
        --project="${PROJECT_ID}"

echo "  ✓ Scheduler configured: runs 1st of every month at 02:00 UTC"

echo ""
echo "================================================="
echo "Deployment Complete!"
echo ""
echo "NEXT STEPS:"
echo "  1. Update config/config.yaml with your project ID and bucket name"
echo "  2. Run initial setup: python -m scripts.setup"
echo "  3. Run historical pipeline manually for initial backfill"
echo "  4. Deploy FastAPI backend:"
echo "     gcloud run deploy lead-lag-api --image=${IMAGE_NAME}:latest"
echo "  5. Test: gcloud scheduler jobs run monthly-pipeline-trigger --location=${REGION}"
echo "================================================="
