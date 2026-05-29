#!/usr/bin/env bash
# deploy_quality_picks.sh
# -----------------------
# Builds and pushes the algorithm-jobs Docker image, creates or updates the
# Cloud Run Job for quality_picks_job, and registers the Cloud Scheduler job
# that triggers it nightly after the daily data ingest completes.
#
# Run from the repo root:
#   bash Algorithm/scripts/deploy_quality_picks.sh
#
# Prerequisites:
#   gcloud auth login
#   gcloud auth configure-docker
#   gcloud config set project capstone-487001

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ID="capstone-487001"
REGION="us-central1"
SERVICE_ACCOUNT="jt-capstone@capstone-487001.iam.gserviceaccount.com"

IMAGE="gcr.io/${PROJECT_ID}/quality-picks:latest"

JOB_NAME="quality-picks-job"
MEMORY="4Gi"
CPU="2"
TIMEOUT="3600"   # seconds — quality_picks typically finishes in 5–15 min;
                 # ceiling is generous to handle BQ slot contention

# Schedule: runs nightly after the daily data ingest (update_data) finishes.
# Adjust this cron to be ~1 hour after your update_data Cloud Scheduler fires.
# Format: "MIN HOUR * * *" in UTC.
# Default: 23:30 UTC = 7:30 PM ET  (assumes update_data starts ~5:30 PM ET)
# ⚠️  Change this to match your actual update_data completion time.
SCHEDULE="30 23 * * *"
SCHEDULER_JOB_NAME="quality-picks-nightly"
SCHEDULER_TIMEZONE="America/New_York"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: Build and push the Docker image ───────────────────────────────────
# Must be run from the repo root so the build context includes both
# Algorithm/ and Model/ directories (see Algorithm/Dockerfile COPY directives).

log "Building quality-picks image..."
docker build \
    --platform linux/amd64 \
    -f Algorithm/Dockerfile.quality_picks \
    -t "${IMAGE}" \
    .

log "Pushing ${IMAGE}..."
docker push "${IMAGE}"

# ── Step 2: Create or update the Cloud Run Job ────────────────────────────────
# Uses --command to keep the Dockerfile's default CMD (quality_picks_job).
# Single task (--tasks 1) — no parallelism needed for this job.

log "Deploying Cloud Run Job: ${JOB_NAME}..."

if gcloud run jobs describe "${JOB_NAME}" --region="${REGION}" --project="${PROJECT_ID}" \
       &>/dev/null; then
    log "Job exists — updating..."
    gcloud run jobs update "${JOB_NAME}" \
        --region="${REGION}" \
        --project="${PROJECT_ID}" \
        --image="${IMAGE}" \
        --service-account="${SERVICE_ACCOUNT}" \
        --memory="${MEMORY}" \
        --cpu="${CPU}" \
        --task-timeout="${TIMEOUT}" \
        --tasks=1 \
        --max-retries=1
else
    log "Job not found — creating..."
    gcloud run jobs create "${JOB_NAME}" \
        --region="${REGION}" \
        --project="${PROJECT_ID}" \
        --image="${IMAGE}" \
        --service-account="${SERVICE_ACCOUNT}" \
        --memory="${MEMORY}" \
        --cpu="${CPU}" \
        --task-timeout="${TIMEOUT}" \
        --tasks=1 \
        --max-retries=1
fi

# ── Step 3: Create or update the Cloud Scheduler job ─────────────────────────
# Triggers the Cloud Run Job on the nightly cron schedule.
# The scheduler invokes the Cloud Run Jobs API directly (no HTTP endpoint needed).

SCHEDULER_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"

log "Configuring Cloud Scheduler: ${SCHEDULER_JOB_NAME}..."

if gcloud scheduler jobs describe "${SCHEDULER_JOB_NAME}" \
       --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    log "Scheduler job exists — updating..."
    gcloud scheduler jobs update http "${SCHEDULER_JOB_NAME}" \
        --location="${REGION}" \
        --project="${PROJECT_ID}" \
        --schedule="${SCHEDULE}" \
        --time-zone="${SCHEDULER_TIMEZONE}" \
        --uri="${SCHEDULER_URI}" \
        --http-method=POST \
        --oauth-service-account-email="${SERVICE_ACCOUNT}" \
        --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform" \
        --attempt-deadline="1800s" \
        --description="Nightly quality_picks_job trigger — runs after daily data ingest"
else
    log "Scheduler job not found — creating..."
    gcloud scheduler jobs create http "${SCHEDULER_JOB_NAME}" \
        --location="${REGION}" \
        --project="${PROJECT_ID}" \
        --schedule="${SCHEDULE}" \
        --time-zone="${SCHEDULER_TIMEZONE}" \
        --uri="${SCHEDULER_URI}" \
        --http-method=POST \
        --oauth-service-account-email="${SERVICE_ACCOUNT}" \
        --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform" \
        --attempt-deadline="1800s" \
        --description="Nightly quality_picks_job trigger — runs after daily data ingest"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

log "✓ Deployment complete."
log ""
log "  Cloud Run Job : ${JOB_NAME} (${REGION})"
log "  Schedule      : ${SCHEDULE} ${SCHEDULER_TIMEZONE} → ${SCHEDULER_JOB_NAME}"
log "  Image         : ${IMAGE}"
log ""
log "To trigger a manual test run:"
log "  gcloud run jobs execute ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID} --wait"
log ""
log "To view logs from the last run:"
log "  gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}' \\"
log "      --project=${PROJECT_ID} --limit=100 --format='value(textPayload)'"
