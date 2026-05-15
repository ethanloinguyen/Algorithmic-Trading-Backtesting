#!/usr/bin/env bash
# entrypoint.sh
# -------------
# Cloud Run Jobs provides CLOUD_RUN_TASK_INDEX (0-based) and CLOUD_RUN_TASK_COUNT
# as native environment variables when running multi-task jobs.
#
# pair_job.py reads PARTITION_ID and NUM_PARTITIONS instead.
# This entrypoint maps the Cloud Run vars to what the job expects,
# then exec's whatever CMD was passed — so it works for all Algorithm jobs
# (pair_job, quality_picks_job, aggregation_job) from a single image.
#
# Cloud Run Job definitions override CMD per job; e.g.:
#   pair_job          → python -m Algorithm.jobs.pair_job
#   quality_picks_job → python -m Algorithm.jobs.quality_picks_job
#   aggregation_job   → python -m Algorithm.jobs.aggregation_job

set -euo pipefail

# Only map partition vars when CLOUD_RUN_TASK_INDEX is actually present.
# This keeps the script safe for local runs that set PARTITION_ID directly.
if [[ -n "${CLOUD_RUN_TASK_INDEX:-}" ]]; then
    export PARTITION_ID="${CLOUD_RUN_TASK_INDEX}"
fi

if [[ -n "${CLOUD_RUN_TASK_COUNT:-}" ]]; then
    export NUM_PARTITIONS="${CLOUD_RUN_TASK_COUNT}"
fi

exec "$@"
