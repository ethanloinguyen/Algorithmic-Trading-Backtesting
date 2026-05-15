# Algorithm/Dockerfile
# --------------------
# Image for ALL Algorithm pipeline jobs:
#   - quality_picks_job.py  (nightly, single instance)
#   - pair_job.py           (quarterly, 50 parallel tasks)
#   - aggregation_job.py    (quarterly, single instance after pairs)
#
# Build from repo root:
#   docker build -f Algorithm/Dockerfile -t algorithm-jobs:latest .
#
# The default CMD runs quality_picks_job (nightly Cloud Run Job).
# Override CMD in each Cloud Run Job definition for the other jobs:
#   pair_job        → ["python", "-m", "Algorithm.jobs.pair_job"]
#   aggregation_job → ["python", "-m", "Algorithm.jobs.aggregation_job"]

FROM python:3.11-slim

# System deps needed by numba (LLVM) and scipy (BLAS/LAPACK)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies from Algorithm/requirements.txt
COPY Algorithm/requirements.txt /app/Algorithm/requirements.txt
RUN pip install --no-cache-dir -r /app/Algorithm/requirements.txt

# Copy Algorithm source (jobs, src, scripts, config)
COPY Algorithm/ /app/Algorithm/

# Copy shared model artifacts if they exist at the repo root level
# (aggregation_job may reference these; safe to include even if unused by other jobs)
COPY Model/ /app/Model/

# entrypoint.sh maps CLOUD_RUN_TASK_INDEX → PARTITION_ID and
# CLOUD_RUN_TASK_COUNT → NUM_PARTITIONS before exec'ing the job.
# This is the fix for the double-parallelization issue from the previous run.
COPY Algorithm/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Add /app to PYTHONPATH so `from Algorithm.src.xxx import` resolves correctly
ENV PYTHONPATH=/app

ENTRYPOINT ["/app/entrypoint.sh"]

# Default: nightly quality_picks job (single Cloud Run task, no parallelism needed)
CMD ["python", "-m", "Algorithm.jobs.quality_picks_job"]
