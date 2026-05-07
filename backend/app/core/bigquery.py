# backend/app/core/bigquery.py
import os
from functools import lru_cache
from google.cloud import bigquery
from app.core.config import get_settings


@lru_cache
def get_bq_client() -> bigquery.Client:
    """
    Returns a cached BigQuery client.
    In Cloud Run, credentials are provided automatically via the attached
    service account (Application Default Credentials). Locally, set
    GOOGLE_APPLICATION_CREDENTIALS to point to a service account key file.
    """
    settings = get_settings()
    creds_file = settings.google_application_credentials
    if creds_file and os.path.isfile(creds_file):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", creds_file)
    return bigquery.Client(project=settings.gcp_project_id)