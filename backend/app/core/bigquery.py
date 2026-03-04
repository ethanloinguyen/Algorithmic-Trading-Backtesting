# backend/app/core/bigquery.py
import os
from functools import lru_cache
from google.cloud import bigquery
from app.core.config import get_settings


@lru_cache
def get_bq_client() -> bigquery.Client:
    """
    Returns a cached BigQuery client.
    Authentication is handled via GOOGLE_APPLICATION_CREDENTIALS env var,
    which points to the service account JSON key file.
    """
    settings = get_settings()
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        settings.google_application_credentials,
    )
    return bigquery.Client(project=settings.gcp_project_id)