# backend/app/core/firestore.py
"""
Firestore client for the LagLens backend.

Connects to the named Firestore database (capstone-firestore) in the GCP
project, using the same service account JSON already used by BigQuery.

The database ID is read from settings.firestore_database_id so it can be
changed via the FIRESTORE_DATABASE_ID env var in backend/.env without
touching code.

The client is cached via lru_cache — only one instance per process.
"""
import os
from functools import lru_cache
from google.cloud import firestore
from app.core.config import get_settings


@lru_cache
def get_fs_client() -> firestore.Client:
    settings = get_settings()

    # In Cloud Run, ADC provides credentials via the attached service account.
    # Locally, fall back to the key file if it exists.
    creds_file = settings.google_application_credentials
    if creds_file and os.path.isfile(creds_file):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", creds_file)

    # Pass database explicitly — without this the client targets "(default)"
    # which does not exist in this project, causing "Database not found" errors.
    return firestore.Client(
        project=settings.gcp_project_id,
        database=settings.firestore_database_id,   # "capstone-firestore"
    )