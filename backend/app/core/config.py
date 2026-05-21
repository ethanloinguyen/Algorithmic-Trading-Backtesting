# backend/app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Google Cloud — common
    google_application_credentials: str = "./secrets/gcp-service-account.json"
    gcp_project_id: str

    # BigQuery
    bq_dataset:      str = "output_results"
    bq_ohlcv_table:  str = "market_data"
    bq_stocks_table: str = "ticker_metadata"

    # Firestore — named database (not the default "(default)" instance)
    # Set in backend/.env as:  FIRESTORE_DATABASE_ID=capstone-firestore
    firestore_database_id: str = "capstone-firestore"

    # Server
    host: str = "0.0.0.0"
    port: int = 4000

    # CORS — comma-separated origins, e.g. "http://localhost:3000"
    allowed_origins: str = "http://localhost:3000"

    # Portfolio diversification — use mock data until BQ final_network is populated
    use_mock_data: bool = True

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def fq_market_data(self) -> str:
        return f"`{self.gcp_project_id}.{self.bq_dataset}.{self.bq_ohlcv_table}`"

    @property
    def fq_general_market_data(self) -> str:
        """
        Extra general stocks outside the lead-lag universe (GOOG, PLTR, LMND, etc.).
        Lives in output_results (same dataset as market_data) so the service account
        has write access.  Never touched by the Algorithm pipeline.
        """
        return f"`{self.gcp_project_id}.{self.bq_dataset}.general_market_data`"

    @property
    def fq_ticker_metadata(self) -> str:
        return f"`{self.gcp_project_id}.{self.bq_dataset}.{self.bq_stocks_table}`"

    @property
    def fq_quality_picks(self) -> str:
        """Fully-qualified path to the nightly quality_picks_scores table."""
        return f"`{self.gcp_project_id}.{self.bq_dataset}.quality_picks_scores`"


@lru_cache
def get_settings() -> Settings:
    return Settings()