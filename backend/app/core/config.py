# backend/app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Google Cloud
    google_application_credentials: str = "./secrets/gcp-service-account.json"
    gcp_project_id: str
    bq_dataset: str = "stock_data"
    bq_ohlcv_table: str = "ohlcv_daily"
    bq_stocks_table: str = "stocks"

    # Server
    host: str = "0.0.0.0"
    port: int = 4000

    # CORS — comma-separated origins
    allowed_origins: str = "http://localhost:3000"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def fq_ohlcv(self) -> str:
        """Fully-qualified BigQuery table: `project.dataset.table`"""
        return f"`{self.gcp_project_id}.{self.bq_dataset}.{self.bq_ohlcv_table}`"

    @property
    def fq_stocks(self) -> str:
        return f"`{self.gcp_project_id}.{self.bq_dataset}.{self.bq_stocks_table}`"


@lru_cache
def get_settings() -> Settings:
    return Settings()