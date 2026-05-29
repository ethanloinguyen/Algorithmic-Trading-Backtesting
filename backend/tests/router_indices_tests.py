"""
Unit tests for app/routers/indices.py

Run with:
    python -m pytest tests/router_indices_tests.py -v

Uses FastAPI TestClient — no real BigQuery or Firestore needed.
All service calls are mocked at their source modules.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.stock import IndexSummary

client = TestClient(app)

URL = "/api/indices"


# Helpers 

def _make_index_summary(**kwargs) -> IndexSummary:
    defaults = dict(
        symbol="SPX", name="S&P 500 Index", value="5000.00",
        change="+0.50%", pct="+0.50%", price="$5000.00", positive=True,
    )
    return IndexSummary(**{**defaults, **kwargs})


def _three_indices() -> list[IndexSummary]:
    return [
        _make_index_summary(symbol="SPX",  name="S&P 500 Index",             value="5000.00"),
        _make_index_summary(symbol="IXIC", name="NASDAQ Composite",           value="16000.00"),
        _make_index_summary(symbol="DJI",  name="Dow Jones Industrial Average", value="39000.00"),
    ]


# Cache hit 

class TestCacheHit:
    def test_returns_200(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()):
            response = client.get(URL)
        assert response.status_code == 200

    def test_returns_cached_data_without_hitting_bigquery(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()) as mock_cache, \
             patch("app.routers.indices.get_index_summaries") as mock_bq:
            client.get(URL)
        mock_cache.assert_called_once()
        mock_bq.assert_not_called()

    def test_does_not_write_to_cache_on_hit(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()), \
             patch("app.routers.indices.set_cached_index_summaries") as mock_set:
            client.get(URL)
        mock_set.assert_not_called()

    def test_response_contains_data_key(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()):
            response = client.get(URL)
        assert "data" in response.json()

    def test_response_data_has_three_indices(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()):
            response = client.get(URL)
        assert len(response.json()["data"]) == 3

    def test_response_contains_correct_symbols(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()):
            response = client.get(URL)
        symbols = [item["symbol"] for item in response.json()["data"]]
        assert "SPX" in symbols
        assert "IXIC" in symbols
        assert "DJI" in symbols

    def test_response_fields_match_model(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()):
            response = client.get(URL)
        item = response.json()["data"][0]
        for field in ("symbol", "name", "value", "change", "pct", "price", "positive"):
            assert field in item, f"Missing field: {field}"


# Cache miss 

class TestCacheMiss:
    def test_returns_200_on_cache_miss(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=None), \
             patch("app.routers.indices.get_index_summaries",
                   return_value=_three_indices()), \
             patch("app.routers.indices.set_cached_index_summaries"):
            response = client.get(URL)
        assert response.status_code == 200

    def test_calls_bigquery_on_cache_miss(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=None), \
             patch("app.routers.indices.get_index_summaries",
                   return_value=_three_indices()) as mock_bq, \
             patch("app.routers.indices.set_cached_index_summaries"):
            client.get(URL)
        mock_bq.assert_called_once()

    def test_writes_to_cache_after_bigquery(self):
        summaries = _three_indices()
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=None), \
             patch("app.routers.indices.get_index_summaries",
                   return_value=summaries), \
             patch("app.routers.indices.set_cached_index_summaries") as mock_set:
            client.get(URL)
        mock_set.assert_called_once_with(summaries)

    def test_response_data_matches_bigquery_result(self):
        summaries = _three_indices()
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=None), \
             patch("app.routers.indices.get_index_summaries",
                   return_value=summaries), \
             patch("app.routers.indices.set_cached_index_summaries"):
            response = client.get(URL)
        symbols = [item["symbol"] for item in response.json()["data"]]
        assert "SPX" in symbols

    def test_cache_write_receives_same_data_returned_to_client(self):
        """The data written to cache must match what's returned in the response."""
        summaries = _three_indices()
        captured = {}
        def capture(data):
            captured["data"] = data

        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=None), \
             patch("app.routers.indices.get_index_summaries",
                   return_value=summaries), \
             patch("app.routers.indices.set_cached_index_summaries",
                   side_effect=capture):
            response = client.get(URL)

        response_symbols = {i["symbol"] for i in response.json()["data"]}
        cached_symbols   = {s.symbol for s in captured["data"]}
        assert response_symbols == cached_symbols

    def test_empty_bigquery_result_returns_empty_data(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=None), \
             patch("app.routers.indices.get_index_summaries",
                   return_value=[]), \
             patch("app.routers.indices.set_cached_index_summaries"):
            response = client.get(URL)
        assert response.status_code == 200
        assert response.json()["data"] == []


# Response shape 

class TestResponseShape:
    def test_positive_true_for_gaining_index(self):
        summaries = [_make_index_summary(symbol="SPX", positive=True)]
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=summaries):
            response = client.get(URL)
        assert response.json()["data"][0]["positive"] is True

    def test_positive_false_for_losing_index(self):
        summaries = [_make_index_summary(symbol="SPX", positive=False,
                                         change="-0.50%", pct="-0.50%")]
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=summaries):
            response = client.get(URL)
        assert response.json()["data"][0]["positive"] is False

    def test_value_is_string(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()):
            response = client.get(URL)
        assert isinstance(response.json()["data"][0]["value"], str)

    def test_content_type_is_json(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()):
            response = client.get(URL)
        assert "application/json" in response.headers["content-type"]

    def test_endpoint_only_accepts_get(self):
        with patch("app.routers.indices.get_cached_index_summaries",
                   return_value=_three_indices()):
            assert client.post(URL).status_code == 405
            assert client.put(URL).status_code  == 405
            assert client.delete(URL).status_code == 405