
# Run with: python -m pytest tests/cache_services_tests.py -v

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest

from app.models.stock import OHLCVCandle, StockSummary, IndexSummary, TimeRange
from app.services.cache_service import (
    _ohlcv_ttl,
    _is_stale,
    _ohlcv_doc_id,
    get_cached_summaries,
    set_cached_summaries,
    get_cached_index_summaries,
    set_cached_index_summaries,
    get_cached_ohlcv,
    set_cached_ohlcv,
    TTL_SUMMARIES,
    TTL_OHLCV_INTRADAY,
    TTL_OHLCV_OTHER,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_summary(**kwargs) -> StockSummary:
    defaults = dict(symbol="AAPL", name="Apple Inc.", price="$150.00",
                    change="+1.00%", volume="10.0M", positive=True)
    return StockSummary(**{**defaults, **kwargs})


def _make_index_summary(**kwargs) -> IndexSummary:
    defaults = dict(symbol="SPX", name="S&P 500", value="5000.00",
                    change="+0.50%", pct="+0.50%", price="$5000.00", positive=True)
    return IndexSummary(**{**defaults, **kwargs})


def _make_candle(**kwargs) -> OHLCVCandle:
    defaults = dict(date="Mar 01", open="150.00", high="155.00",
                    low="149.00", close="153.00", volume="5.0M")
    return OHLCVCandle(**{**defaults, **kwargs})


def _fresh_timestamp() -> datetime:
    """A timestamp 1 second ago — always considered fresh."""
    return datetime.now(timezone.utc) - timedelta(seconds=1)


def _stale_timestamp(ttl: timedelta) -> datetime:
    """A timestamp older than the given TTL — always considered stale."""
    return datetime.now(timezone.utc) - ttl - timedelta(seconds=10)


def _make_snap(exists: bool, data: dict | None = None) -> MagicMock:
    snap = MagicMock()
    snap.exists = exists
    snap.to_dict.return_value = data or {}
    return snap


def _mock_fs(snap: MagicMock) -> MagicMock:
    """Return a Firestore client mock that returns `snap` on any .get() call."""
    fs = MagicMock()
    fs.collection.return_value.document.return_value.get.return_value = snap
    return fs


# ── _ohlcv_ttl ────────────────────────────────────────────────────────────────

class TestOhlcvTtl:
    def test_one_day_returns_intraday_ttl(self):
        assert _ohlcv_ttl(TimeRange.ONE_DAY) == TTL_OHLCV_INTRADAY

    def test_one_week_returns_other_ttl(self):
        assert _ohlcv_ttl(TimeRange.ONE_WEEK) == TTL_OHLCV_OTHER

    def test_one_month_returns_other_ttl(self):
        assert _ohlcv_ttl(TimeRange.ONE_MONTH) == TTL_OHLCV_OTHER

    def test_one_year_returns_other_ttl(self):
        assert _ohlcv_ttl(TimeRange.ONE_YEAR) == TTL_OHLCV_OTHER

    def test_five_years_returns_other_ttl(self):
        assert _ohlcv_ttl(TimeRange.FIVE_YEARS) == TTL_OHLCV_OTHER

    def test_intraday_ttl_shorter_than_other(self):
        assert TTL_OHLCV_INTRADAY < TTL_OHLCV_OTHER


# ── _is_stale ─────────────────────────────────────────────────────────────────

class TestIsStale:
    def test_none_updated_at_is_stale(self):
        assert _is_stale({"updated_at": None}, TTL_SUMMARIES) is True

    def test_missing_updated_at_is_stale(self):
        assert _is_stale({}, TTL_SUMMARIES) is True

    def test_fresh_timestamp_is_not_stale(self):
        data = {"updated_at": _fresh_timestamp()}
        assert _is_stale(data, TTL_SUMMARIES) is False

    def test_expired_timestamp_is_stale(self):
        data = {"updated_at": _stale_timestamp(TTL_SUMMARIES)}
        assert _is_stale(data, TTL_SUMMARIES) is True

    def test_exactly_on_ttl_boundary_is_stale(self):
        # At exactly TTL age, it should be considered stale (> not >=)
        data = {"updated_at": datetime.now(timezone.utc) - TTL_SUMMARIES - timedelta(milliseconds=1)}
        assert _is_stale(data, TTL_SUMMARIES) is True

    def test_naive_datetime_treated_as_utc(self):
        """Naive datetimes (no tzinfo) should be handled without crashing."""
        naive = datetime.utcnow() - timedelta(hours=1)
        assert naive.tzinfo is None
        # Should not raise, and should be considered stale for a 5-min TTL
        result = _is_stale({"updated_at": naive}, TTL_SUMMARIES)
        assert result is True

    def test_fresh_naive_datetime_is_not_stale(self):
        naive = datetime.utcnow() - timedelta(seconds=1)
        assert _is_stale({"updated_at": naive}, TTL_SUMMARIES) is False

    def test_different_ttls_respected(self):
        # 30 seconds ago: stale for 1-min TTL, fresh for 5-min TTL
        ts = {"updated_at": datetime.now(timezone.utc) - timedelta(seconds=30)}
        assert _is_stale(ts, TTL_OHLCV_INTRADAY) is False  # 1 min TTL — still fresh
        assert _is_stale(ts, TTL_SUMMARIES) is False         # 5 min TTL — still fresh

        ts_old = {"updated_at": datetime.now(timezone.utc) - timedelta(seconds=90)}
        assert _is_stale(ts_old, TTL_OHLCV_INTRADAY) is True  # 1 min TTL — stale
        assert _is_stale(ts_old, TTL_SUMMARIES) is False        # 5 min TTL — still fresh


# ── _ohlcv_doc_id ─────────────────────────────────────────────────────────────

class TestOhlcvDocId:
    def test_basic_format(self):
        assert _ohlcv_doc_id("AAPL", TimeRange.ONE_DAY) == f"ohlcv_AAPL_{TimeRange.ONE_DAY.value}"

    def test_lowercase_symbol_uppercased(self):
        result = _ohlcv_doc_id("aapl", TimeRange.ONE_MONTH)
        assert result.startswith("ohlcv_AAPL_")

    def test_different_ranges_produce_different_keys(self):
        key1 = _ohlcv_doc_id("MSFT", TimeRange.ONE_DAY)
        key2 = _ohlcv_doc_id("MSFT", TimeRange.ONE_YEAR)
        assert key1 != key2

    def test_different_symbols_produce_different_keys(self):
        key1 = _ohlcv_doc_id("AAPL", TimeRange.ONE_WEEK)
        key2 = _ohlcv_doc_id("MSFT", TimeRange.ONE_WEEK)
        assert key1 != key2


# ── get_cached_summaries ──────────────────────────────────────────────────────

class TestGetCachedSummaries:
    @patch("app.services.cache_service.get_fs_client")
    def test_returns_none_when_doc_does_not_exist(self, mock_fs_client):
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=False))
        assert get_cached_summaries() is None

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_none_when_stale(self, mock_fs_client):
        data = {
            "data": [_make_summary().model_dump()],
            "updated_at": _stale_timestamp(TTL_SUMMARIES),
        }
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=True, data=data))
        assert get_cached_summaries() is None

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_summaries_when_fresh(self, mock_fs_client):
        summary = _make_summary()
        data = {
            "data": [summary.model_dump()],
            "updated_at": _fresh_timestamp(),
        }
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=True, data=data))
        result = get_cached_summaries()
        assert result is not None
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_none_on_firestore_exception(self, mock_fs_client):
        mock_fs_client.side_effect = Exception("Firestore unavailable")
        assert get_cached_summaries() is None

    @patch("app.services.cache_service.get_fs_client")
    def test_uses_custom_doc_id(self, mock_fs_client):
        fs = MagicMock()
        fs.collection.return_value.document.return_value.get.return_value = _make_snap(exists=False)
        mock_fs_client.return_value = fs

        get_cached_summaries(doc_id="index_summaries")

        fs.collection.return_value.document.assert_called_with("index_summaries")

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_multiple_summaries(self, mock_fs_client):
        summaries = [_make_summary(symbol="AAPL"), _make_summary(symbol="MSFT")]
        data = {
            "data": [s.model_dump() for s in summaries],
            "updated_at": _fresh_timestamp(),
        }
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=True, data=data))
        result = get_cached_summaries()
        assert len(result) == 2


# ── set_cached_summaries ──────────────────────────────────────────────────────

class TestSetCachedSummaries:
    @patch("app.services.cache_service.get_fs_client")
    def test_calls_firestore_set(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs
        summaries = [_make_summary()]

        set_cached_summaries(summaries)

        fs.collection.return_value.document.return_value.set.assert_called_once()

    @patch("app.services.cache_service.get_fs_client")
    def test_uses_default_doc_id(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs

        set_cached_summaries([_make_summary()])

        fs.collection.return_value.document.assert_called_with("stock_summaries")

    @patch("app.services.cache_service.get_fs_client")
    def test_uses_custom_doc_id(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs

        set_cached_summaries([_make_summary()], doc_id="my_doc")

        fs.collection.return_value.document.assert_called_with("my_doc")

    @patch("app.services.cache_service.get_fs_client")
    def test_data_serialized_with_model_dump(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs
        summary = _make_summary()

        set_cached_summaries([summary])

        call_args = fs.collection.return_value.document.return_value.set.call_args
        payload = call_args[0][0]
        assert "data" in payload
        assert payload["data"][0]["symbol"] == "AAPL"

    @patch("app.services.cache_service.get_fs_client")
    def test_does_not_raise_on_firestore_exception(self, mock_fs_client):
        mock_fs_client.side_effect = Exception("write failed")
        # Should silently swallow the exception
        set_cached_summaries([_make_summary()])


# ── get_cached_index_summaries ────────────────────────────────────────────────

class TestGetCachedIndexSummaries:
    @patch("app.services.cache_service.get_fs_client")
    def test_returns_none_when_doc_missing(self, mock_fs_client):
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=False))
        assert get_cached_index_summaries() is None

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_none_when_stale(self, mock_fs_client):
        data = {
            "data": [_make_index_summary().model_dump()],
            "updated_at": _stale_timestamp(TTL_SUMMARIES),
        }
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=True, data=data))
        assert get_cached_index_summaries() is None

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_index_summaries_when_fresh(self, mock_fs_client):
        summary = _make_index_summary()
        data = {
            "data": [summary.model_dump()],
            "updated_at": _fresh_timestamp(),
        }
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=True, data=data))
        result = get_cached_index_summaries()
        assert result is not None
        assert result[0].symbol == "SPX"

    @patch("app.services.cache_service.get_fs_client")
    def test_uses_correct_doc_id(self, mock_fs_client):
        fs = MagicMock()
        fs.collection.return_value.document.return_value.get.return_value = _make_snap(exists=False)
        mock_fs_client.return_value = fs

        get_cached_index_summaries()

        fs.collection.return_value.document.assert_called_with("index_summaries")

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_none_on_exception(self, mock_fs_client):
        mock_fs_client.side_effect = Exception("boom")
        assert get_cached_index_summaries() is None


# ── set_cached_index_summaries ────────────────────────────────────────────────

class TestSetCachedIndexSummaries:
    @patch("app.services.cache_service.get_fs_client")
    def test_calls_firestore_set(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs

        set_cached_index_summaries([_make_index_summary()])

        fs.collection.return_value.document.return_value.set.assert_called_once()

    @patch("app.services.cache_service.get_fs_client")
    def test_uses_correct_doc_id(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs

        set_cached_index_summaries([_make_index_summary()])

        fs.collection.return_value.document.assert_called_with("index_summaries")

    @patch("app.services.cache_service.get_fs_client")
    def test_does_not_raise_on_exception(self, mock_fs_client):
        mock_fs_client.side_effect = Exception("write failed")
        set_cached_index_summaries([_make_index_summary()])  # should not raise


# ── get_cached_ohlcv ──────────────────────────────────────────────────────────

class TestGetCachedOhlcv:
    @patch("app.services.cache_service.get_fs_client")
    def test_returns_none_when_doc_missing(self, mock_fs_client):
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=False))
        assert get_cached_ohlcv("AAPL", TimeRange.ONE_DAY) is None

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_none_when_stale(self, mock_fs_client):
        data = {
            "data": [_make_candle().model_dump()],
            "updated_at": _stale_timestamp(TTL_OHLCV_INTRADAY),
        }
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=True, data=data))
        assert get_cached_ohlcv("AAPL", TimeRange.ONE_DAY) is None

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_candles_when_fresh(self, mock_fs_client):
        candle = _make_candle()
        data = {
            "data": [candle.model_dump()],
            "updated_at": _fresh_timestamp(),
        }
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=True, data=data))
        result = get_cached_ohlcv("AAPL", TimeRange.ONE_MONTH)
        assert result is not None
        assert len(result) == 1
        assert result[0].close == "153.00"

    @patch("app.services.cache_service.get_fs_client")
    def test_uses_correct_doc_id_for_symbol_and_range(self, mock_fs_client):
        fs = MagicMock()
        fs.collection.return_value.document.return_value.get.return_value = _make_snap(exists=False)
        mock_fs_client.return_value = fs

        get_cached_ohlcv("aapl", TimeRange.ONE_WEEK)

        expected_doc_id = _ohlcv_doc_id("aapl", TimeRange.ONE_WEEK)
        fs.collection.return_value.document.assert_called_with(expected_doc_id)

    @patch("app.services.cache_service.get_fs_client")
    def test_intraday_uses_short_ttl(self, mock_fs_client):
        """Data 2 mins old: stale for 1D (1-min TTL), fresh for 1W (10-min TTL)."""
        two_mins_ago = datetime.now(timezone.utc) - timedelta(minutes=2)
        data = {"data": [_make_candle().model_dump()], "updated_at": two_mins_ago}
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=True, data=data))

        assert get_cached_ohlcv("AAPL", TimeRange.ONE_DAY) is None      # 1-min TTL expired
        assert get_cached_ohlcv("AAPL", TimeRange.ONE_WEEK) is not None  # 10-min TTL still valid

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_none_on_exception(self, mock_fs_client):
        mock_fs_client.side_effect = Exception("Firestore down")
        assert get_cached_ohlcv("AAPL", TimeRange.ONE_DAY) is None

    @patch("app.services.cache_service.get_fs_client")
    def test_returns_multiple_candles(self, mock_fs_client):
        candles = [_make_candle(date="Mar 01"), _make_candle(date="Mar 02")]
        data = {
            "data": [c.model_dump() for c in candles],
            "updated_at": _fresh_timestamp(),
        }
        mock_fs_client.return_value = _mock_fs(_make_snap(exists=True, data=data))
        result = get_cached_ohlcv("AAPL", TimeRange.ONE_MONTH)
        assert len(result) == 2


# ── set_cached_ohlcv ──────────────────────────────────────────────────────────

class TestSetCachedOhlcv:
    @patch("app.services.cache_service.get_fs_client")
    def test_calls_firestore_set(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs

        set_cached_ohlcv("AAPL", TimeRange.ONE_DAY, [_make_candle()])

        fs.collection.return_value.document.return_value.set.assert_called_once()

    @patch("app.services.cache_service.get_fs_client")
    def test_uses_correct_doc_id(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs

        set_cached_ohlcv("AAPL", TimeRange.ONE_WEEK, [_make_candle()])

        expected = _ohlcv_doc_id("AAPL", TimeRange.ONE_WEEK)
        fs.collection.return_value.document.assert_called_with(expected)

    @patch("app.services.cache_service.get_fs_client")
    def test_data_serialized_correctly(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs
        candle = _make_candle()

        set_cached_ohlcv("AAPL", TimeRange.ONE_MONTH, [candle])

        call_args = fs.collection.return_value.document.return_value.set.call_args
        payload = call_args[0][0]
        assert payload["data"][0]["close"] == "153.00"
        assert "updated_at" in payload

    @patch("app.services.cache_service.get_fs_client")
    def test_does_not_raise_on_exception(self, mock_fs_client):
        mock_fs_client.side_effect = Exception("write failed")
        set_cached_ohlcv("AAPL", TimeRange.ONE_DAY, [_make_candle()])  # should not raise

    @patch("app.services.cache_service.get_fs_client")
    def test_empty_candles_list_still_writes(self, mock_fs_client):
        fs = MagicMock()
        mock_fs_client.return_value = fs

        set_cached_ohlcv("AAPL", TimeRange.ONE_DAY, [])

        call_args = fs.collection.return_value.document.return_value.set.call_args
        payload = call_args[0][0]
        assert payload["data"] == []