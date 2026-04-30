# run with python -m pytest tests/test_bigquery_services.py -v
# unit tests to test bigquery_services and api calls

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.models.stock import TimeRange
from app.services.bigquery_services import (
    _abbreviate_volume,
    _format_price,
    _format_date_label,
    _summaries_from_query,
    get_ohlcv,
    get_stock_summaries,
    get_index_summaries,
    get_stock_detail,
    get_pair_data,
    get_network_data,
    RANGE_DAYS,
    INDEX_META,
)

class TestAbbreviateVolume:
    def test_billions(self):
        assert _abbreviate_volume(2_500_000_000) == "2.5B"

    def test_billions_exact(self):
        assert _abbreviate_volume(1_000_000_000) == "1.0B"

    def test_millions(self):
        assert _abbreviate_volume(3_750_000) == "3.8M"

    def test_millions_exact(self):
        assert _abbreviate_volume(1_000_000) == "1.0M"

    def test_thousands(self):
        assert _abbreviate_volume(45_000) == "45.0K"

    def test_thousands_exact(self):
        assert _abbreviate_volume(1_000) == "1.0K"

    def test_small_number(self):
        assert _abbreviate_volume(999) == "999"

    def test_zero(self):
        assert _abbreviate_volume(0) == "0"

    def test_float_input(self):
        assert _abbreviate_volume(1_500_000.0) == "1.5M"

class TestFormatPrice:
    def test_basic(self):
        assert _format_price(123.45) == "$123.45"

    def test_thousands_separator(self):
        assert _format_price(1234.56) == "$1,234.56"

    def test_large_number(self):
        assert _format_price(1_000_000.00) == "$1,000,000.00"

    def test_rounds_to_two_decimal_places(self):
        assert _format_price(9.999) == "$10.00"

    def test_zero(self):
        assert _format_price(0.0) == "$0.00"

class TestFormatDateLabel:
    def test_one_day_formats_as_time(self):
        with patch("app.services.bigquery_services.datetime") as mock_dt:
            mock_d = MagicMock()
            mock_d.strftime.return_value = "9:30 AM"
            mock_dt.fromisoformat.return_value = mock_d
            result = _format_date_label("2024-03-15T09:30:00", TimeRange.ONE_DAY)
        assert result == "9:30 AM"
        mock_d.strftime.assert_called_once_with("%-I:%M %p")

    def test_one_day_space_separator(self):
        """BigQuery sometimes uses a space instead of T — both are parsed."""
        with patch("app.services.bigquery_services.datetime") as mock_dt:
            mock_d = MagicMock()
            mock_d.strftime.return_value = "2:00 PM"
            mock_dt.fromisoformat.return_value = mock_d
            result = _format_date_label("2024-03-15 14:00:00", TimeRange.ONE_DAY)
        assert result == "2:00 PM"

    def test_five_years_formats_as_month_year(self):
        result = _format_date_label("2024-03-15T00:00:00", TimeRange.FIVE_YEARS)
        assert "Mar" in result and "24" in result

    def test_other_ranges_format_as_month_day(self):
        result = _format_date_label("2024-03-01T00:00:00", TimeRange.ONE_MONTH)
        assert result == "Mar 01"

    def test_invalid_date_returns_raw_string(self):
        raw = "not-a-date"
        result = _format_date_label(raw, TimeRange.ONE_WEEK)
        assert result == raw

    def test_one_week_range(self):
        result = _format_date_label("2024-06-15T00:00:00", TimeRange.ONE_WEEK)
        assert result == "Jun 15"

    def test_one_year_range(self):
        result = _format_date_label("2024-12-25T00:00:00", TimeRange.ONE_YEAR)
        assert result == "Dec 25"

def _make_row(ticker, company_name, today_close, prev_close, today_volume):
    row = MagicMock()
    row.ticker        = ticker
    row.company_name  = company_name
    row.today_close   = today_close
    row.prev_close    = prev_close
    row.today_volume  = today_volume
    return row


class TestSummariesFromQuery:
    def test_positive_change(self):
        row = _make_row("AAPL", "Apple Inc.", 110.0, 100.0, 1_000_000)
        results = _summaries_from_query([row])
        assert len(results) == 1
        s = results[0]
        assert s.symbol   == "AAPL"
        assert s.positive is True
        assert s.change   == "+10.00%"

    def test_negative_change(self):
        row = _make_row("TSLA", "Tesla", 90.0, 100.0, 500_000)
        results = _summaries_from_query([row])
        s = results[0]
        assert s.positive is False
        assert "-10.00%" in s.change

    def test_zero_prev_close_does_not_crash(self):
        row = _make_row("XYZ", "XYZ Corp", 50.0, 0.0, 100)
        results = _summaries_from_query([row])
        assert results[0].change == "+0.00%"

    def test_price_formatted_with_dollar_sign(self):
        row = _make_row("MSFT", "Microsoft", 300.0, 290.0, 2_000_000)
        results = _summaries_from_query([row])
        assert results[0].price.startswith("$")

    def test_volume_abbreviated(self):
        row = _make_row("NVDA", "NVIDIA", 500.0, 490.0, 50_000_000)
        results = _summaries_from_query([row])
        assert results[0].volume == "50.0M"

    def test_empty_rows_returns_empty_list(self):
        assert _summaries_from_query([]) == []

    def test_multiple_rows(self):
        rows = [
            _make_row("AAPL", "Apple",     150.0, 140.0, 1_000_000),
            _make_row("GOOG", "Alphabet",  100.0, 105.0, 2_000_000),
        ]
        results = _summaries_from_query(rows)
        assert len(results) == 2

class TestGetStockSummaries:
    def test_empty_symbols_returns_empty_list(self):
        """Should short-circuit without hitting BigQuery."""
        result = get_stock_summaries([])
        assert result == []

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_symbols_uppercased_in_query(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data     = "proj.ds.market_data"
        mock_settings.return_value.fq_ticker_metadata = "proj.ds.ticker_metadata"

        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client.return_value.query.return_value = mock_job

        get_stock_summaries(["aapl", "msft"])

        call = mock_client.return_value.query.call_args
        # job_config may be passed as a positional or keyword arg depending on platform
        job_config = call.kwargs.get("job_config") if call.kwargs.get("job_config") else call[1].get("job_config") if len(call) > 1 and isinstance(call[1], dict) else call[0][1]
        params = job_config.query_parameters
        # ArrayQueryParameter uses .values (plural), ScalarQueryParameter uses .value
        array_param = next(p for p in params if hasattr(p, "values"))
        assert all(s == s.upper() for s in array_param.values)

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_returns_stock_summaries(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data     = "proj.ds.market_data"
        mock_settings.return_value.fq_ticker_metadata = "proj.ds.ticker_metadata"

        row = _make_row("AAPL", "Apple Inc.", 150.0, 140.0, 5_000_000)
        mock_job = MagicMock()
        mock_job.result.return_value = [row]
        mock_client.return_value.query.return_value = mock_job

        result = get_stock_summaries(["AAPL"])
        assert len(result) == 1
        assert result[0].symbol == "AAPL"


class TestGetOHLCV:
    def _make_ohlcv_row(self, date, open_, high, low, close, volume):
        row = MagicMock()
        row.date   = date
        row.open   = open_
        row.high   = high
        row.low    = low
        row.close  = close
        row.volume = volume
        return row

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_returns_candles(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data = "proj.ds.market_data"

        row = self._make_ohlcv_row(
            "2024-03-01T09:30:00", 150.0, 155.0, 149.0, 153.0, 1_000_000
        )
        mock_job = MagicMock()
        mock_job.result.return_value = [row]
        mock_client.return_value.query.return_value = mock_job

        candles = get_ohlcv("AAPL", TimeRange.ONE_MONTH)
        assert len(candles) == 1
        c = candles[0]
        assert c.open  == "150.00"
        assert c.high  == "155.00"
        assert c.low   == "149.00"
        assert c.close == "153.00"

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_symbol_uppercased(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data = "proj.ds.market_data"
        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client.return_value.query.return_value = mock_job

        get_ohlcv("aapl", TimeRange.ONE_WEEK)

        params = mock_client.return_value.query.call_args[1]["job_config"].query_parameters
        symbol_param = next(p for p in params if p.name == "symbol")
        assert symbol_param.value == "AAPL"

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_correct_days_used_for_range(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data = "proj.ds.market_data"
        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client.return_value.query.return_value = mock_job

        get_ohlcv("AAPL", TimeRange.ONE_YEAR)

        params = mock_client.return_value.query.call_args[1]["job_config"].query_parameters
        days_param = next(p for p in params if p.name == "days")
        assert days_param.value == RANGE_DAYS[TimeRange.ONE_YEAR]  # 365

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_volume_abbreviated_in_candle(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data = "proj.ds.market_data"
        row = self._make_ohlcv_row("2024-01-01T09:30:00", 100, 110, 99, 105, 25_000_000)
        mock_job = MagicMock()
        mock_job.result.return_value = [row]
        mock_client.return_value.query.return_value = mock_job

        candles = get_ohlcv("MSFT", TimeRange.ONE_MONTH)
        assert candles[0].volume == "25.0M"

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_empty_result_returns_empty_list(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data = "proj.ds.market_data"
        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client.return_value.query.return_value = mock_job

        assert get_ohlcv("FAKE", TimeRange.ONE_DAY) == []


class TestGetIndexSummaries:
    @patch("app.services.bigquery_services.get_stock_summaries")
    def test_returns_index_summaries_with_correct_names(self, mock_summaries):
        from app.models.stock import StockSummary
        mock_summaries.return_value = [
            StockSummary(symbol="SPX",  name="SPX",  price="$5000.00", change="+1.00%", volume="1.0B", positive=True),
            StockSummary(symbol="IXIC", name="IXIC", price="$16000.00", change="-0.50%", volume="2.0B", positive=False),
            StockSummary(symbol="DJI",  name="DJI",  price="$39000.00", change="+0.20%", volume="500.0M", positive=True),
        ]

        results = get_index_summaries()
        names = {r.symbol: r.name for r in results}

        assert names["SPX"]  == INDEX_META["SPX"]
        assert names["IXIC"] == INDEX_META["IXIC"]
        assert names["DJI"]  == INDEX_META["DJI"]

    @patch("app.services.bigquery_services.get_stock_summaries")
    def test_price_field_stripped_of_dollar_sign(self, mock_summaries):
        from app.models.stock import StockSummary
        mock_summaries.return_value = [
            StockSummary(symbol="SPX", name="SPX", price="$5,000.00", change="+1.00%", volume="1.0B", positive=True),
        ]

        results = get_index_summaries()
        assert not results[0].value.startswith("$")
        assert "," not in results[0].value

class TestGetStockDetail:
    def _make_detail_row(self, company_name, sector, industry, market_cap, pe_ratio, high_52w, low_52w):
        row = MagicMock()
        row.company_name = company_name
        row.sector       = sector
        row.industry     = industry
        row.market_cap   = market_cap
        row.pe_ratio     = pe_ratio
        row.high_52w     = high_52w
        row.low_52w      = low_52w
        return row

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_returns_stock_detail(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data     = "proj.ds.market_data"
        mock_settings.return_value.fq_ticker_metadata = "proj.ds.ticker_metadata"

        row = self._make_detail_row("Apple Inc.", "Technology", "Consumer Electronics", 3e12, 28.5, 200.0, 130.0)
        mock_job = MagicMock()
        mock_job.result.return_value = [row]
        mock_client.return_value.query.return_value = mock_job

        detail = get_stock_detail("AAPL")
        assert detail.symbol   == "AAPL"
        assert detail.name     == "Apple Inc."
        assert detail.sector   == "Technology"
        assert detail.pe_ratio == 28.5
        assert detail.high_52w == 200.0
        assert detail.low_52w  == 130.0

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_no_rows_returns_fallback(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data     = "proj.ds.market_data"
        mock_settings.return_value.fq_ticker_metadata = "proj.ds.ticker_metadata"

        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client.return_value.query.return_value = mock_job

        detail = get_stock_detail("FAKE")
        assert detail.symbol == "FAKE"
        assert detail.name   == "FAKE"

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_none_pe_ratio_handled(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data     = "proj.ds.market_data"
        mock_settings.return_value.fq_ticker_metadata = "proj.ds.ticker_metadata"

        row = self._make_detail_row("Corp", "Finance", "Banking", 1e9, None, 100.0, 80.0)
        mock_job = MagicMock()
        mock_job.result.return_value = [row]
        mock_client.return_value.query.return_value = mock_job

        detail = get_stock_detail("XYZ")
        assert detail.pe_ratio is None

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_symbol_uppercased(self, mock_settings, mock_client):
        mock_settings.return_value.fq_market_data     = "proj.ds.market_data"
        mock_settings.return_value.fq_ticker_metadata = "proj.ds.ticker_metadata"

        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client.return_value.query.return_value = mock_job

        detail = get_stock_detail("aapl")
        assert detail.symbol == "AAPL"

class TestGetPairData:
    def _make_pair_row(self):
        row = MagicMock()
        row.ticker_i        = "AAPL"
        row.ticker_j        = "MSFT"
        row.best_lag        = 2
        row.mean_dcor       = 0.3456
        row.signal_strength = 72.5
        row.frequency       = 0.123
        row.half_life       = 5.0
        row.oos_sharpe_net  = 1.234
        row.sector_i        = "Technology"
        row.sector_j        = "Technology"
        return row

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_not_found_returns_found_false(self, mock_settings, mock_client):
        mock_settings.return_value.gcp_project_id = "proj"
        mock_settings.return_value.bq_dataset     = "ds"

        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client.return_value.query.return_value = mock_job

        import app.services.portfolio_service as ps
        original = ps._TABLE_NAMES
        ps._TABLE_NAMES = {"broad_market": "final_network"}
        try:
            result = get_pair_data("AAPL", "MSFT")
        finally:
            ps._TABLE_NAMES = original

        assert result.found is False
        assert result.ticker_i == "AAPL"
        assert result.ticker_j == "MSFT"

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_found_returns_pair_detail(self, mock_settings, mock_client):
        mock_settings.return_value.gcp_project_id = "proj"
        mock_settings.return_value.bq_dataset     = "ds"

        row = self._make_pair_row()
        mock_job = MagicMock()
        mock_job.result.return_value = [row]
        mock_client.return_value.query.return_value = mock_job

        import app.services.portfolio_service as ps
        original = ps._TABLE_NAMES
        ps._TABLE_NAMES = {"broad_market": "final_network"}
        try:
            result = get_pair_data("AAPL", "MSFT")
        finally:
            ps._TABLE_NAMES = original

        assert result.found           is True
        assert result.ticker_i        == "AAPL"
        assert result.best_lag        == 2
        assert result.signal_strength == 72.5


class TestGetNetworkData:
    def _make_network_row(self, ti="AAPL", tj="MSFT", signal=70.0, lag=1, dcor=0.3, si="Tech", sj="Tech", ci=0.5, cj=0.4):
        row = MagicMock()
        row.ticker_i        = ti
        row.ticker_j        = tj
        row.signal_strength = signal
        row.best_lag        = lag
        row.mean_dcor       = dcor
        row.sector_i        = si
        row.sector_j        = sj
        row.cent_i          = ci
        row.cent_j          = cj
        return row

    def _make_row_for(self, ti, tj):
        """Convenience: make a row with just custom ti/tj, all other defaults."""
        return self._make_network_row(ti=ti, tj=tj)

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_returns_network_response(self, mock_settings, mock_client):
        mock_settings.return_value.gcp_project_id = "proj"
        mock_settings.return_value.bq_dataset     = "ds"

        row = self._make_network_row()
        mock_job = MagicMock()
        mock_job.result.return_value = [row]
        mock_client.return_value.query.return_value = mock_job

        import app.services.portfolio_service as ps
        original = ps._TABLE_NAMES
        ps._TABLE_NAMES = {"broad_market": "final_network"}
        try:
            result = get_network_data()
        finally:
            ps._TABLE_NAMES = original

        assert len(result.nodes) == 2
        assert len(result.edges) == 1
        assert result.analysis_mode == "broad_market"

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_fallback_query_used_on_exception(self, mock_settings, mock_client):
        """When the primary query (with centrality) fails, the fallback runs."""
        mock_settings.return_value.gcp_project_id = "proj"
        mock_settings.return_value.bq_dataset     = "ds"

        mock_job_fail = MagicMock()
        mock_job_fail.result.side_effect = Exception("column not found")

        row = self._make_network_row(ci=0.0, cj=0.0)
        mock_job_ok = MagicMock()
        mock_job_ok.result.return_value = [row]

        mock_client.return_value.query.side_effect = [mock_job_fail, mock_job_ok]

        import app.services.portfolio_service as ps
        original = ps._TABLE_NAMES
        ps._TABLE_NAMES = {"broad_market": "final_network"}
        try:
            result = get_network_data()
        finally:
            ps._TABLE_NAMES = original

        assert len(result.edges) == 1

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_empty_rows_returns_empty_nodes_and_edges(self, mock_settings, mock_client):
        mock_settings.return_value.gcp_project_id = "proj"
        mock_settings.return_value.bq_dataset     = "ds"

        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client.return_value.query.return_value = mock_job

        import app.services.portfolio_service as ps
        original = ps._TABLE_NAMES
        ps._TABLE_NAMES = {"broad_market": "final_network"}
        try:
            result = get_network_data()
        finally:
            ps._TABLE_NAMES = original

        assert result.nodes == []
        assert result.edges == []

    @patch("app.services.bigquery_services.get_bq_client")
    @patch("app.services.bigquery_services.get_settings")
    def test_out_degree_counted_correctly(self, mock_settings, mock_client):
        """AAPL appears as ticker_i in 2 rows, so out_degree should be 2."""
        mock_settings.return_value.gcp_project_id = "proj"
        mock_settings.return_value.bq_dataset     = "ds"

        rows = [
            self._make_row_for("AAPL", "MSFT"),
            self._make_row_for("AAPL", "NVDA"),
        ]
        mock_job = MagicMock()
        mock_job.result.return_value = rows
        mock_client.return_value.query.return_value = mock_job

        import app.services.portfolio_service as ps
        original = ps._TABLE_NAMES
        ps._TABLE_NAMES = {"broad_market": "final_network"}
        try:
            result = get_network_data()
        finally:
            ps._TABLE_NAMES = original

        aapl_node = next(n for n in result.nodes if n.id == "AAPL")
        assert aapl_node.out_degree == 2


class TestConstants:
    def test_all_time_ranges_covered_in_range_days(self):
        for tr in TimeRange:
            assert tr in RANGE_DAYS, f"{tr} missing from RANGE_DAYS"

    def test_range_days_positive(self):
        for tr, days in RANGE_DAYS.items():
            assert days > 0, f"{tr} has non-positive days: {days}"

    def test_index_meta_keys(self):
        assert "SPX"  in INDEX_META
        assert "IXIC" in INDEX_META
        assert "DJI"  in INDEX_META