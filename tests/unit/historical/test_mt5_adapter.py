"""Tests for `historical.mt5_adapter.MT5HistoricalSource`, exercised
entirely against `historical.mt5_testing.FakeMt5Client` -- no real
`MetaTrader5` package or live terminal is ever required, which is itself
one of the properties under test (see `TestMissingDependency`).
"""

from __future__ import annotations

import sys
from datetime import timedelta

import pandas as pd
import pytest

from quant_platform.core.exceptions import MissingDependencyError, SourceError
from quant_platform.core.types import Timeframe
from quant_platform.historical.mt5_adapter import MT5AdapterConfig, MT5HistoricalSource
from quant_platform.historical.mt5_testing import FakeMt5Client
from quant_platform.historical.source import SourceRequest
from quant_platform.historical.timezones import FixedOffsetTimezone

SERVER_TZ = FixedOffsetTimezone(timedelta(hours=2), name="EET")


def _config(**overrides) -> MT5AdapterConfig:
    base = {"broker": "TestBroker", "source_symbol": "XAUUSDm", "server_timezone": SERVER_TZ}
    base.update(overrides)
    return MT5AdapterConfig(**base)


def _request(**overrides) -> SourceRequest:
    base = {
        "symbol": "XAUUSDm", "timeframe": Timeframe.M1,
        "start": pd.Timestamp("2024-01-02", tz="UTC"), "end": pd.Timestamp("2024-01-03", tz="UTC"),
    }
    base.update(overrides)
    return SourceRequest(**base)


class TestConfigValidation:
    def test_requires_broker(self) -> None:
        with pytest.raises(ValueError, match="broker identity"):
            _config(broker="")

    def test_requires_source_symbol(self) -> None:
        with pytest.raises(ValueError, match="source_symbol"):
            _config(source_symbol="")

    def test_password_excluded_from_repr(self) -> None:
        cfg = _config(password="super-secret")
        assert "super-secret" not in repr(cfg)


class TestMissingDependency:
    """Simulates the package being absent via `sys.modules[name] = None`
    (the standard idiom for forcing `ImportError` on a bare `import`) rather
    than relying on the ambient test environment actually lacking the real
    `MetaTrader5` package -- that package may well be pip-installed here
    (e.g. as a project dependency for other tooling) even though no MT5
    terminal is running, so the *un-injected-client* failure mode must be
    exercised deterministically rather than by environmental accident.
    """

    def test_connect_without_injected_client_and_without_mt5_installed_raises(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "MetaTrader5", None)
        source = MT5HistoricalSource(_config())
        with pytest.raises(MissingDependencyError, match="MetaTrader5"):
            source.connect()

    def test_never_reports_connected_after_missing_dependency_failure(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "MetaTrader5", None)
        source = MT5HistoricalSource(_config())
        with pytest.raises(MissingDependencyError):
            source.connect()
        assert source.is_connected() is False


class TestConnectionLifecycle:
    def test_not_connected_before_connect(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        assert source.is_connected() is False

    def test_connected_after_connect(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        source.connect()
        assert source.is_connected() is True

    def test_initialize_failure_raises_source_error_and_stays_disconnected(self) -> None:
        fake = FakeMt5Client(fail_initialize=True)
        source = MT5HistoricalSource(_config(), client=fake)
        with pytest.raises(SourceError, match="initialize"):
            source.connect()
        assert source.is_connected() is False

    def test_disconnect_calls_shutdown_when_this_adapter_owns_the_client(self) -> None:
        fake = FakeMt5Client()
        source = MT5HistoricalSource(_config(), client=fake, owns_client=True)
        source.connect()
        source.disconnect()
        assert fake.shutdown_call_count == 1
        assert source.is_connected() is False

    def test_disconnect_does_not_call_shutdown_on_an_externally_owned_client(self) -> None:
        fake = FakeMt5Client()
        source = MT5HistoricalSource(_config(), client=fake, owns_client=False)
        source.connect()
        source.disconnect()
        assert fake.shutdown_call_count == 0
        # Ownership only governs whether shutdown() is called on the shared
        # client -- this adapter's own view of its connection state still
        # updates so it does not attempt fetch() after disconnecting.
        assert source.is_connected() is False

    def test_fetch_before_connect_raises(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        with pytest.raises(SourceError, match="before a successful connect"):
            source.fetch(_request())


class TestFetch:
    def test_returns_canonical_schema_with_utc_open_time(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        source.connect()
        batch = source.fetch(_request())
        assert str(batch.data["open_time"].dt.tz) == "UTC"
        assert len(batch.data) == 1_440  # 24h of M1 bars, hand-computed

    def test_exclusive_end_boundary_never_includes_a_bar_at_exactly_end(self) -> None:
        # Regression test: MT5's copy_rates_range is inclusive of date_to
        # on the wire, but SourceRequest.end is exclusive. Found via manual
        # boundary testing during implementation -- the adapter must
        # subtract before handing the boundary to the fake/real client.
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        source.connect()
        request = _request()
        batch = source.fetch(request)
        assert batch.data["open_time"].max() < request.end
        assert batch.data["open_time"].max() == request.end - pd.Timedelta(minutes=1)

    def test_invalid_symbol_raises_source_error(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client(fail_symbol_select=True))
        source.connect()
        with pytest.raises(SourceError, match="invalid or unavailable"):
            source.fetch(_request())

    def test_copy_rates_range_failure_raises_source_error(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client(return_none_with_error=True))
        source.connect()
        with pytest.raises(SourceError, match="copy_rates_range failed"):
            source.fetch(_request())

    def test_malformed_response_raises_source_error(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client(return_malformed=True))
        source.connect()
        with pytest.raises(SourceError, match="malformed response"):
            source.fetch(_request())

    def test_genuinely_empty_range_returns_empty_batch_not_an_error(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        source.connect()
        # A window starting exactly on an M1 bar boundary would legitimately
        # include that one bar (open_time falls in [start, end)), so this
        # window is placed entirely between two bar boundaries instead, to
        # prove the genuinely-no-data case rather than an off-by-one.
        request = _request(
            start=pd.Timestamp("2024-01-02T00:00:00.500", tz="UTC"),
            end=pd.Timestamp("2024-01-02T00:00:00.900", tz="UTC"),
        )
        batch = source.fetch(request)
        assert len(batch.data) == 0
        assert list(batch.data.columns) == [
            "open_time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread",
        ]

    def test_server_timezone_offset_is_correctly_removed_in_utc_output(self) -> None:
        # A UTC+2 server whose fake generator seeds bars starting at the
        # requested (local-converted) `date_from` must yield a first
        # open_time equal to the ORIGINAL UTC request.start once converted
        # back -- proving the round-trip local<->UTC conversion in `fetch`
        # is lossless and directionally correct, not just "some timezone".
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        source.connect()
        request = _request()
        batch = source.fetch(request)
        assert batch.data["open_time"].iloc[0] == request.start


class TestFetchAllPaginationThroughAdapter:
    def test_pages_yield_monotonic_unique_full_coverage(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        source.connect()
        request = _request(
            start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2024-01-04", tz="UTC"), max_batch_size=500,
        )
        batches = list(source.fetch_all(request))
        all_times = pd.concat([b.data["open_time"] for b in batches]).reset_index(drop=True)
        assert len(all_times) == 3 * 1_440
        assert all_times.is_monotonic_increasing
        assert not all_times.duplicated().any()
        assert all_times.iloc[0] == request.start
        assert all_times.iloc[-1] == request.end - pd.Timedelta(minutes=1)

    def test_single_batch_when_max_batch_size_covers_everything(self) -> None:
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        source.connect()
        request = _request(max_batch_size=10_000)
        batches = list(source.fetch_all(request))
        assert len(batches) == 1
        assert batches[0].is_partial is False

    @pytest.mark.parametrize("max_batch_size", [17, 100, 500, 1_440, 10_000])
    def test_result_is_identical_regardless_of_chunk_size(self, max_batch_size: int) -> None:
        # `generate_fake_rates` is keyed on absolute time (see
        # `test_mt5_testing.py::TestQueryIndependence`), so this only holds
        # because of that fix -- paging through the exact same range with
        # wildly different page sizes must yield bit-identical results.
        request = _request(
            start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2024-01-04", tz="UTC"),
            max_batch_size=max_batch_size,
        )
        source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        source.connect()
        batches = list(source.fetch_all(request))
        combined = pd.concat([b.data for b in batches], ignore_index=True)

        reference_source = MT5HistoricalSource(_config(), client=FakeMt5Client())
        reference_source.connect()
        reference = next(iter(reference_source.fetch_all(_request(
            start=request.start, end=request.end, max_batch_size=10_000_000,
        )))).data

        pd.testing.assert_frame_equal(combined, reference)


class TestNoCredentialsInLogOutput:
    """Section P requires structured logging around every operation in
    this pipeline (connect, fetch, ...) but explicitly forbids ever
    logging a credential. This exercises the adapter's actual logging
    calls (via pytest's `caplog`, which captures real `logging` records,
    not just `print`/stdout) with a password set, across every operation
    that logs anything."""

    def test_connect_and_fetch_never_log_the_password(self, caplog) -> None:
        cfg = _config(login=999999, password="super-secret-password-value", server="SomeServer")
        source = MT5HistoricalSource(cfg, client=FakeMt5Client())
        with caplog.at_level("DEBUG"):
            source.connect()
            source.fetch(_request())
            source.disconnect()
        for record in caplog.records:
            assert "super-secret-password-value" not in record.getMessage()
            assert "super-secret-password-value" not in repr(record.args)
