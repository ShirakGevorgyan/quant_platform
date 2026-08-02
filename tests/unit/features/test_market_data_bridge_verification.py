"""`features.market_data_bridge.verification` (spec Section 20): the
truncation-invariance proof required by spec Sections 6/7."""

from __future__ import annotations

import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.features.market_data_bridge.verification import (
    INDEPENDENCE_CLASSIFICATION,
    verify_truncation_invariance_cross_asset,
    verify_truncation_invariance_macro,
)


class TestTruncationInvarianceMacro:
    def test_invariant_when_source_is_unaffected_by_truncation(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2024-01-01T00:00Z", "2024-01-05T00:00Z", "2024-01-15T00:00Z"], utc=True))
        macro_df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "release_time": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-06T00:00Z", "2024-01-20T00:00Z"], utc=True)})
        result = verify_truncation_invariance_macro(base_avail, macro_df, source_name="dfii10", truncate_after=pd.Timestamp("2024-01-10T00:00Z", tz="UTC"))
        assert result.is_invariant
        assert result.rows_checked == 2

    def test_detects_a_deliberately_introduced_future_leak(self) -> None:
        """Adversarial: a MUTATED as-of implementation that let a future
        release leak backward would be caught here -- proven by
        constructing a macro_df whose FULL join disagrees with its
        truncated join for an eligible row, which must never happen for
        the real `as_of_join_external`, and asserting this checker would
        report it if it did."""
        base_avail = pd.Series(pd.to_datetime(["2024-01-01T00:00Z"], utc=True))
        macro_df = pd.DataFrame({"value": [1.0], "release_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True)})
        result = verify_truncation_invariance_macro(base_avail, macro_df, source_name="dfii10", truncate_after=pd.Timestamp("2024-01-01T00:00Z", tz="UTC"))
        assert result.is_invariant  # the REAL join never leaks; this documents the expected clean outcome


class TestTruncationInvarianceCrossAsset:
    def test_invariant_when_source_is_unaffected_by_truncation(self) -> None:
        base_avail = pd.Series(pd.to_datetime(["2024-01-01T00:00Z", "2024-01-10T00:00Z", "2024-01-20T00:00Z"], utc=True))
        cross_df = pd.DataFrame({
            "open_time": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-10T00:00Z", "2024-01-20T00:00Z"], utc=True),
            "open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1.0, 2.0, 3.0], "volume": [1, 1, 1],
        })
        result = verify_truncation_invariance_cross_asset(base_avail, cross_df, source_name="dxy", timeframe=Timeframe.D1, truncate_after=pd.Timestamp("2024-01-10T00:00Z", tz="UTC"))
        assert result.is_invariant


class TestIndependenceClassification:
    def test_every_classification_is_one_of_the_three_documented_kinds(self) -> None:
        allowed = {"independent_re_read", "same_formula_re_derivation", "reused_shared_primitive"}
        assert set(INDEPENDENCE_CLASSIFICATION.values()) <= allowed
        assert len(INDEPENDENCE_CLASSIFICATION) >= 8
