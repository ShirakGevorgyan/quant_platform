"""Milestone 11, Phase 3, Part B: point-in-time safety. "Labels MUST
NEVER observe future macro releases, future cross-asset values, future
revisions, future timestamps, future bars beyond configured horizon."

Future macro releases / future cross-asset values / future revisions
are addressed structurally: this package never imports anything that
could read them in the first place (verified below via AST inspection
of every module's own import statements, not a grep of prose/docstring
text that happens to mention `features`/`market_data`). Future
timestamps and future-bars-beyond-horizon are addressed functionally,
proven directly against each family's generator."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.direction import build_direction_specification, generate_direction_labels
from quant_platform.labels.forward_volatility import (
    build_forward_volatility_specification,
    generate_forward_volatility_labels,
)
from quant_platform.labels.next_return import build_next_return_specification, generate_next_return_labels
from quant_platform.labels.pricing import PriceBasis
from quant_platform.labels.records import materialize_label_records
from quant_platform.labels.triple_barrier import (
    build_triple_barrier_specification,
    generate_triple_barrier_labels,
)
from quant_platform.labels.volatility import REALIZED_STDDEV_ESTIMATOR_NAME

_FORBIDDEN_IMPORT_PREFIXES = (
    "quant_platform.features", "quant_platform.market_data", "quant_platform.qualification", "quant_platform.feature_discovery",
    "quant_platform.ml.", "quant_platform.paper_trading", "quant_platform.execution_gateway", "quant_platform.portfolio_risk",
)
_ALLOWED_ML_IMPORT = "quant_platform.ml.persistence"


def _imported_module_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


class TestNoDependencyOnMarketDataOrFeaturesOrCrossAsset:
    """This package structurally cannot observe future macro releases,
    future cross-asset values, or future revisions -- it never imports
    anything that reads market_data/features/qualification/
    feature_discovery source data in the first place."""

    def test_no_labels_source_module_imports_a_forbidden_package(self) -> None:
        labels_dir = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "labels"
        assert labels_dir.is_dir()
        offenders = []
        for path in sorted(labels_dir.glob("*.py")):
            for module_name in _imported_module_names(path):
                if module_name == _ALLOWED_ML_IMPORT:
                    continue
                if any(module_name == prefix.rstrip(".") or module_name.startswith(prefix) for prefix in _FORBIDDEN_IMPORT_PREFIXES):
                    offenders.append((path.name, module_name))
        assert offenders == []


class TestEventTimeAvailabilityTimeAreWallClockIndependent:
    def test_identical_regardless_of_when_materialization_runs(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        definition = LabelDefinition(specification=spec, generate=generate_next_return_labels)
        bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")

        first = materialize_label_records(bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        second = materialize_label_records(bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        assert [r.event_time for r in first] == [r.event_time for r in second]
        assert [r.availability_time for r in first] == [r.availability_time for r in second]

    def test_derived_purely_from_open_time_never_from_now(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        definition = LabelDefinition(specification=spec, generate=generate_next_return_labels)
        bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")
        records = materialize_label_records(bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)

        expected_event_time = ohlcv_source_data["open_time"].iloc[0] + Timeframe.M1.duration
        expected_availability_time = expected_event_time + Timeframe.M1.duration * 5
        assert records[0].event_time == expected_event_time.isoformat()
        assert records[0].availability_time == expected_availability_time.isoformat()


class TestNeverReadsBeyondConfiguredHorizon:
    """For each family, corrupting `source_data` strictly AFTER the
    configured horizon must never change a row's already-generated
    label value -- proof that no generator peeks past its own
    horizon."""

    def test_next_return(self, ohlcv_source_data: pd.DataFrame) -> None:
        horizon_bars = 5
        row = 10
        spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=horizon_bars, created_from_dataset="ds1", created_from_manifest="m1")
        baseline = generate_next_return_labels(ohlcv_source_data, spec)

        corrupted = ohlcv_source_data.copy()
        corrupted.loc[row + horizon_bars + 1 :, "close"] = 999999.0
        corrupted_values = generate_next_return_labels(corrupted, spec)
        assert corrupted_values.iloc[row] == pytest.approx(baseline.iloc[row])

    def test_direction(self, ohlcv_source_data: pd.DataFrame) -> None:
        horizon_bars = 5
        row = 10
        spec = build_direction_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=horizon_bars, neutral_threshold=0.001, created_from_dataset="ds1", created_from_manifest="m1")
        baseline = generate_direction_labels(ohlcv_source_data, spec)

        corrupted = ohlcv_source_data.copy()
        corrupted.loc[row + horizon_bars + 1 :, "close"] = 999999.0
        corrupted_values = generate_direction_labels(corrupted, spec)
        assert corrupted_values.iloc[row] == pytest.approx(baseline.iloc[row])

    def test_forward_volatility(self, ohlcv_source_data: pd.DataFrame) -> None:
        horizon_bars = 10
        row = 20
        spec = build_forward_volatility_specification(
            horizon_bars=horizon_bars, volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        baseline = generate_forward_volatility_labels(ohlcv_source_data, spec)

        corrupted = ohlcv_source_data.copy()
        corrupted.loc[row + horizon_bars + 1 :, "close"] = corrupted["close"].iloc[row + horizon_bars] * 5.0
        corrupted_values = generate_forward_volatility_labels(corrupted, spec)
        assert corrupted_values.iloc[row] == pytest.approx(baseline.iloc[row])

    def test_triple_barrier(self, ohlcv_source_data: pd.DataFrame) -> None:
        max_holding_bars = 10
        row = 30
        spec = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=max_holding_bars, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        baseline = generate_triple_barrier_labels(ohlcv_source_data, spec)

        corrupted = ohlcv_source_data.copy()
        beyond = row + max_holding_bars + 1
        corrupted.loc[beyond:, "high"] = corrupted["close"].iloc[row] * 100.0  # a wildly-touched-looking future bar
        corrupted.loc[beyond:, "low"] = corrupted["close"].iloc[row] * 100.0
        corrupted_values = generate_triple_barrier_labels(corrupted, spec)
        assert corrupted_values.iloc[row] == pytest.approx(baseline.iloc[row])

    def test_triple_barrier_volatility_sizing_is_past_only(self, ohlcv_source_data: pd.DataFrame) -> None:
        """The barrier WIDTH at row `t` must depend only on trailing
        (past) volatility -- corrupting data far in the future must not
        change row `t`'s barrier sizing/outcome even indirectly."""
        row = 30
        spec = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=5, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        baseline = generate_triple_barrier_labels(ohlcv_source_data, spec)

        corrupted = ohlcv_source_data.copy()
        corrupted.loc[row + 50 :, "close"] = np.linspace(1.0, 5000.0, len(corrupted) - (row + 50))
        corrupted_values = generate_triple_barrier_labels(corrupted, spec)
        assert corrupted_values.iloc[row] == pytest.approx(baseline.iloc[row])
