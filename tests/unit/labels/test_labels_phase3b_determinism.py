"""Milestone 11, Phase 3, Part B: determinism proofs for the 5 concrete
label families -- repeat generation/replay/verification/reconciliation
x10 in-process, plus a subprocess-based `PYTHONHASHSEED` proof, mirroring
`test_labels_determinism.py`'s (Part A) own established convention."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.composite import build_composite_from_definitions, reconcile_composite
from quant_platform.labels.direction import build_direction_specification, generate_direction_labels
from quant_platform.labels.manifest import build_label_manifest
from quant_platform.labels.next_return import build_next_return_specification, generate_next_return_labels
from quant_platform.labels.pricing import PriceBasis
from quant_platform.labels.replay import LabelReplay
from quant_platform.labels.triple_barrier import (
    build_triple_barrier_specification,
    generate_triple_barrier_labels,
)
from quant_platform.labels.verification import LabelVerifier
from quant_platform.labels.volatility import REALIZED_STDDEV_ESTIMATOR_NAME
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

_REPEAT_COUNT = 10

_DETERMINISM_SCRIPT = textwrap.dedent("""
    import json

    import numpy as np
    import pandas as pd

    from quant_platform.labels.builder import LabelBuilder, LabelDefinition
    from quant_platform.labels.pricing import PriceBasis
    from quant_platform.labels.triple_barrier import build_triple_barrier_specification, generate_triple_barrier_labels
    from quant_platform.labels.volatility import REALIZED_STDDEV_ESTIMATOR_NAME

    rng = np.random.default_rng(11)
    n = 120
    open_time = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    returns = rng.normal(0, 0.003, size=n)
    close = 100.0 * np.cumprod(1 + returns)
    open_ = np.roll(close, 1); open_[0] = 100.0
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    source_data = pd.DataFrame({"open_time": open_time, "open": open_, "high": high, "low": low, "close": close})

    spec = build_triple_barrier_specification(
        profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
        volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
    )
    definition = LabelDefinition(specification=spec, generate=generate_triple_barrier_labels)
    bundle = LabelBuilder().build(definition, source_data, source_content_id="src1")

    payload = {"specification": spec.to_json_dict(), "bundle": bundle.to_json_dict()}
    print(json.dumps(payload, sort_keys=True))
""")


def _strip_generated_at(payload: dict) -> dict:
    payload = dict(payload)
    if isinstance(payload.get("bundle"), dict):
        payload["bundle"] = {k: v for k, v in payload["bundle"].items() if k != "generated_at"}
    return payload


class TestSubprocessDeterminism:
    @pytest.mark.parametrize("pythonhashseed", ["0", "1", "random"])
    def test_triple_barrier_payload_identical_across_hashseeds(self, tmp_path, pythonhashseed: str) -> None:
        script_path = tmp_path / "phase3b_determinism_script.py"
        script_path.write_text(_DETERMINISM_SCRIPT, encoding="utf-8")

        baseline = subprocess.run(
            [sys.executable, str(script_path)], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        varied = subprocess.run(
            [sys.executable, str(script_path)], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": pythonhashseed},
        )
        baseline_payload = _strip_generated_at(json.loads(baseline.stdout))
        varied_payload = _strip_generated_at(json.loads(varied.stdout))
        assert baseline_payload == varied_payload


class TestRepeatGeneration:
    def test_ten_repeated_next_return_generations_are_identical(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        results = [generate_next_return_labels(ohlcv_source_data, spec) for _ in range(_REPEAT_COUNT)]
        for r in results[1:]:
            pd.testing.assert_series_equal(r, results[0], check_names=False)

    def test_ten_repeated_triple_barrier_generations_are_identical(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        results = [generate_triple_barrier_labels(ohlcv_source_data, spec) for _ in range(_REPEAT_COUNT)]
        for r in results[1:]:
            pd.testing.assert_series_equal(r, results[0], check_names=False)


class TestRepeatReplay:
    def test_ten_repeated_replay_calls_are_identical(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        definition = LabelDefinition(specification=spec, generate=generate_next_return_labels)
        bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")

        results = []
        for _ in range(_REPEAT_COUNT):
            raw = LabelReplay().replay(definition, ohlcv_source_data, source_content_id="src1", original=bundle).to_json_dict()
            raw.pop("generated_at", None)
            results.append(raw)
        assert all(r == results[0] for r in results)
        assert results[0]["replayed"] is True


class TestRepeatVerification:
    def test_ten_repeated_verify_calls_are_identical(self, ohlcv_source_data: pd.DataFrame) -> None:
        spec = build_direction_specification(
            price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.001, created_from_dataset="ds1", created_from_manifest="m1",
        )
        definition = LabelDefinition(specification=spec, generate=generate_direction_labels)
        bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")
        manifest = build_label_manifest(spec, generation_timestamp=format_utc_timestamp(utc_now()))

        results = []
        for _ in range(_REPEAT_COUNT):
            raw = LabelVerifier().verify(bundle, manifest, definition, ohlcv_source_data, source_content_id="src1").to_json_dict()
            raw.pop("generated_at", None)
            raw["reconciliation"] = {k: v for k, v in raw["reconciliation"].items() if k != "generated_at"}
            results.append(raw)
        assert all(r == results[0] for r in results)
        assert results[0]["verified"] is True


class TestRepeatReconciliation:
    def test_ten_repeated_composite_reconcile_calls_are_identical(self, ohlcv_source_data: pd.DataFrame) -> None:
        return_spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        direction_spec = build_direction_specification(
            price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.001, created_from_dataset="ds1", created_from_manifest="m1",
        )
        definitions = (
            LabelDefinition(specification=return_spec, generate=generate_next_return_labels),
            LabelDefinition(specification=direction_spec, generate=generate_direction_labels),
        )
        composite = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        manifests = tuple(build_label_manifest(d.specification, generation_timestamp=format_utc_timestamp(utc_now())) for d in definitions)

        results = []
        for _ in range(_REPEAT_COUNT):
            result = reconcile_composite(composite, composite, baseline_manifests=manifests, candidate_manifests=manifests)
            results.append(result.reconciled)
        assert all(results)
