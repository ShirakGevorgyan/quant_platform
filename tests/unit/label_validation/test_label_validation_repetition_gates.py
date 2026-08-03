"""Milestone 11, Phase 3, Part C: repetition and determinism gates.
Mirrors `labels`'/`feature_discovery`'s/`qualification`'s own
`test_*_determinism.py` / `test_*_repetition_gates.py` convention
exactly: a subprocess, cross-`PYTHONHASHSEED` proof for the full
qualification pipeline, plus x10 in-process repeats for qualification,
verification, reconciliation, and replay."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from quant_platform.label_validation.engine import LabelQualificationEngine
from quant_platform.label_validation.reconciliation import LabelValidationReconciliation
from quant_platform.label_validation.replay import LabelValidationReplay
from quant_platform.label_validation.verification import LabelValidationVerifier
from quant_platform.labels.builder import LabelBundle, LabelDefinition
from quant_platform.labels.manifest import LabelManifest

_DETERMINISM_SCRIPT = textwrap.dedent("""
    import json

    import numpy as np
    import pandas as pd

    from quant_platform.label_validation.engine import LabelQualificationEngine
    from quant_platform.labels.builder import LabelBuilder, LabelDefinition
    from quant_platform.labels.manifest import build_label_manifest
    from quant_platform.labels.next_return import build_next_return_specification, generate_next_return_labels
    from quant_platform.labels.pricing import PriceBasis
    from quant_platform.ml.persistence import format_utc_timestamp, utc_now

    rng = np.random.default_rng(5)
    row_count = 300
    open_time = pd.date_range("2024-01-01", periods=row_count, freq="1min", tz="UTC")
    returns = rng.normal(0, 0.003, size=row_count)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.001, size=row_count)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.001, size=row_count)))
    source_data = pd.DataFrame({"open_time": open_time, "open": open_, "high": high, "low": low, "close": close})

    specification = build_next_return_specification(
        price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1",
    )
    definition = LabelDefinition(specification=specification, generate=generate_next_return_labels)
    bundle = LabelBuilder().build(definition, source_data, source_content_id="test-source-content-id-0001")
    manifest = build_label_manifest(specification, generation_timestamp="2024-01-01T00:00:00+00:00")
    report = LabelQualificationEngine().qualify(bundle, manifest)

    payload = {"bundle": bundle.to_json_dict(), "manifest": manifest.to_json_dict(), "report": report.to_json_dict()}
    print(json.dumps(payload, sort_keys=True))
""")


def _strip_volatile(payload: dict) -> dict:
    payload = dict(payload)
    if isinstance(payload.get("bundle"), dict):
        payload["bundle"] = {k: v for k, v in payload["bundle"].items() if k != "generated_at"}
    if isinstance(payload.get("report"), dict):
        payload["report"] = {k: v for k, v in payload["report"].items() if k != "qualified_at"}
    return payload


class TestSubprocessDeterminism:
    @pytest.mark.parametrize("pythonhashseed", ["0", "1", "random"])
    def test_qualification_payload_is_identical_across_hashseeds(self, tmp_path, pythonhashseed: str) -> None:
        script_path = tmp_path / "determinism_script.py"
        script_path.write_text(_DETERMINISM_SCRIPT, encoding="utf-8")

        baseline = subprocess.run(
            [sys.executable, str(script_path)], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        varied = subprocess.run(
            [sys.executable, str(script_path)], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": pythonhashseed},
        )
        baseline_payload = _strip_volatile(json.loads(baseline.stdout))
        varied_payload = _strip_volatile(json.loads(varied.stdout))
        assert baseline_payload == varied_payload


class TestRepeatInProcess:
    def test_ten_repeated_qualify_calls_are_identical(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        results = []
        for _ in range(10):
            raw = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest).to_json_dict()
            raw.pop("qualified_at")
            results.append(raw)
        assert all(r == results[0] for r in results)

    def test_ten_repeated_verify_calls_are_identical(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        results = []
        for _ in range(10):
            raw = LabelValidationVerifier().verify(report, next_return_bundle, next_return_manifest).to_json_dict()
            raw.pop("generated_at")
            raw["reconciliation"] = {k: v for k, v in raw["reconciliation"].items() if k != "generated_at"}
            results.append(raw)
        assert all(r == results[0] for r in results)
        assert results[0]["verified"] is True

    def test_ten_repeated_reconcile_calls_are_identical(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        results = []
        for _ in range(10):
            raw = LabelValidationReconciliation().reconcile(report, report).to_json_dict()
            raw.pop("generated_at")
            results.append(raw)
        assert all(r == results[0] for r in results)
        assert results[0]["reconciled"] is True

    def test_ten_repeated_replay_calls_are_identical(
        self, next_return_definition: LabelDefinition, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest,
        ohlcv_source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        original_report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        results = []
        for _ in range(10):
            raw = LabelValidationReplay().replay_and_requalify(
                next_return_definition, ohlcv_source_data, source_content_id=source_content_id, manifest=next_return_manifest, original_report=original_report,
            ).to_json_dict()
            raw.pop("generated_at")
            results.append(raw)
        assert all(r == results[0] for r in results)
        assert results[0]["qualification_identical"] is True
