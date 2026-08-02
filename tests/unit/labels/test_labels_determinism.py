"""Milestone 11, Phase 3, Part A: determinism proofs. `labels/` is fully
standalone (no `historical`/`features`/`market_data` dependency), so the
subprocess proof needs no synthetic historical pipeline -- just a
specification, a synthetic generator, and source data, exactly as any
other test in this directory uses. Repeat determinism tests (x10,
in-process) mirror `qualification`'s/`feature_discovery`'s own
`test_*_repetition_gates.py`/`test_*_determinism.py` convention."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.diagnostics import compute_label_diagnostics
from quant_platform.labels.manifest import build_label_manifest
from quant_platform.labels.models import LabelFamily, build_label_specification
from quant_platform.labels.reconciliation import LabelReconciliation
from quant_platform.labels.verification import LabelVerifier
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

_DETERMINISM_SCRIPT = textwrap.dedent("""
    import json

    import numpy as np
    import pandas as pd

    from quant_platform.labels.builder import LabelBuilder, LabelDefinition
    from quant_platform.labels.diagnostics import compute_label_diagnostics
    from quant_platform.labels.manifest import build_label_manifest
    from quant_platform.labels.models import LabelFamily, build_label_specification

    def marker_generator(source_data, specification):
        n = len(source_data)
        values = np.arange(n, dtype="float64")
        values[-3:] = np.nan
        return pd.Series(values, index=source_data.index)

    open_time = pd.date_range("2024-01-01", periods=20, freq="1min", tz="UTC")
    close = 100.0 + np.arange(20, dtype="float64")
    source_data = pd.DataFrame({"open_time": open_time, "close": close, "high": close + 0.5, "low": close - 0.5})

    specification = build_label_specification(
        label_family=LabelFamily.NEXT_RETURN, generation_version="v1", price_basis="close", prediction_horizon="5 bars",
        availability_rule="available at reference_time + 5 bars", reference_price="close at event_time",
        event_time_rule="bar close time", generation_rule="structural test fixture -- not a real generation rule",
        created_from_dataset="dataset-0001", created_from_manifest="manifest-0001", parameters={"horizon_bars": 5},
    )
    definition = LabelDefinition(specification=specification, generate=marker_generator)
    bundle = LabelBuilder().build(definition, source_data, source_content_id="source-0001")
    manifest = build_label_manifest(specification, generation_timestamp="2024-01-01T00:00:00+00:00", feature_identity="feat-1", qualification_identity="qual-1")
    diagnostics = compute_label_diagnostics(bundle, manifest)

    payload = {"specification": specification.to_json_dict(), "bundle": bundle.to_json_dict(), "manifest": manifest.to_json_dict(), "diagnostics": diagnostics.to_json_dict()}
    print(json.dumps(payload, sort_keys=True))
""")


def _strip_volatile(payload: dict) -> dict:
    payload = dict(payload)
    if isinstance(payload.get("bundle"), dict):
        payload["bundle"] = {k: v for k, v in payload["bundle"].items() if k != "generated_at"}
    return payload


class TestSubprocessDeterminism:
    @pytest.mark.parametrize("pythonhashseed", ["0", "1", "random"])
    def test_payload_is_identical_across_hashseeds(self, tmp_path, pythonhashseed: str) -> None:
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
    def test_ten_repeated_specification_builds_are_identical(self) -> None:
        ids = set()
        for _ in range(10):
            spec = build_label_specification(
                label_family=LabelFamily.FORWARD_VOLATILITY, generation_version="v1", price_basis="close", prediction_horizon="10 bars",
                availability_rule="rule", reference_price="close", event_time_rule="close time", generation_rule="rule",
                created_from_dataset="dataset-x", created_from_manifest="manifest-x", parameters={"vol_window": 20},
            )
            ids.add(spec.label_specification_id)
        assert len(ids) == 1

    def test_ten_repeated_builds_are_identical(self, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str) -> None:
        content_ids = set()
        for _ in range(10):
            bundle = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
            content_ids.add(bundle.identity.content_id)
        assert len(content_ids) == 1

    def test_ten_repeated_manifest_builds_are_identical(self, specification) -> None:
        checksums = set()
        for _ in range(10):
            manifest = build_label_manifest(specification, generation_timestamp=format_utc_timestamp(utc_now()), feature_identity="f", qualification_identity="q")
            checksums.add(manifest.manifest_checksum)
        assert len(checksums) == 1

    def test_ten_repeated_diagnostics_are_identical(self, bundle, manifest) -> None:
        scores = set()
        for _ in range(10):
            diagnostics = compute_label_diagnostics(bundle, manifest)
            scores.add(diagnostics.overall_score)
        assert len(scores) == 1

    def test_ten_repeated_verify_calls_are_identical(self, bundle, manifest, definition, source_data, source_content_id) -> None:
        results = []
        for _ in range(10):
            raw = LabelVerifier().verify(bundle, manifest, definition, source_data, source_content_id=source_content_id).to_json_dict()
            raw.pop("generated_at")
            raw["reconciliation"] = {k: v for k, v in raw["reconciliation"].items() if k != "generated_at"}
            results.append(raw)
        assert all(r == results[0] for r in results)
        assert results[0]["verified"] is True

    def test_ten_repeated_reconcile_calls_are_identical(self, bundle, manifest) -> None:
        results = []
        for _ in range(10):
            raw = LabelReconciliation().reconcile(bundle, bundle, baseline_manifest=manifest, candidate_manifest=manifest).to_json_dict()
            raw.pop("generated_at")
            results.append(raw)
        assert all(r == results[0] for r in results)
        assert results[0]["reconciled"] is True
