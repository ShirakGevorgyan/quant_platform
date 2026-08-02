from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.labels.identity import LabelIdentity, compute_label_identity


class TestComputeLabelIdentity:
    def test_deterministic_across_calls(self) -> None:
        values = pd.Series([1.0, 2.0, np.nan, 4.0])
        first = compute_label_identity("spec-1", values, source_content_id="source-1")
        second = compute_label_identity("spec-1", values, source_content_id="source-1")
        assert first.content_id == second.content_id
        assert first == second

    def test_different_values_different_content_id(self) -> None:
        a = compute_label_identity("spec-1", pd.Series([1.0, 2.0]), source_content_id="source-1")
        b = compute_label_identity("spec-1", pd.Series([1.0, 3.0]), source_content_id="source-1")
        assert a.content_id != b.content_id

    def test_row_order_matters(self) -> None:
        a = compute_label_identity("spec-1", pd.Series([1.0, 2.0]), source_content_id="source-1")
        b = compute_label_identity("spec-1", pd.Series([2.0, 1.0]), source_content_id="source-1")
        assert a.content_id != b.content_id

    def test_different_specification_id_different_content_id(self) -> None:
        values = pd.Series([1.0, 2.0])
        a = compute_label_identity("spec-1", values, source_content_id="source-1")
        b = compute_label_identity("spec-2", values, source_content_id="source-1")
        assert a.content_id != b.content_id

    def test_different_source_content_id_different_content_id(self) -> None:
        values = pd.Series([1.0, 2.0])
        a = compute_label_identity("spec-1", values, source_content_id="source-1")
        b = compute_label_identity("spec-1", values, source_content_id="source-2")
        assert a.content_id != b.content_id

    def test_nan_is_a_legitimate_value_not_rejected(self) -> None:
        identity = compute_label_identity("spec-1", pd.Series([1.0, np.nan]), source_content_id="source-1")
        assert identity.row_count == 2

    def test_all_nan_still_produces_an_identity(self) -> None:
        identity = compute_label_identity("spec-1", pd.Series([np.nan, np.nan]), source_content_id="source-1")
        assert identity.row_count == 2

    def test_row_count_matches_length(self) -> None:
        identity = compute_label_identity("spec-1", pd.Series([1.0, 2.0, 3.0]), source_content_id="source-1")
        assert identity.row_count == 3


class TestJsonRoundTrip:
    def test_round_trip(self) -> None:
        identity = compute_label_identity("spec-1", pd.Series([1.0, np.nan, 3.0]), source_content_id="source-1")
        restored = LabelIdentity.from_json_dict(identity.to_json_dict())
        assert restored == identity
