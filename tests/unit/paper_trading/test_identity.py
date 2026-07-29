"""Milestone 7: `paper_trading.identity.compute_content_id` is the shared
building block every domain object's deterministic id goes through --
these tests pin its namespacing and payload-sensitivity behavior directly,
independent of any one caller (`specs.py`, and later `events.py`/
`orders.py`/`fills.py`)."""

from __future__ import annotations

import pytest

from quant_platform.paper_trading.identity import compute_content_id, is_valid_sha256_hex


class TestComputeContentId:
    def test_identical_kind_and_payload_produce_identical_id(self) -> None:
        assert compute_content_id("order", {"a": 1}) == compute_content_id("order", {"a": 1})

    def test_result_is_a_valid_sha256_hex_digest(self) -> None:
        assert is_valid_sha256_hex(compute_content_id("order", {"a": 1}))

    def test_different_kind_same_payload_produces_different_id(self) -> None:
        """The whole reason `kind` exists: an `Order` and a `Fill` that
        happen to share every other field must never collide."""
        assert compute_content_id("order", {"a": 1}) != compute_content_id("fill", {"a": 1})

    def test_different_payload_same_kind_produces_different_id(self) -> None:
        assert compute_content_id("order", {"a": 1}) != compute_content_id("order", {"a": 2})

    def test_empty_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            compute_content_id("", {"a": 1})
