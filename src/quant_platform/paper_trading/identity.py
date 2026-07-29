"""Shared content-addressed identity computation for every domain object
in `quant_platform.paper_trading` that needs a deterministic id --
`paper_session_spec_id` (`specs.py`), and later `event_id`/`decision_id`/
`order_id`/`fill_id`/snapshot ids (`events.py`/`strategy.py`/`orders.py`/
`fills.py`/`portfolio.py`). One shared building block means every one of
those ids is computed the same way -- a namespaced sha256 of the object's
own canonical JSON payload -- rather than each module re-deriving its own
hashing convention.

The `kind` namespace tag exists so that an `Order` and a `Fill` that
happen to share every other field value (same session, same instrument,
same quantity, same timestamp) can never collide on identity -- their
envelopes differ in `kind` even when `payload` does not."""

from __future__ import annotations

from quant_platform.ml.fingerprints import fingerprint_json, is_valid_sha256_hex

IDENTITY_SCHEMA_VERSION = 1


def compute_content_id(kind: str, payload: dict[str, object]) -> str:
    """Deterministic sha256 hex digest of `payload`, namespaced by `kind`
    (e.g. `"paper_session_spec"`, `"market_event"`, `"strategy_decision"`,
    `"order"`, `"fill"`). Two calls with the same `kind` and an
    identical `payload` always produce the same id; changing either
    changes it."""
    if not kind:
        raise ValueError("compute_content_id: kind must not be empty")
    envelope: dict[str, object] = {"identity_schema_version": IDENTITY_SCHEMA_VERSION, "kind": kind, "payload": payload}
    return fingerprint_json(envelope)


__all__ = ["IDENTITY_SCHEMA_VERSION", "compute_content_id", "is_valid_sha256_hex"]
