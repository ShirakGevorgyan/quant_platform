"""Bounded deterministic replay input (Milestone 7, Section 32). Loads a
small JSONL source file of normalized `MarketEvent` records for acceptance
testing -- one JSON object per line, each carrying the same `"kind"`
discriminator `events.market_event_from_json_dict` already dispatches on.

Section 32 explicitly permits validating and preparing a BOUNDED replay
source before a session begins (unlike a live/forward stream, which
`events.py`'s own docstring reserves for a not-yet-built `market_data.py`
-- no live/forward event source exists in this milestone; Section 32
explicitly excludes building a market-data downloader or MT5 ingestion).
`load_replay_events` therefore does the full cross-event validation
(chronological order, sequence strictly increasing, no duplicate
event_id/sequence, single instrument unless explicitly allowed, must end
with exactly one `EndOfStreamEvent`) up front, once, and returns an
immutable tuple -- never a partially-validated stream.

`compute_replay_source_identity` gives the CLI/acceptance workflow a
deterministic content hash of the loaded sequence, so two runs against the
same file (or two byte-identical files) are provably using the same
input without re-hashing the raw file bytes (which would also change on
harmless whitespace/line-ending differences the event model itself
normalizes away)."""

from __future__ import annotations

import json
from pathlib import Path

from quant_platform.core.exceptions import MarketEventError, MarketEventOrderError
from quant_platform.paper_trading.events import (
    EndOfStreamEvent,
    MarketEvent,
    market_event_from_json_dict,
    market_event_id,
    market_event_sequence,
    market_event_time,
)
from quant_platform.paper_trading.identity import compute_content_id

REPLAY_SOURCE_IDENTITY_KIND = "paper_trading_replay_source"
MAXIMUM_REPLAY_EVENT_COUNT = 1_000_000
"""Section 37: an operational resource cap, entirely separate from any
financial risk limit -- a replay source larger than this is refused before
ever being handed to the runner, rather than silently consuming unbounded
memory."""


def validate_replay_sequence(events: tuple[MarketEvent, ...], *, allow_mixed_instruments: bool = False, require_contiguous_sequence: bool = False) -> None:
    """Validates cross-event properties of an already-loaded, already
    per-event-validated (each `MarketEvent.__post_init__` already ran)
    sequence. Raises `MarketEventOrderError` (a `MarketEventError`
    subclass reserved exactly for this per its own docstring) on the
    first violation found, in file order, so the error message points at
    the first offending event rather than an aggregate summary."""
    if not events:
        return
    if len(events) > MAXIMUM_REPLAY_EVENT_COUNT:
        raise MarketEventOrderError(f"Replay source has {len(events)} events, exceeding the maximum bounded replay size of {MAXIMUM_REPLAY_EVENT_COUNT}")

    seen_event_ids: set[str] = set()
    seen_sequences: set[int] = set()
    instruments: set[str] = set()
    previous_sequence: int | None = None
    previous_time = None

    for index, event in enumerate(events):
        event_id = market_event_id(event)
        if event_id in seen_event_ids:
            raise MarketEventOrderError(f"Replay source contains a duplicate event_id at position {index}: {event_id!r}")
        seen_event_ids.add(event_id)

        sequence = market_event_sequence(event)
        if sequence in seen_sequences:
            raise MarketEventOrderError(f"Replay source contains a duplicate sequence number at position {index}: {sequence}")
        seen_sequences.add(sequence)
        if previous_sequence is not None:
            if sequence <= previous_sequence:
                raise MarketEventOrderError(f"Replay source sequence numbers are not strictly increasing at position {index}: {previous_sequence} -> {sequence}")
            if require_contiguous_sequence and sequence != previous_sequence + 1:
                raise MarketEventOrderError(f"Replay source sequence numbers are not contiguous at position {index}: {previous_sequence} -> {sequence}")
        previous_sequence = sequence

        event_time = market_event_time(event)
        if previous_time is not None and event_time < previous_time:
            raise MarketEventOrderError(f"Replay source is not in chronological order at position {index}: {previous_time} -> {event_time}")
        previous_time = event_time

        instruments.add(event.instrument)
        if not allow_mixed_instruments and len(instruments) > 1:
            raise MarketEventOrderError(f"Replay source mixes multiple instruments {sorted(instruments)!r} at position {index} -- not explicitly supported (allow_mixed_instruments=False)")

    end_of_stream_positions = [i for i, e in enumerate(events) if isinstance(e, EndOfStreamEvent)]
    if not end_of_stream_positions:
        raise MarketEventOrderError("Replay source does not end with an EndOfStreamEvent -- a bounded replay session can never reach COMPLETED without one")
    if end_of_stream_positions != [len(events) - 1]:
        raise MarketEventOrderError(f"Replay source must contain exactly one EndOfStreamEvent, as its LAST event; found at position(s) {end_of_stream_positions} of {len(events)}")


def load_replay_events(path: Path, *, allow_mixed_instruments: bool = False, require_contiguous_sequence: bool = False) -> tuple[MarketEvent, ...]:
    """Reads a JSONL replay source file, decodes each non-blank line as a
    normalized `MarketEvent`, then validates the FULL loaded sequence via
    `validate_replay_sequence` before returning it. Any decode/schema
    failure raises `MarketEventError` naming the offending line number --
    "reject corrupted files" (Section 32) never partially loads a source."""
    if not path.is_file():
        raise MarketEventError(f"Replay source file not found: {path}")

    events: list[MarketEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MarketEventError(f"Replay source file {path} line {line_number}: corrupted JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketEventError(f"Replay source file {path} line {line_number}: expected a JSON object, got {type(raw).__name__}")
            try:
                event = market_event_from_json_dict(raw)
            except (MarketEventError, KeyError, ValueError, TypeError) as exc:
                raise MarketEventError(f"Replay source file {path} line {line_number}: {exc}") from exc
            events.append(event)

    sequence = tuple(events)
    validate_replay_sequence(sequence, allow_mixed_instruments=allow_mixed_instruments, require_contiguous_sequence=require_contiguous_sequence)
    return sequence


def compute_replay_source_identity(events: tuple[MarketEvent, ...]) -> str:
    """Deterministic content identity of a loaded, validated replay
    sequence -- the ordered list of each event's own `to_identity_payload()`,
    content-addressed the same way every other identity in this package is
    (`identity.compute_content_id`)."""
    payload: dict[str, object] = {"events": [e.to_identity_payload() for e in events]}
    return compute_content_id(REPLAY_SOURCE_IDENTITY_KIND, payload)


def write_replay_events(path: Path, events: tuple[MarketEvent, ...]) -> None:
    """The inverse of `load_replay_events` -- writes one JSON object per
    line in the given order (durable round-trip order, never sorted;
    see `specs.py`'s own durable-order-vs-identity-canonicalization rule).
    Used by fixture/acceptance-test setup, not by the runner itself."""
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_json_dict(), sort_keys=False))
            handle.write("\n")


__all__ = [
    "MAXIMUM_REPLAY_EVENT_COUNT",
    "REPLAY_SOURCE_IDENTITY_KIND",
    "compute_replay_source_identity",
    "load_replay_events",
    "validate_replay_sequence",
    "write_replay_events",
]
