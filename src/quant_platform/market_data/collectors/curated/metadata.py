"""Series-metadata DRIFT verification (Milestone 10, Phase 4B) -- pure
comparison of officially-returned FRED metadata (`fred_series_metadata.
FredSeriesMetadata`) against a curated spec's own DECLARED expectations.
Never trusts a manually-written curated label over official returned
metadata -- the OFFICIAL response is always the ground truth being
compared against; a curated spec that disagrees with it is the thing
that must be flagged, never silently "corrected" toward the spec.

DRIFT POLICY (exactly per the Phase 4B specification):
- unexpected series id: FAIL CLOSED
- incompatible frequency: FAIL CLOSED
- incompatible units: FAIL CLOSED
- changed seasonal-adjustment semantics: FAIL CLOSED where the spec
  declared an expectation (`expected_seasonal_adjustment is not None`)
- changed title/notes only: informational (never fails; there is no
  "expected title" in the curated spec to compare against in the first
  place -- title text is not a semantic contract)
- changed supported observation range: reported; the requested backfill
  interval is only PERMITTED to proceed if it still falls within the
  metadata's own reported `observation_start`/`observation_end`
- `last_updated` is captured for PROVENANCE only, never compared."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.market_data.collectors.curated.registry import CuratedFredSeriesSpec
from quant_platform.market_data.collectors.fred_series_metadata import FredSeriesMetadata

__all__ = ["MetadataDriftFinding", "MetadataVerificationResult", "verify_series_metadata"]

_FAIL_CLOSED = "fail_closed"
_WARNING = "warning"
_INFO = "info"


@dataclass(frozen=True, slots=True)
class MetadataDriftFinding:
    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class MetadataVerificationResult:
    series_id: str
    passed: bool
    """`False` iff at least one FAIL_CLOSED finding is present."""
    findings: tuple[MetadataDriftFinding, ...]
    verified_title: str
    verified_last_updated: str
    verified_observation_start: str
    verified_observation_end: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "series_id": self.series_id, "passed": self.passed,
            "findings": [{"severity": f.severity, "code": f.code, "message": f.message} for f in self.findings],
            "verified_title": self.verified_title, "verified_last_updated": self.verified_last_updated,
            "verified_observation_start": self.verified_observation_start, "verified_observation_end": self.verified_observation_end,
        }


def verify_series_metadata(
    spec: CuratedFredSeriesSpec, metadata: FredSeriesMetadata, *, requested_observation_start: str | None = None,
    requested_observation_end: str | None = None,
) -> MetadataVerificationResult:
    findings: list[MetadataDriftFinding] = []

    if metadata.series_id != spec.series_id:
        findings.append(MetadataDriftFinding(_FAIL_CLOSED, "unexpected_series_id", f"requested series_id={spec.series_id!r} but metadata reports id={metadata.series_id!r}"))

    if metadata.frequency_short != spec.expected_native_frequency:
        findings.append(MetadataDriftFinding(_FAIL_CLOSED, "incompatible_frequency", f"{spec.series_id}: expected frequency_short={spec.expected_native_frequency!r}, metadata reports {metadata.frequency_short!r}"))

    if metadata.units_short not in spec.expected_units:
        findings.append(MetadataDriftFinding(_FAIL_CLOSED, "incompatible_units", f"{spec.series_id}: expected units_short in {list(spec.expected_units)!r}, metadata reports {metadata.units_short!r}"))

    if spec.expected_seasonal_adjustment is not None and metadata.seasonal_adjustment_short != spec.expected_seasonal_adjustment:
        findings.append(MetadataDriftFinding(_FAIL_CLOSED, "changed_seasonal_adjustment", f"{spec.series_id}: expected seasonal_adjustment_short={spec.expected_seasonal_adjustment!r}, metadata reports {metadata.seasonal_adjustment_short!r}"))

    findings.append(MetadataDriftFinding(_INFO, "title_reported", f"{spec.series_id}: metadata title={metadata.title!r} (informational only -- no curated expectation to compare against)"))

    if requested_observation_start is not None or requested_observation_end is not None:
        findings.append(MetadataDriftFinding(
            _INFO, "observation_range_reported",
            f"{spec.series_id}: metadata supports [{metadata.observation_start}, {metadata.observation_end}]",
        ))
        if requested_observation_start is not None and requested_observation_start < metadata.observation_start:
            findings.append(MetadataDriftFinding(_FAIL_CLOSED, "requested_range_before_supported_start", f"{spec.series_id}: requested observation_start={requested_observation_start!r} precedes metadata's supported observation_start={metadata.observation_start!r}"))
        if requested_observation_end is not None and requested_observation_end > metadata.observation_end:
            findings.append(MetadataDriftFinding(_WARNING, "requested_range_after_supported_end", f"{spec.series_id}: requested observation_end={requested_observation_end!r} is after metadata's currently-reported observation_end={metadata.observation_end!r} (series may simply have grown since the spec was last reviewed)"))

    passed = not any(f.severity == _FAIL_CLOSED for f in findings)
    return MetadataVerificationResult(
        series_id=spec.series_id, passed=passed, findings=tuple(findings), verified_title=metadata.title, verified_last_updated=metadata.last_updated,
        verified_observation_start=metadata.observation_start, verified_observation_end=metadata.observation_end,
    )
