"""`QualificationDiagnostics` (Milestone 11, Phase 1). Part 1 built the
lean summary (`SplitDiagnostics`: per-split row counts/missingness/span
plus a flat per-dimension score map). Part 2 adds the DEEP, evidence-
based layer the spec calls for: one `Evidence` record per finding,
grouped into 6 sections -- `structural_evidence`, `temporal_evidence`,
`statistical_evidence`, `coverage_evidence`, `stability_evidence`,
`safety_evidence` -- so a caller can see WHY a dimension scored the way
it did, not just the score itself. This is purely additive: every Part 1
field/behavior (`SplitDiagnostics`, the flat `dimension_scores` map,
`compute_diagnostics`'s existing positional signature) is unchanged;
Part 2 only adds new fields (with `()` defaults) and one new optional
keyword parameter (`required_feature_names`) to `compute_diagnostics`.

DELIBERATE OVERLAP, DOCUMENTED RATHER THAN HIDDEN: `temporal_evidence`'s
"future visibility" and `safety_evidence`'s "leakage evidence" both
surface `VerificationFacts.leakage_messages` (the SAME underlying
`features.validation.validate_research_dataset` TARGET_LEAKAGE_SUSPECTED
findings already computed once by `QualificationVerifier`) -- they are
two different QUESTIONS asked of one fact ("is temporal ordering
violated" vs. "is the dataset safe to train on"), not two competing
detectors. Likewise `statistical_evidence`'s zero/near-zero-variance
items and `safety_evidence`'s mutable-alias items both come from ONE
`features.drift.compare_splits(df, df, ...)` self-comparison call per
split (constant/near-constant/highly-correlated-pair detection is a
side effect of that single call, never reimplemented here).

MACRO/CROSS-ASSET SCOPE: availability/coverage evidence for macro and
cross-asset sources is read exclusively from `ResearchDatasetManifest.
market_data_lineage` (the payload `features.market_data_bridge.lineage.
build_market_data_lineage` already persisted at build time) -- this
module never re-reads raw macro/cross-asset source data and never
recomputes a fresh `features.market_data_bridge.staleness.
StalenessFinding` (that would require re-joining raw external series,
crossing the "no second FeatureEngine/builder" boundary this package
has held since Part 1). Per-row staleness re-verification is therefore
OUT OF SCOPE post-hoc; `market_data_lineage.coverage_decision`'s own
`status`/`coverage_fraction` per source (frozen at build time) is the
closest available signal, and is reported as such."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import QualificationVerificationError, ResearchDatasetError
from quant_platform.features.drift import compare_splits
from quant_platform.features.manifests import ResearchDatasetManifest, ResearchDatasetStore
from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.qualification.evidence import Evidence, affected_split, make_evidence
from quant_platform.qualification.models import DatasetQualificationReport, QualificationDimensionKind
from quant_platform.qualification.verifier import QualificationVerifier, VerificationFacts

__all__ = [
    "QUALIFICATION_DIAGNOSTICS_SCHEMA_VERSION",
    "QualificationDiagnostics",
    "SplitDiagnostics",
    "compute_diagnostics",
]

QUALIFICATION_DIAGNOSTICS_SCHEMA_VERSION = 2
_NON_FEATURE_COLUMNS = frozenset({"open_time", "label", "label_valid"})
_CORRELATION_THRESHOLD = 0.999
_ROLLING_WINDOW_MIN = 10
_ABNORMAL_SKEW_THRESHOLD = 2.0
_ABNORMAL_KURTOSIS_THRESHOLD = 7.0


@dataclass(frozen=True, slots=True)
class SplitDiagnostics:
    split_name: str
    row_count: int
    feature_null_fractions: dict[str, float]
    open_time_min: str | None
    open_time_max: str | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name, "row_count": self.row_count, "feature_null_fractions": self.feature_null_fractions,
            "open_time_min": self.open_time_min, "open_time_max": self.open_time_max,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SplitDiagnostics:
        return cls(
            split_name=str(raw["split_name"]), row_count=int(str(raw["row_count"])),
            feature_null_fractions={str(k): float(str(v)) for k, v in as_json_dict(raw.get("feature_null_fractions") or {}, field_name="feature_null_fractions").items()},
            open_time_min=(None if raw.get("open_time_min") is None else str(raw["open_time_min"])),
            open_time_max=(None if raw.get("open_time_max") is None else str(raw["open_time_max"])),
        )


@dataclass(frozen=True, slots=True)
class QualificationDiagnostics:
    schema_version: int
    dataset_id: str
    version: str
    content_id: str
    split_diagnostics: tuple[SplitDiagnostics, ...]
    dimension_scores: dict[str, float]
    overall_score: float
    decision: str
    generated_at: str
    structural_evidence: tuple[Evidence, ...] = ()
    temporal_evidence: tuple[Evidence, ...] = ()
    statistical_evidence: tuple[Evidence, ...] = ()
    coverage_evidence: tuple[Evidence, ...] = ()
    stability_evidence: tuple[Evidence, ...] = ()
    safety_evidence: tuple[Evidence, ...] = ()

    @property
    def all_evidence(self) -> tuple[Evidence, ...]:
        return (
            self.structural_evidence + self.temporal_evidence + self.statistical_evidence
            + self.coverage_evidence + self.stability_evidence + self.safety_evidence
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "version": self.version, "content_id": self.content_id,
            "split_diagnostics": [s.to_json_dict() for s in self.split_diagnostics], "dimension_scores": self.dimension_scores,
            "overall_score": self.overall_score, "decision": self.decision, "generated_at": self.generated_at,
            "structural_evidence": [e.to_json_dict() for e in self.structural_evidence],
            "temporal_evidence": [e.to_json_dict() for e in self.temporal_evidence],
            "statistical_evidence": [e.to_json_dict() for e in self.statistical_evidence],
            "coverage_evidence": [e.to_json_dict() for e in self.coverage_evidence],
            "stability_evidence": [e.to_json_dict() for e in self.stability_evidence],
            "safety_evidence": [e.to_json_dict() for e in self.safety_evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> QualificationDiagnostics:
        require_schema_version(raw, supported=QUALIFICATION_DIAGNOSTICS_SCHEMA_VERSION, context="QualificationDiagnostics")

        def _evidence_list(field_name: str) -> tuple[Evidence, ...]:
            return tuple(
                Evidence.from_json_dict(as_json_dict(e, field_name=f"{field_name}[]"))
                for e in as_json_list(raw.get(field_name) or [], field_name=field_name)
            )

        return cls(
            schema_version=QUALIFICATION_DIAGNOSTICS_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]), version=str(raw["version"]),
            content_id=str(raw["content_id"]),
            split_diagnostics=tuple(
                SplitDiagnostics.from_json_dict(as_json_dict(s, field_name="split_diagnostics[]"))
                for s in as_json_list(raw.get("split_diagnostics") or [], field_name="split_diagnostics")
            ),
            dimension_scores={str(k): float(str(v)) for k, v in as_json_dict(raw.get("dimension_scores") or {}, field_name="dimension_scores").items()},
            overall_score=float(str(raw["overall_score"])), decision=str(raw["decision"]), generated_at=str(raw["generated_at"]),
            structural_evidence=_evidence_list("structural_evidence"), temporal_evidence=_evidence_list("temporal_evidence"),
            statistical_evidence=_evidence_list("statistical_evidence"), coverage_evidence=_evidence_list("coverage_evidence"),
            stability_evidence=_evidence_list("stability_evidence"), safety_evidence=_evidence_list("safety_evidence"),
        )


def _split_diagnostics(split_name: str, df: pd.DataFrame) -> SplitDiagnostics:
    feature_columns = [c for c in df.columns if c not in _NON_FEATURE_COLUMNS]
    null_fractions = {col: (float(df[col].isna().mean()) if len(df) else 1.0) for col in feature_columns}
    has_open_time = "open_time" in df.columns and len(df) > 0
    return SplitDiagnostics(
        split_name=split_name, row_count=len(df), feature_null_fractions=null_fractions,
        open_time_min=(str(df["open_time"].min()) if has_open_time else None),
        open_time_max=(str(df["open_time"].max()) if has_open_time else None),
    )


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLUMNS]


def _rolling_window(n: int) -> int:
    return max(_ROLLING_WINDOW_MIN, n // 20)


def _skew_kurtosis(values: np.ndarray) -> tuple[float, float]:
    """Population (not sample-bias-corrected) skewness/excess kurtosis,
    computed directly via numpy rather than `pandas.Series.skew`/`.kurt`
    -- avoids those methods' broad, pandas-stubs-inferred return type
    (a large scalar union that isn't safely narrowable to `float` under
    strict mypy) while keeping full control over the exact formula."""
    mean, std = float(values.mean()), float(values.std())
    normalized = (values - mean) / std
    skew = float(np.mean(normalized**3))
    kurtosis = float(np.mean(normalized**4) - 3.0)
    return skew, kurtosis


# --------------------------------------------------------------------------
# Structural Diagnostics: schema, manifests, lineage, identity, replay evidence
# --------------------------------------------------------------------------
def _structural_evidence(manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None, facts: VerificationFacts) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    dim = QualificationDimensionKind.STRUCTURAL_INTEGRITY

    if not facts.artifacts.readable:
        evidence.append(make_evidence(
            finding="manifests: dataset artifacts are unreadable or fail their own checksum verification", severity=Severity.CRITICAL,
            dimension=dim, evidence=(f"error_message={facts.artifacts.error_message!r}",),
            affected_artifacts=(f"dataset_id={manifest.dataset_id} content_id={manifest.content_id}",), blocking=True,
        ))
        return tuple(evidence)

    assert splits is not None
    for split_name, df in sorted(splits.items()):
        expected = set(manifest.feature_names)
        present = set(_feature_columns(df))
        missing = sorted(expected - present)
        evidence.append(make_evidence(
            finding=f"schema: split {split_name!r} has {len(df.columns)} column(s), {len(present)} declared feature column(s)",
            severity=(Severity.CRITICAL if missing else Severity.INFO), dimension=dim,
            evidence=(f"columns={sorted(df.columns)}", f"missing_declared_features={missing}"),
            affected_artifacts=(affected_split(manifest.dataset_id, manifest.content_id, split_name),),
        ))

    evidence.append(make_evidence(
        finding="manifests: content-store metadata.json cross-check", severity=(Severity.INFO if facts.artifacts.metadata_checksums_match_manifest and facts.artifacts.row_counts_match_manifest else Severity.CRITICAL),
        dimension=dim, evidence=(
            f"metadata_checksums_match_manifest={facts.artifacts.metadata_checksums_match_manifest}",
            f"row_counts_match_manifest={facts.artifacts.row_counts_match_manifest}",
        ), affected_artifacts=(f"dataset_id={manifest.dataset_id} content_id={manifest.content_id}",),
    ))

    evidence.append(make_evidence(
        finding="lineage: required provenance fields present" if facts.lineage_present else "lineage: required provenance field(s) missing",
        severity=(Severity.INFO if facts.lineage_present else Severity.CRITICAL), dimension=dim,
        evidence=(f"missing_lineage_fields={list(facts.missing_lineage_fields)}",),
        affected_artifacts=(f"dataset_id={manifest.dataset_id} version={manifest.version}",), blocking=not facts.lineage_present,
    ))

    evidence.append(make_evidence(
        finding="identity: dataset_id reproducible from manifest recipe fields" if facts.identity_matches else "identity: dataset_id does NOT reproduce from manifest recipe fields",
        severity=(Severity.INFO if facts.identity_matches else Severity.CRITICAL), dimension=dim,
        evidence=(f"identity_message={facts.identity_message!r}",),
        affected_artifacts=(f"dataset_id={manifest.dataset_id}",), blocking=not facts.identity_matches,
    ))

    evidence.append(make_evidence(
        finding="replay evidence: manifest.output_content_hashes/row_counts vs. content store metadata.json",
        severity=(Severity.INFO if facts.artifacts.metadata_checksums_match_manifest and facts.artifacts.row_counts_match_manifest else Severity.CRITICAL),
        dimension=dim, evidence=(f"output_content_hashes={manifest.output_content_hashes}", f"row_counts={manifest.row_counts}"),
        affected_artifacts=(f"dataset_id={manifest.dataset_id} content_id={manifest.content_id}",),
        blocking=not (facts.artifacts.metadata_checksums_match_manifest and facts.artifacts.row_counts_match_manifest),
    ))

    if not facts.required_features_present:
        evidence.append(make_evidence(
            finding="schema: required feature(s) missing", severity=Severity.CRITICAL, dimension=dim,
            evidence=(f"missing_required_features={list(facts.missing_required_features)}",),
            affected_artifacts=(f"dataset_id={manifest.dataset_id}",), blocking=True,
        ))
    return tuple(evidence)


# --------------------------------------------------------------------------
# Temporal Diagnostics: macro/cross-asset availability, session alignment,
# stale observations, future visibility
# --------------------------------------------------------------------------
def _market_data_lineage_findings(manifest: ResearchDatasetManifest, source_kind: str) -> list[dict[str, object]]:
    lineage = manifest.market_data_lineage
    if not lineage:
        return []
    coverage_decision = lineage.get("coverage_decision")
    if not isinstance(coverage_decision, dict):
        return []
    findings = coverage_decision.get("findings")
    if not isinstance(findings, list):
        return []
    return [f for f in findings if isinstance(f, dict) and f.get("source_kind") == source_kind]


def _temporal_evidence(manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None, facts: VerificationFacts) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    dim = QualificationDimensionKind.TEMPORAL_INTEGRITY

    if manifest.market_data_lineage is None:
        evidence.append(make_evidence(
            finding="macro/cross-asset availability: this dataset's base data was not sourced through features.market_data_bridge",
            severity=Severity.INFO, dimension=dim, evidence=("manifest.market_data_lineage is None",),
            affected_artifacts=(f"dataset_id={manifest.dataset_id}",),
        ))
    else:
        for source_kind, label in (("macro", "macro availability"), ("cross_asset", "cross-asset availability")):
            findings = _market_data_lineage_findings(manifest, source_kind)
            if not findings:
                continue
            for f in findings:
                status = f.get("status")
                evidence.append(make_evidence(
                    finding=f"{label}: source {f.get('source_name')!r} status={status!r}",
                    severity=(Severity.INFO if status == "ok" else Severity.WARNING), dimension=dim,
                    evidence=(f"required={f.get('required')}", f"coverage_fraction={f.get('coverage_fraction')}"),
                    affected_artifacts=(f"dataset_id={manifest.dataset_id} source={f.get('source_name')}",),
                    recommendation=(None if status == "ok" else "Investigate this source's availability/coverage before relying on this dataset for research involving it."),
                ))
        evidence.append(make_evidence(
            finding="stale macro/stale cross-asset: post-hoc per-row staleness re-verification is out of scope (would require re-joining raw source data)",
            severity=Severity.INFO, dimension=dim, evidence=("see market_data_lineage.coverage_decision for the build-time availability signal recorded above",),
            affected_artifacts=(f"dataset_id={manifest.dataset_id}",),
        ))

    if splits:
        duration = manifest.base_timeframe.duration
        for split_name, df in sorted(splits.items()):
            if "open_time" not in df.columns or len(df) == 0:
                continue
            epoch = pd.Timestamp("1970-01-01", tz="UTC")
            offset_seconds = (df["open_time"] - epoch).dt.total_seconds() % duration.total_seconds()
            misaligned = int((offset_seconds != 0.0).sum())
            if misaligned:
                evidence.append(make_evidence(
                    finding=f"session alignment: split {split_name!r} has {misaligned} row(s) not aligned to {manifest.base_timeframe.value} boundaries",
                    severity=Severity.WARNING, dimension=dim, evidence=(f"misaligned_row_count={misaligned}", f"total_rows={len(df)}"),
                    affected_artifacts=(affected_split(manifest.dataset_id, manifest.content_id, split_name),),
                    recommendation="Investigate the resampling/alignment step that produced this split's open_time values.",
                ))

    if facts.leakage_messages:
        for message in facts.leakage_messages:
            evidence.append(make_evidence(
                finding="future visibility: suspected target leakage", severity=Severity.CRITICAL, dimension=dim,
                evidence=(message,), affected_artifacts=(f"dataset_id={manifest.dataset_id}",), blocking=True,
                recommendation="Investigate the flagged feature(s) for direct or near-direct encoding of the label.",
            ))
    else:
        evidence.append(make_evidence(
            finding="future visibility: no suspected target leakage found", severity=Severity.INFO, dimension=dim,
            evidence=("features.validation.validate_research_dataset reported no TARGET_LEAKAGE_SUSPECTED issues",),
            affected_artifacts=(f"dataset_id={manifest.dataset_id}",),
        ))
    return tuple(evidence)


# --------------------------------------------------------------------------
# Statistical Diagnostics: NaN, Infinity, duplicate rows/timestamps, zero
# variance, near-zero variance, abnormal distributions
# --------------------------------------------------------------------------
def _statistical_evidence(manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    dim = QualificationDimensionKind.STATISTICAL_INTEGRITY
    if not splits:
        return ()

    for split_name, df in sorted(splits.items()):
        artifact = affected_split(manifest.dataset_id, manifest.content_id, split_name)
        feature_columns = _feature_columns(df)
        numeric_columns = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]

        nan_counts = {c: int(df[c].isna().sum()) for c in feature_columns if int(df[c].isna().sum()) > 0}
        if nan_counts:
            evidence.append(make_evidence(
                finding=f"NaN: split {split_name!r} has NaN values in {len(nan_counts)} feature column(s)", severity=Severity.WARNING,
                dimension=dim, evidence=(f"nan_counts={nan_counts}",), affected_artifacts=(artifact,),
            ))

        inf_counts = {c: int(np.isinf(df[c].to_numpy(dtype="float64")).sum()) for c in numeric_columns if int(np.isinf(df[c].to_numpy(dtype="float64")).sum()) > 0}
        if inf_counts:
            evidence.append(make_evidence(
                finding=f"Infinity: split {split_name!r} has non-finite (+/-inf) values in {len(inf_counts)} feature column(s)", severity=Severity.CRITICAL,
                dimension=dim, evidence=(f"inf_counts={inf_counts}",), affected_artifacts=(artifact,),
                recommendation="Trace the feature computation that produced an infinite value -- likely a division by zero.",
            ))

        duplicate_row_count = int(df.duplicated().sum())
        if duplicate_row_count:
            evidence.append(make_evidence(
                finding=f"duplicate rows: split {split_name!r} has {duplicate_row_count} fully duplicate row(s)", severity=Severity.WARNING,
                dimension=dim, evidence=(f"duplicate_row_count={duplicate_row_count}", f"total_rows={len(df)}"), affected_artifacts=(artifact,),
            ))

        if "open_time" in df.columns:
            duplicate_ts_count = int(df["open_time"].duplicated().sum())
            if duplicate_ts_count:
                evidence.append(make_evidence(
                    finding=f"duplicate timestamps: split {split_name!r} has {duplicate_ts_count} duplicate open_time value(s)", severity=Severity.CRITICAL,
                    dimension=dim, evidence=(f"duplicate_timestamp_count={duplicate_ts_count}",), affected_artifacts=(artifact,),
                ))

        if len(numeric_columns) >= 1 and len(df) > 0:
            drift = compare_splits(df[numeric_columns], df[numeric_columns], reference_name=split_name, comparison_name=split_name)
            if drift.constant_features:
                evidence.append(make_evidence(
                    finding=f"zero variance: split {split_name!r} has {len(drift.constant_features)} constant feature(s)", severity=Severity.WARNING,
                    dimension=dim, evidence=(f"constant_features={list(drift.constant_features)}",), affected_artifacts=(artifact,),
                    recommendation="A constant feature carries no information for model training; verify this is expected before use.",
                ))
            if drift.near_constant_features:
                evidence.append(make_evidence(
                    finding=f"near-zero variance: split {split_name!r} has {len(drift.near_constant_features)} near-constant feature(s)", severity=Severity.INFO,
                    dimension=dim, evidence=(f"near_constant_features={list(drift.near_constant_features)}",), affected_artifacts=(artifact,),
                ))

            abnormal: dict[str, dict[str, float]] = {}
            for col in numeric_columns:
                # Skew/kurtosis of a sample containing +/-inf is not meaningful (mean/std themselves
                # become inf/NaN, producing `inf - inf` RuntimeWarnings below) -- finite-only view.
                # A non-finite column is already flagged separately, above, at CRITICAL severity.
                all_values = df[col].dropna().to_numpy(dtype="float64")
                clean_values = all_values[np.isfinite(all_values)]
                if len(clean_values) < 3 or clean_values.std() == 0:
                    continue
                skew, kurtosis = _skew_kurtosis(clean_values)
                if abs(skew) > _ABNORMAL_SKEW_THRESHOLD or abs(kurtosis) > _ABNORMAL_KURTOSIS_THRESHOLD:
                    abnormal[col] = {"skew": skew, "kurtosis": kurtosis}
            if abnormal:
                evidence.append(make_evidence(
                    finding=f"abnormal distributions: split {split_name!r} has {len(abnormal)} feature(s) with |skew| > {_ABNORMAL_SKEW_THRESHOLD} or |excess kurtosis| > {_ABNORMAL_KURTOSIS_THRESHOLD}",
                    severity=Severity.INFO, dimension=dim, evidence=(f"abnormal={abnormal}",), affected_artifacts=(artifact,),
                    recommendation="Heuristic thresholds, not a formal statistical test -- investigate flagged features before assuming a genuine distributional problem.",
                ))
    return tuple(evidence)


# --------------------------------------------------------------------------
# Coverage Diagnostics: feature/source/macro/cross-asset coverage, warmup,
# missing intervals
# --------------------------------------------------------------------------
def _coverage_evidence(manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    dim = QualificationDimensionKind.COVERAGE
    if not splits:
        return ()

    feature_columns = sorted({c for df in splits.values() for c in _feature_columns(df)})
    for col in feature_columns:
        present_count = sum(len(df) for df in splits.values() if col in df.columns)
        null_count = sum(int(df[col].isna().sum()) for df in splits.values() if col in df.columns)
        fraction = (null_count / present_count) if present_count else 1.0
        if fraction > 0:
            evidence.append(make_evidence(
                finding=f"feature coverage: {col!r} is null in {fraction:.1%} of rows across all splits", severity=(Severity.WARNING if fraction > 0.5 else Severity.INFO),
                dimension=dim, evidence=(f"null_count={null_count}", f"present_count={present_count}"),
                affected_artifacts=(f"dataset_id={manifest.dataset_id}",),
            ))

    all_open_times = pd.concat([df["open_time"] for df in splits.values() if "open_time" in df.columns and len(df)], ignore_index=True) if any(len(df) for df in splits.values()) else pd.Series([], dtype="datetime64[ns, UTC]")
    if len(all_open_times):
        observed_start, observed_end = pd.Timestamp(all_open_times.min()), pd.Timestamp(all_open_times.max())
        requested_span = (manifest.utc_end - manifest.utc_start).total_seconds()
        observed_span = (observed_end - observed_start).total_seconds()
        coverage_fraction = max(0.0, min(1.0, observed_span / requested_span)) if requested_span > 0 else 1.0
        evidence.append(make_evidence(
            finding=f"source coverage: observed span covers {coverage_fraction:.1%} of the requested date range", severity=(Severity.WARNING if coverage_fraction < 0.5 else Severity.INFO),
            dimension=dim, evidence=(f"requested=[{manifest.utc_start}, {manifest.utc_end})", f"observed=[{observed_start}, {observed_end}]"),
            affected_artifacts=(f"dataset_id={manifest.dataset_id}",),
        ))

    for source_kind, label in (("macro", "macro coverage"), ("cross_asset", "cross-asset coverage")):
        for f in _market_data_lineage_findings(manifest, source_kind):
            evidence.append(make_evidence(
                finding=f"{label}: source {f.get('source_name')!r} coverage_fraction={f.get('coverage_fraction')}",
                severity=(Severity.INFO if f.get("status") == "ok" else Severity.WARNING), dimension=dim,
                evidence=(f"status={f.get('status')}",), affected_artifacts=(f"dataset_id={manifest.dataset_id} source={f.get('source_name')}",),
            ))

    duration = manifest.base_timeframe.duration
    for split_name, df in sorted(splits.items()):
        if "open_time" not in df.columns or len(df) == 0:
            continue
        artifact = affected_split(manifest.dataset_id, manifest.content_id, split_name)
        feature_cols = _feature_columns(df)
        if feature_cols:
            any_null = df[feature_cols].isna().any(axis=1)
            warmup_rows = int(any_null.cummin().sum()) if any_null.iloc[0] else 0
            if warmup_rows:
                evidence.append(make_evidence(
                    finding=f"warmup: split {split_name!r} has a leading run of {warmup_rows} row(s) with at least one null feature", severity=Severity.INFO,
                    dimension=dim, evidence=(f"warmup_row_count={warmup_rows}", f"total_rows={len(df)}"), affected_artifacts=(artifact,),
                ))

        ordered = df["open_time"].sort_values()
        gaps = ordered.diff().dropna()
        missing_intervals = int((gaps > duration).sum())
        if missing_intervals:
            evidence.append(make_evidence(
                finding=f"missing intervals: split {split_name!r} has {missing_intervals} gap(s) larger than {manifest.base_timeframe.value}", severity=Severity.WARNING,
                dimension=dim, evidence=(f"missing_interval_count={missing_intervals}", f"max_gap={gaps.max()}"), affected_artifacts=(artifact,),
                recommendation="Investigate whether these gaps are expected (market closures) or a data pipeline defect.",
            ))
    return tuple(evidence)


# --------------------------------------------------------------------------
# Stability Diagnostics: rolling variance, rolling missingness, distribution
# drift/PSI, regime drift
# --------------------------------------------------------------------------
def _stability_evidence(manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    dim = QualificationDimensionKind.STABILITY
    if not splits or "train" not in splits or len(splits["train"]) == 0:
        return ()

    train_df = splits["train"]
    train_artifact = affected_split(manifest.dataset_id, manifest.content_id, "train")
    numeric_columns = [c for c in _feature_columns(train_df) if pd.api.types.is_numeric_dtype(train_df[c])]
    window = _rolling_window(len(train_df))

    if window < len(train_df):
        unstable_variance: dict[str, float] = {}
        unstable_missingness: dict[str, float] = {}
        for col in numeric_columns:
            rolling_std = train_df[col].rolling(window=window, min_periods=window).std()
            clean_rolling_std = rolling_std.dropna()
            if len(clean_rolling_std) >= 2 and clean_rolling_std.mean() > 0:
                cv = float(clean_rolling_std.std() / clean_rolling_std.mean())
                if cv > 1.0:
                    unstable_variance[col] = cv

            rolling_missingness = train_df[col].isna().rolling(window=window, min_periods=window).mean()
            clean_rolling_missingness = rolling_missingness.dropna()
            if len(clean_rolling_missingness) >= 2 and (clean_rolling_missingness.max() - clean_rolling_missingness.min()) > 0.3:
                unstable_missingness[col] = float(clean_rolling_missingness.max() - clean_rolling_missingness.min())

        if unstable_variance:
            evidence.append(make_evidence(
                finding=f"rolling variance: {len(unstable_variance)} feature(s) show high coefficient-of-variation in their rolling (window={window}) std across train",
                severity=Severity.INFO, dimension=dim, evidence=(f"coefficient_of_variation={unstable_variance}",), affected_artifacts=(train_artifact,),
            ))
        if unstable_missingness:
            evidence.append(make_evidence(
                finding=f"rolling missingness: {len(unstable_missingness)} feature(s) show a large swing in rolling (window={window}) null fraction across train",
                severity=Severity.WARNING, dimension=dim, evidence=(f"rolling_null_fraction_swing={unstable_missingness}",), affected_artifacts=(train_artifact,),
            ))

    for comparison_name in ("validation", "test"):
        if comparison_name not in splits or len(splits[comparison_name]) == 0:
            continue
        drift = compare_splits(train_df, splits[comparison_name], reference_name="train", comparison_name=comparison_name)
        flagged = {fr.feature_name: fr.population_stability_index for fr in drift.feature_reports if fr.population_stability_index == fr.population_stability_index and fr.population_stability_index >= 0.25}
        evidence.append(make_evidence(
            finding=f"distribution drift/PSI: train vs {comparison_name} -- {len(flagged)}/{len(drift.feature_reports)} feature(s) with PSI >= 0.25",
            severity=(Severity.WARNING if flagged else Severity.INFO), dimension=dim, evidence=(f"psi_by_feature={flagged}",),
            affected_artifacts=(train_artifact, affected_split(manifest.dataset_id, manifest.content_id, comparison_name)),
        ))

    if len(train_df) >= 20:
        midpoint = len(train_df) // 2
        first_half, second_half = train_df.iloc[:midpoint], train_df.iloc[midpoint:]
        regime_drift = compare_splits(first_half, second_half, reference_name="train_first_half", comparison_name="train_second_half")
        regime_flagged = {fr.feature_name: fr.population_stability_index for fr in regime_drift.feature_reports if fr.population_stability_index == fr.population_stability_index and fr.population_stability_index >= 0.25}
        evidence.append(make_evidence(
            finding=f"regime drift: train's first half vs second half -- {len(regime_flagged)}/{len(regime_drift.feature_reports)} feature(s) with PSI >= 0.25",
            severity=(Severity.WARNING if regime_flagged else Severity.INFO), dimension=dim, evidence=(f"psi_by_feature={regime_flagged}",),
            affected_artifacts=(train_artifact,),
            recommendation=("A large intra-train PSI suggests the train split itself spans more than one regime; consider this when interpreting model performance." if regime_flagged else None),
        ))
    return tuple(evidence)


# --------------------------------------------------------------------------
# Safety Diagnostics: leakage evidence, mutable aliases, label
# contamination, preprocessing contamination
# --------------------------------------------------------------------------
def _safety_evidence(manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None, facts: VerificationFacts) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    dim = QualificationDimensionKind.SAFETY
    if not splits:
        return ()

    if facts.leakage_messages:
        for message in facts.leakage_messages:
            evidence.append(make_evidence(
                finding="leakage evidence: suspected target leakage", severity=Severity.CRITICAL, dimension=dim,
                evidence=(message,), affected_artifacts=(f"dataset_id={manifest.dataset_id}",), blocking=True,
            ))

    for split_name, df in sorted(splits.items()):
        artifact = affected_split(manifest.dataset_id, manifest.content_id, split_name)
        numeric_columns = [c for c in _feature_columns(df) if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_columns) >= 2 and len(df) > 0:
            drift = compare_splits(df[numeric_columns], df[numeric_columns], reference_name=split_name, comparison_name=split_name)
            if drift.highly_correlated_pairs:
                evidence.append(make_evidence(
                    finding=f"mutable aliases: split {split_name!r} has {len(drift.highly_correlated_pairs)} feature pair(s) correlated >= {_CORRELATION_THRESHOLD}",
                    severity=Severity.WARNING, dimension=dim,
                    evidence=(f"pairs={[(a, b, round(v, 4)) for a, b, v in drift.highly_correlated_pairs]}",), affected_artifacts=(artifact,),
                    recommendation="Two near-identical feature columns may indicate an accidental alias/copy; confirm both are intentionally distinct.",
                ))

        if "label" in df.columns and len(df) > 0:
            label = df["label"]
            if pd.api.types.is_numeric_dtype(label):
                contaminated: dict[str, float] = {}
                for col in numeric_columns:
                    if df[col].std(skipna=True) in (0, None) or label.std(skipna=True) in (0, None):
                        continue
                    corr = float(df[col].corr(label))
                    if corr == corr and abs(corr) >= _CORRELATION_THRESHOLD:
                        contaminated[col] = corr
                if contaminated:
                    evidence.append(make_evidence(
                        finding=f"label contamination: split {split_name!r} has {len(contaminated)} feature(s) correlated >= {_CORRELATION_THRESHOLD} with label",
                        severity=Severity.CRITICAL, dimension=dim, evidence=(f"correlations={contaminated}",), affected_artifacts=(artifact,), blocking=False,
                        recommendation="A near-perfect feature/label correlation is a strong leakage signal even if not caught by the reserved-prefix check; investigate before training.",
                    ))

    if manifest.preprocessing_definition and not manifest.fitted_preprocessing_fingerprint:
        evidence.append(make_evidence(
            finding="preprocessing contamination: preprocessing_definition is non-empty but no fitted_preprocessing_fingerprint was recorded",
            severity=Severity.WARNING, dimension=dim, evidence=(f"preprocessing_definition={manifest.preprocessing_definition}",),
            affected_artifacts=(f"dataset_id={manifest.dataset_id} version={manifest.version}",),
            recommendation="A declared-but-unfitted preprocessing pipeline cannot be verified as train-only-fit; rebuild through ResearchDatasetBuilder.",
        ))
    return tuple(evidence)


def compute_diagnostics(
    manifest: ResearchDatasetManifest, report: DatasetQualificationReport, research_store: ResearchDatasetStore,
    *, required_feature_names: frozenset[str] = frozenset(),
) -> QualificationDiagnostics:
    facts: VerificationFacts | None
    try:
        verified_facts = QualificationVerifier().verify(manifest, research_store, required_feature_names=required_feature_names)
        facts = verified_facts
        splits = verified_facts.artifacts.splits
    except QualificationVerificationError:
        facts = None
        try:
            splits = research_store.read_artifacts(manifest.dataset_id, manifest.content_id)
        except ResearchDatasetError:
            splits = None

    split_diagnostics = tuple(_split_diagnostics(name, df) for name, df in sorted((splits or {}).items()))
    dimension_scores = {r.dimension.value: r.score for r in report.dimension_results}

    if facts is not None:
        structural_evidence = _structural_evidence(manifest, splits, facts)
        temporal_evidence = _temporal_evidence(manifest, splits, facts)
        safety_evidence = _safety_evidence(manifest, splits, facts)
    else:
        structural_evidence = temporal_evidence = safety_evidence = ()
    statistical_evidence = _statistical_evidence(manifest, splits)
    coverage_evidence = _coverage_evidence(manifest, splits)
    stability_evidence = _stability_evidence(manifest, splits)

    return QualificationDiagnostics(
        schema_version=QUALIFICATION_DIAGNOSTICS_SCHEMA_VERSION, dataset_id=manifest.dataset_id, version=manifest.version, content_id=manifest.content_id,
        split_diagnostics=split_diagnostics, dimension_scores=dimension_scores, overall_score=report.decision.overall_score,
        decision=report.decision.decision.value, generated_at=format_utc_timestamp(utc_now()),
        structural_evidence=structural_evidence, temporal_evidence=temporal_evidence, statistical_evidence=statistical_evidence,
        coverage_evidence=coverage_evidence, stability_evidence=stability_evidence, safety_evidence=safety_evidence,
    )
