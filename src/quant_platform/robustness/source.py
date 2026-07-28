"""Verified backtest input contract (Milestone 6, Section 3). A
robustness run must never trust a backtest merely because its manifest
says COMPLETED -- this module is the ONE place `verify_backtest` is
invoked before any statistical analysis runs, and the ONE place a
`RobustnessSpec`'s declared source identities (dataset/split-plan/
instrument/bar-interval) are cross-checked against what the source
backtest's OWN persisted `BacktestSpec` actually says."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.backtesting.manifests import BacktestEventStore, BacktestManifest, BacktestManifestStore
from quant_platform.backtesting.models import BacktestStage
from quant_platform.backtesting.specs import BacktestSpec
from quant_platform.backtesting.verification import verify_backtest
from quant_platform.calibration.manifests import CalibrationManifestStore
from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    RobustnessSourceVerificationError,
    SchemaVersionError,
)
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import ValidationReport
from quant_platform.ml.persistence import (
    as_json_list,
    format_utc_timestamp,
    parse_json_strict,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.specs import RobustnessSpec

SOURCE_VERIFICATION_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SourceVerificationReport:
    """Section 3: persisted proof that a robustness run's source backtest
    was independently re-verified (never merely trusted from its own
    manifest's `stage` field) before any downstream analysis ran."""

    schema_version: int
    source_backtest_id: str
    verify_backtest_is_ready: bool
    verify_backtest_critical_count: int
    verify_backtest_issue_codes: tuple[str, ...]
    dataset_content_id_matches: bool
    split_plan_fingerprint_matches: bool
    instrument_identity_matches: bool
    bar_interval_matches: bool
    total_outer_folds: int
    generated_at: str

    @property
    def all_checks_passed(self) -> bool:
        return (
            self.verify_backtest_is_ready and self.dataset_content_id_matches and self.split_plan_fingerprint_matches
            and self.instrument_identity_matches and self.bar_interval_matches
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "source_backtest_id": self.source_backtest_id,
            "verify_backtest_is_ready": self.verify_backtest_is_ready, "verify_backtest_critical_count": self.verify_backtest_critical_count,
            "verify_backtest_issue_codes": list(self.verify_backtest_issue_codes),
            "dataset_content_id_matches": self.dataset_content_id_matches, "split_plan_fingerprint_matches": self.split_plan_fingerprint_matches,
            "instrument_identity_matches": self.instrument_identity_matches, "bar_interval_matches": self.bar_interval_matches,
            "total_outer_folds": self.total_outer_folds, "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SourceVerificationReport:
        require_schema_version(raw, supported=SOURCE_VERIFICATION_REPORT_SCHEMA_VERSION, context="SourceVerificationReport")
        return cls(
            schema_version=SOURCE_VERIFICATION_REPORT_SCHEMA_VERSION, source_backtest_id=str(raw["source_backtest_id"]),
            verify_backtest_is_ready=bool(raw["verify_backtest_is_ready"]), verify_backtest_critical_count=int(str(raw["verify_backtest_critical_count"])),
            verify_backtest_issue_codes=tuple(str(c) for c in as_json_list(raw.get("verify_backtest_issue_codes") or [], field_name="verify_backtest_issue_codes")),
            dataset_content_id_matches=bool(raw["dataset_content_id_matches"]), split_plan_fingerprint_matches=bool(raw["split_plan_fingerprint_matches"]),
            instrument_identity_matches=bool(raw["instrument_identity_matches"]), bar_interval_matches=bool(raw["bar_interval_matches"]),
            total_outer_folds=int(str(raw["total_outer_folds"])), generated_at=str(raw["generated_at"]),
        )


@dataclass(frozen=True, slots=True)
class VerifiedBacktestSource:
    """What Section 3's contract hands back to the rest of this package:
    the source backtest's own manifest and spec, ALREADY independently
    re-verified -- never re-derived a second time downstream."""

    manifest: BacktestManifest
    backtest_spec: BacktestSpec
    verification_report: ValidationReport
    source_verification_report: SourceVerificationReport


_UNVERIFIABLE_SOURCE_ERRORS: tuple[type[Exception], ...] = (ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, KeyError, ValueError, TypeError)


def verify_and_load_source_backtest(
    spec: RobustnessSpec, *, backtest_manifest_store: BacktestManifestStore, artifact_store: MLArtifactStore, event_store: BacktestEventStore,
    calibration_manifest_store: CalibrationManifestStore, experiment_manifest_store: ExperimentManifestStore,
    execution_manifest_store: ExecutionManifestStore, research_manifest_store: ResearchManifestStore,
    research_dataset_store: ResearchDatasetStore, dataset_loader: DatasetLoader,
) -> VerifiedBacktestSource:
    """Section 3's ENTIRE contract in one function: load the source
    backtest's manifest, reject anything not `COMPLETED`, run the full
    `verify_backtest` raw-source reconstruction, require `is_ready ==
    True`, decode the source's own persisted `BacktestSpec`, and cross-
    check every source identity this `RobustnessSpec` declared. Fails
    closed (`RobustnessSourceVerificationError`) on the first violation
    -- never proceeds with a partially-trusted source."""
    try:
        manifest = backtest_manifest_store.load(spec.source_backtest_id)
    except _UNVERIFIABLE_SOURCE_ERRORS as exc:
        raise RobustnessSourceVerificationError(
            f"verify_and_load_source_backtest: could not load manifest for source_backtest_id={spec.source_backtest_id!r}: {exc}",
            context={"source_backtest_id": spec.source_backtest_id},
        ) from exc
    if manifest.stage is not BacktestStage.COMPLETED:
        raise RobustnessSourceVerificationError(
            f"verify_and_load_source_backtest: source backtest {spec.source_backtest_id!r} has not reached "
            f"BacktestStage.COMPLETED (currently {manifest.stage.value!r}) -- refusing to analyze an in-progress, "
            "failed, or otherwise non-terminal backtest run",
            context={"source_backtest_id": spec.source_backtest_id, "stage": manifest.stage.value},
        )
    if manifest.spec_reference is None:
        raise RobustnessSourceVerificationError(
            f"verify_and_load_source_backtest: source backtest {spec.source_backtest_id!r} has no recorded BACKTEST_SPEC artifact",
            context={"source_backtest_id": spec.source_backtest_id},
        )

    verification_report = verify_backtest(
        spec.source_backtest_id, backtest_manifest_store=backtest_manifest_store, artifact_store=artifact_store, event_store=event_store,
        calibration_manifest_store=calibration_manifest_store, experiment_manifest_store=experiment_manifest_store,
        execution_manifest_store=execution_manifest_store, research_manifest_store=research_manifest_store,
        research_dataset_store=research_dataset_store, dataset_loader=dataset_loader,
    )
    if not verification_report.is_ready:
        codes = sorted({i.code for i in verification_report.criticals} | {i.code for i in verification_report.errors})
        raise RobustnessSourceVerificationError(
            f"verify_and_load_source_backtest: source backtest {spec.source_backtest_id!r} failed independent "
            f"re-verification (verify_backtest.is_ready=False) -- {len(verification_report.criticals)} critical "
            f"issue(s): {codes}",
            context={"source_backtest_id": spec.source_backtest_id, "critical_count": len(verification_report.criticals), "issue_codes": ",".join(codes)},
        )

    try:
        raw = artifact_store.read_artifact(manifest.spec_reference.content_hash)
        backtest_spec = BacktestSpec.from_json_dict(parse_json_strict(raw.decode("utf-8")))
    except _UNVERIFIABLE_SOURCE_ERRORS as exc:
        raise RobustnessSourceVerificationError(
            f"verify_and_load_source_backtest: source backtest {spec.source_backtest_id!r}'s own BacktestSpec could not be read/decoded: {exc}",
            context={"source_backtest_id": spec.source_backtest_id},
        ) from exc

    dataset_matches = backtest_spec.dataset_content_id == spec.dataset_content_id
    split_matches = backtest_spec.split_plan_fingerprint == spec.split_plan_fingerprint
    instrument_matches = backtest_spec.instrument_identity == spec.instrument_identity
    bar_interval_matches = backtest_spec.bar_interval == spec.bar_interval

    source_report = SourceVerificationReport(
        schema_version=SOURCE_VERIFICATION_REPORT_SCHEMA_VERSION, source_backtest_id=spec.source_backtest_id,
        verify_backtest_is_ready=verification_report.is_ready, verify_backtest_critical_count=len(verification_report.criticals),
        verify_backtest_issue_codes=tuple(sorted({i.code for i in verification_report.issues})),
        dataset_content_id_matches=dataset_matches, split_plan_fingerprint_matches=split_matches,
        instrument_identity_matches=instrument_matches, bar_interval_matches=bar_interval_matches,
        total_outer_folds=(manifest.total_outer_folds or 0), generated_at=format_utc_timestamp(utc_now()),
    )
    if not source_report.all_checks_passed:
        raise RobustnessSourceVerificationError(
            f"verify_and_load_source_backtest: source_backtest_id={spec.source_backtest_id!r} failed identity "
            f"cross-checks against the declared RobustnessSpec -- dataset_content_id_matches={dataset_matches}, "
            f"split_plan_fingerprint_matches={split_matches}, instrument_identity_matches={instrument_matches}, "
            f"bar_interval_matches={bar_interval_matches}",
            context={"source_backtest_id": spec.source_backtest_id},
        )
    if (manifest.total_outer_folds or 0) < spec.minimum_fold_count:
        raise RobustnessSourceVerificationError(
            f"verify_and_load_source_backtest: source backtest {spec.source_backtest_id!r} has "
            f"{manifest.total_outer_folds!r} outer folds, below RobustnessSpec.minimum_fold_count={spec.minimum_fold_count!r}",
            context={"source_backtest_id": spec.source_backtest_id, "total_outer_folds": manifest.total_outer_folds, "minimum_fold_count": spec.minimum_fold_count},
        )

    return VerifiedBacktestSource(manifest=manifest, backtest_spec=backtest_spec, verification_report=verification_report, source_verification_report=source_report)


__all__ = [
    "SOURCE_VERIFICATION_REPORT_SCHEMA_VERSION",
    "SourceVerificationReport",
    "VerifiedBacktestSource",
    "verify_and_load_source_backtest",
]
