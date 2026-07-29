"""Eligibility/source verification (Milestone 7, Section 4). Before a
paper-trading session may be created, `verify_paper_trading_eligibility`
independently walks and re-verifies the ENTIRE chain a `PaperTradingSpec`
claims to rest on -- it never trusts a persisted `is_ready=True` flag or a
caller-declared identity string at face value:

    PromotionDecision exists
      -> its content genuinely decodes
      -> its decision kind is exactly ELIGIBLE_FOR_PAPER_TRADING
      -> it references the requested robustness result
      -> the robustness result independently re-verifies (`robustness.
         verification.verify_robustness`, which itself recomputes every
         statistic and the promotion decision from raw evidence)
      -> the source backtest independently re-verifies (`robustness.
         source.verify_and_load_source_backtest`, which calls `backtesting.
         verification.verify_backtest`)
      -> the resolved strategy/model/calibration/feature identities match
         what the `PaperTradingSpec` declares
      -> every required source artifact is present, COMPLETED, and
         unmodified

Any expected verification failure (missing/tampered/rejected/incomplete
candidate) is captured as a `EligibilityVerificationReport` with
`is_eligible=False` and a `failed_step`/`failure_reason` -- this function
never raises for an ordinary "not eligible" outcome. Structural failures
inside the underlying stores (I/O errors, corrupted artifacts) surface as
the SAME kind of report entry (any `QuantPlatformError` raised by a lower
layer is caught and recorded), never as an uncaught exception -- session
creation always gets a clear, typed answer. `require_paper_trading_
eligibility` is the fail-closed convenience wrapper `runner.py` calls
directly, raising `PaperTradingEligibilityError` if the report is not
eligible."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.backtesting.manifests import BacktestEventStore, BacktestManifestStore
from quant_platform.calibration.manifests import CalibrationManifestStore
from quant_platform.core.exceptions import PaperTradingEligibilityError, QuantPlatformError
from quant_platform.execution.manifests import ExecutionManifestStore
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.manifests import ExperimentManifestStore
from quant_platform.ml.models import ValidationSeverity
from quant_platform.ml.persistence import (
    as_json_dict,
    format_utc_timestamp,
    parse_json_strict,
    require_schema_version,
    utc_now,
)
from quant_platform.paper_trading.specs import PaperTradingSpec, compute_paper_session_spec_id
from quant_platform.robustness.manifests import RobustnessManifestStore
from quant_platform.robustness.models import PromotionDecisionKind, RobustnessStage
from quant_platform.robustness.promotion import PromotionDecision
from quant_platform.robustness.source import verify_and_load_source_backtest
from quant_platform.robustness.specs import RobustnessSpec
from quant_platform.robustness.verification import verify_robustness

ELIGIBILITY_VERIFICATION_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EligibilityVerificationEnvironment:
    """Every store/loader the eligibility chain (and everything it calls
    transitively -- `verify_robustness`, `verify_and_load_source_backtest`,
    `verify_backtest`) needs. Mirrors `ml_cli.py`'s own `_RobustnessSourceStores`
    bundle one layer up; `paper_trading`'s CLI layer is responsible for
    constructing one instance of this per invocation from its own config,
    exactly like every other store-bundle in this platform."""

    robustness_manifest_store: RobustnessManifestStore
    artifact_store: MLArtifactStore
    backtest_manifest_store: BacktestManifestStore
    backtest_event_store: BacktestEventStore
    calibration_manifest_store: CalibrationManifestStore
    experiment_manifest_store: ExperimentManifestStore
    execution_manifest_store: ExecutionManifestStore
    research_manifest_store: ResearchManifestStore
    research_dataset_store: ResearchDatasetStore
    dataset_loader: DatasetLoader


@dataclass(frozen=True, slots=True)
class EligibilityVerificationReport:
    """Persisted verbatim into the session manifest (Section 4: "Persist
    eligibility-verification report in manifest"). Each of the 9 named
    booleans corresponds exactly to one link in Section 4's chain, in
    order; `failed_step` names the FIRST one that failed (short-circuits,
    since later steps are meaningless once an earlier one has already
    failed), never left ambiguous between "failed" and "not attempted"."""

    schema_version: int
    paper_session_spec_id: str
    robustness_id: str
    promotion_decision_content_hash: str
    promotion_decision_exists: bool
    promotion_decision_content_verifies: bool
    decision_kind_is_eligible_for_paper_trading: bool
    promotion_decision_references_requested_robustness: bool
    robustness_result_verifies: bool
    source_backtest_verifies: bool
    identities_match: bool
    required_source_artifacts_present: bool
    no_artifact_failed_incomplete_stale_or_mismatched: bool
    resolved_strategy_candidate_identity: str | None
    resolved_model_artifact_identity: str | None
    resolved_calibration_artifact_identity: str | None
    resolved_feature_spec_identity: str | None
    failed_step: str | None
    failure_reason: str | None
    generated_at: str

    @property
    def is_eligible(self) -> bool:
        return (
            self.promotion_decision_exists and self.promotion_decision_content_verifies and self.decision_kind_is_eligible_for_paper_trading
            and self.promotion_decision_references_requested_robustness and self.robustness_result_verifies and self.source_backtest_verifies
            and self.identities_match and self.required_source_artifacts_present and self.no_artifact_failed_incomplete_stale_or_mismatched
            and self.failed_step is None
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "paper_session_spec_id": self.paper_session_spec_id, "robustness_id": self.robustness_id,
            "promotion_decision_content_hash": self.promotion_decision_content_hash, "promotion_decision_exists": self.promotion_decision_exists,
            "promotion_decision_content_verifies": self.promotion_decision_content_verifies,
            "decision_kind_is_eligible_for_paper_trading": self.decision_kind_is_eligible_for_paper_trading,
            "promotion_decision_references_requested_robustness": self.promotion_decision_references_requested_robustness,
            "robustness_result_verifies": self.robustness_result_verifies, "source_backtest_verifies": self.source_backtest_verifies,
            "identities_match": self.identities_match, "required_source_artifacts_present": self.required_source_artifacts_present,
            "no_artifact_failed_incomplete_stale_or_mismatched": self.no_artifact_failed_incomplete_stale_or_mismatched,
            "resolved_strategy_candidate_identity": self.resolved_strategy_candidate_identity,
            "resolved_model_artifact_identity": self.resolved_model_artifact_identity,
            "resolved_calibration_artifact_identity": self.resolved_calibration_artifact_identity,
            "resolved_feature_spec_identity": self.resolved_feature_spec_identity, "failed_step": self.failed_step,
            "failure_reason": self.failure_reason, "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> EligibilityVerificationReport:
        require_schema_version(raw, supported=ELIGIBILITY_VERIFICATION_REPORT_SCHEMA_VERSION, context="EligibilityVerificationReport")
        return cls(
            schema_version=ELIGIBILITY_VERIFICATION_REPORT_SCHEMA_VERSION, paper_session_spec_id=str(raw["paper_session_spec_id"]),
            robustness_id=str(raw["robustness_id"]), promotion_decision_content_hash=str(raw["promotion_decision_content_hash"]),
            promotion_decision_exists=bool(raw["promotion_decision_exists"]), promotion_decision_content_verifies=bool(raw["promotion_decision_content_verifies"]),
            decision_kind_is_eligible_for_paper_trading=bool(raw["decision_kind_is_eligible_for_paper_trading"]),
            promotion_decision_references_requested_robustness=bool(raw["promotion_decision_references_requested_robustness"]),
            robustness_result_verifies=bool(raw["robustness_result_verifies"]), source_backtest_verifies=bool(raw["source_backtest_verifies"]),
            identities_match=bool(raw["identities_match"]), required_source_artifacts_present=bool(raw["required_source_artifacts_present"]),
            no_artifact_failed_incomplete_stale_or_mismatched=bool(raw["no_artifact_failed_incomplete_stale_or_mismatched"]),
            resolved_strategy_candidate_identity=(None if raw.get("resolved_strategy_candidate_identity") is None else str(raw["resolved_strategy_candidate_identity"])),
            resolved_model_artifact_identity=(None if raw.get("resolved_model_artifact_identity") is None else str(raw["resolved_model_artifact_identity"])),
            resolved_calibration_artifact_identity=(None if raw.get("resolved_calibration_artifact_identity") is None else str(raw["resolved_calibration_artifact_identity"])),
            resolved_feature_spec_identity=(None if raw.get("resolved_feature_spec_identity") is None else str(raw["resolved_feature_spec_identity"])),
            failed_step=(None if raw.get("failed_step") is None else str(raw["failed_step"])),
            failure_reason=(None if raw.get("failure_reason") is None else str(raw["failure_reason"])), generated_at=str(raw["generated_at"]),
        )


def verify_paper_trading_eligibility(spec: PaperTradingSpec, *, environment: EligibilityVerificationEnvironment) -> EligibilityVerificationReport:
    robustness_id = spec.verified_robustness_id
    promotion_decision_content_hash = spec.verified_promotion_decision_id
    paper_session_spec_id = compute_paper_session_spec_id(spec).paper_session_spec_id

    passed_steps = {
        "promotion_decision_exists": False, "promotion_decision_content_verifies": False, "decision_kind_is_eligible_for_paper_trading": False,
        "promotion_decision_references_requested_robustness": False, "robustness_result_verifies": False, "source_backtest_verifies": False,
        "identities_match": False, "required_source_artifacts_present": False, "no_artifact_failed_incomplete_stale_or_mismatched": False,
    }
    resolved: dict[str, str | None] = {
        "resolved_strategy_candidate_identity": None, "resolved_model_artifact_identity": None,
        "resolved_calibration_artifact_identity": None, "resolved_feature_spec_identity": None,
    }

    def _report(*, failed_step: str | None, failure_reason: str | None) -> EligibilityVerificationReport:
        return EligibilityVerificationReport(
            schema_version=ELIGIBILITY_VERIFICATION_REPORT_SCHEMA_VERSION, paper_session_spec_id=paper_session_spec_id, robustness_id=robustness_id,
            promotion_decision_content_hash=promotion_decision_content_hash, failed_step=failed_step, failure_reason=failure_reason,
            generated_at=format_utc_timestamp(utc_now()), **passed_steps, **resolved,  # type: ignore[arg-type]
        )

    try:
        manifest = environment.robustness_manifest_store.load(robustness_id)
    except QuantPlatformError as exc:
        return _report(failed_step="promotion_decision_exists", failure_reason=f"robustness manifest not found for robustness_id={robustness_id!r}: {exc}")

    if manifest.stage is not RobustnessStage.COMPLETED:
        return _report(
            failed_step="no_artifact_failed_incomplete_stale_or_mismatched",
            failure_reason=f"robustness run {robustness_id!r} is not COMPLETED (stage={manifest.stage.value!r}) -- incomplete source artifact",
        )

    promotion_reference = manifest.artifact("promotion_decision")
    if promotion_reference is None:
        return _report(failed_step="promotion_decision_exists", failure_reason=f"robustness run {robustness_id!r} has no persisted promotion_decision artifact")
    if promotion_reference.content_hash != promotion_decision_content_hash:
        return _report(
            failed_step="promotion_decision_exists",
            failure_reason=(
                f"declared verified_promotion_decision_id={promotion_decision_content_hash!r} does not match robustness run "
                f"{robustness_id!r}'s own promotion_decision artifact content_hash={promotion_reference.content_hash!r} -- "
                "tampered, stale, or belongs to a different evaluation"
            ),
        )
    passed_steps["promotion_decision_exists"] = True

    try:
        raw_promotion_bytes = environment.artifact_store.read_artifact(promotion_decision_content_hash)
        promotion = PromotionDecision.from_json_dict(as_json_dict(parse_json_strict(raw_promotion_bytes.decode("utf-8")), field_name="promotion_decision"))
    except (QuantPlatformError, ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        # `parse_json_strict` raises plain `ValueError` on malformed JSON
        # (not a `QuantPlatformError`), and a structurally-wrong-but-valid
        # JSON payload can raise `KeyError`/`TypeError`/`ValueError` out of
        # `PromotionDecision.from_json_dict` itself (e.g. an invalid enum
        # value) -- every one of those is an ordinary "this artifact is
        # corrupted or malformed" outcome that belongs in the report, not
        # an uncaught crash.
        return _report(failed_step="promotion_decision_content_verifies", failure_reason=f"promotion_decision artifact could not be read/decoded: {exc}")
    passed_steps["promotion_decision_content_verifies"] = True

    if promotion.decision is not PromotionDecisionKind.ELIGIBLE_FOR_PAPER_TRADING:
        return _report(
            failed_step="decision_kind_is_eligible_for_paper_trading",
            failure_reason=f"promotion decision kind is {promotion.decision.value!r}, not 'eligible_for_paper_trading' ({promotion.decision_reason})",
        )
    passed_steps["decision_kind_is_eligible_for_paper_trading"] = True

    if promotion.robustness_id != robustness_id:
        return _report(
            failed_step="promotion_decision_references_requested_robustness",
            failure_reason=f"promotion decision references robustness_id={promotion.robustness_id!r}, not the requested {robustness_id!r} -- decision belongs to another candidate",
        )
    passed_steps["promotion_decision_references_requested_robustness"] = True

    try:
        if manifest.spec_reference is None:
            raise PaperTradingEligibilityError(f"robustness run {robustness_id!r} has no persisted spec_reference")
        raw_spec_bytes = environment.artifact_store.read_artifact(manifest.spec_reference.content_hash)
        robustness_spec = RobustnessSpec.from_json_dict(as_json_dict(parse_json_strict(raw_spec_bytes.decode("utf-8")), field_name="robustness_spec"))
        verification_report = verify_robustness(
            robustness_id, robustness_manifest_store=environment.robustness_manifest_store, artifact_store=environment.artifact_store,
            backtest_manifest_store=environment.backtest_manifest_store, backtest_event_store=environment.backtest_event_store,
            calibration_manifest_store=environment.calibration_manifest_store, experiment_manifest_store=environment.experiment_manifest_store,
            execution_manifest_store=environment.execution_manifest_store, research_manifest_store=environment.research_manifest_store,
            research_dataset_store=environment.research_dataset_store, dataset_loader=environment.dataset_loader,
        )
    except (QuantPlatformError, ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        return _report(failed_step="robustness_result_verifies", failure_reason=f"verify_robustness raised: {exc}")

    if not verification_report.is_ready:
        codes = ", ".join(sorted({issue.code for issue in verification_report.issues if issue.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR)}))
        return _report(failed_step="robustness_result_verifies", failure_reason=f"verify_robustness found unresolved CRITICAL/ERROR issue(s): {codes}")
    passed_steps["robustness_result_verifies"] = True

    try:
        source = verify_and_load_source_backtest(
            robustness_spec, backtest_manifest_store=environment.backtest_manifest_store, artifact_store=environment.artifact_store,
            event_store=environment.backtest_event_store, calibration_manifest_store=environment.calibration_manifest_store,
            experiment_manifest_store=environment.experiment_manifest_store, execution_manifest_store=environment.execution_manifest_store,
            research_manifest_store=environment.research_manifest_store, research_dataset_store=environment.research_dataset_store,
            dataset_loader=environment.dataset_loader,
        )
    except QuantPlatformError as exc:
        return _report(failed_step="source_backtest_verifies", failure_reason=f"source backtest could not be independently verified: {exc}")
    passed_steps["source_backtest_verifies"] = True

    resolved["resolved_strategy_candidate_identity"] = source.manifest.backtest_id
    resolved["resolved_calibration_artifact_identity"] = source.backtest_spec.source_calibration_id

    try:
        experiment_manifest = environment.experiment_manifest_store.load(source.backtest_spec.source_experiment_id)
    except QuantPlatformError as exc:
        return _report(failed_step="identities_match", failure_reason=f"source experiment {source.backtest_spec.source_experiment_id!r} could not be loaded: {exc}")
    resolved["resolved_model_artifact_identity"] = experiment_manifest.identity.experiment_id
    resolved["resolved_feature_spec_identity"] = experiment_manifest.spec.feature_binding.feature_registry_fingerprint

    identity_mismatches: list[str] = []
    if spec.strategy_candidate_identity != resolved["resolved_strategy_candidate_identity"]:
        identity_mismatches.append(f"strategy_candidate_identity: declared={spec.strategy_candidate_identity!r} resolved={resolved['resolved_strategy_candidate_identity']!r}")
    if spec.model_artifact_identity != resolved["resolved_model_artifact_identity"]:
        identity_mismatches.append(f"model_artifact_identity: declared={spec.model_artifact_identity!r} resolved={resolved['resolved_model_artifact_identity']!r}")
    if spec.calibration_artifact_identity != resolved["resolved_calibration_artifact_identity"]:
        identity_mismatches.append(f"calibration_artifact_identity: declared={spec.calibration_artifact_identity!r} resolved={resolved['resolved_calibration_artifact_identity']!r}")
    if spec.feature_spec_identity != resolved["resolved_feature_spec_identity"]:
        identity_mismatches.append(f"feature_spec_identity: declared={spec.feature_spec_identity!r} resolved={resolved['resolved_feature_spec_identity']!r}")
    if identity_mismatches:
        return _report(failed_step="identities_match", failure_reason="; ".join(identity_mismatches))
    passed_steps["identities_match"] = True

    required_kinds = ("source_verification_report", "return_series_bundle", "bootstrap_report", "fold_stability_report", "stress_report", "selection_report", "promotion_decision")
    missing_kinds = [kind for kind in required_kinds if manifest.artifact(kind) is None]
    if missing_kinds:
        return _report(failed_step="required_source_artifacts_present", failure_reason=f"robustness run {robustness_id!r} is missing required artifact(s): {', '.join(missing_kinds)}")
    passed_steps["required_source_artifacts_present"] = True

    if source.manifest.stage.value != "completed":
        return _report(failed_step="no_artifact_failed_incomplete_stale_or_mismatched", failure_reason=f"source backtest {source.manifest.backtest_id!r} is not COMPLETED (stage={source.manifest.stage.value!r})")
    passed_steps["no_artifact_failed_incomplete_stale_or_mismatched"] = True

    return _report(failed_step=None, failure_reason=None)


def require_paper_trading_eligibility(spec: PaperTradingSpec, *, environment: EligibilityVerificationEnvironment) -> EligibilityVerificationReport:
    """Fail-closed convenience wrapper (Section 0: "reject before session
    start without verified ELIGIBLE_FOR_PAPER_TRADING") -- `runner.py`
    calls this directly rather than `verify_paper_trading_eligibility`,
    so an ineligible candidate can never silently proceed past this
    point."""
    report = verify_paper_trading_eligibility(spec, environment=environment)
    if not report.is_eligible:
        raise PaperTradingEligibilityError(
            f"Session spec {report.paper_session_spec_id!r} is NOT eligible for paper trading "
            f"(failed_step={report.failed_step!r}): {report.failure_reason}"
        )
    return report


__all__ = [
    "ELIGIBILITY_VERIFICATION_REPORT_SCHEMA_VERSION",
    "EligibilityVerificationEnvironment",
    "EligibilityVerificationReport",
    "require_paper_trading_eligibility",
    "verify_paper_trading_eligibility",
]
