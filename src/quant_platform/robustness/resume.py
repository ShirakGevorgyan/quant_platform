"""Resume support (Milestone 6, Section 14) -- discovering how far into
an interrupted `RobustnessSpec` run's LINEAR pipeline the persisted
artifacts can genuinely, verifiably be trusted. Mirrors `backtesting.
resume`'s "never trust the manifest's claim alone" philosophy: a
manifest merely claiming `stage=STRESS_COMPLETED` is not sufficient --
this module re-reads the specific artifact `named_artifacts` claims for
every stage up to and including the current one, and demotes the resume
start point to the first stage whose artifact is missing, unreadable, or
fails its own decoded self-check.

Unlike `backtesting.resume` (which verifies a variable number of outer
folds), this package's pipeline is a fixed, linear sequence of AT MOST
ONE artifact-producing step per stage (see `robustness.models.
RobustnessStage`) -- `verify_completed_robustness_stages` therefore walks
`_ROBUSTNESS_STAGE_ORDER` forward, not a fold index set."""

from __future__ import annotations

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    RobustnessError,
    RobustnessResumeError,
    SchemaVersionError,
)
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.persistence import parse_json_strict
from quant_platform.robustness.bootstrap import BootstrapReport, DownsideAnalysisReport
from quant_platform.robustness.manifests import RobustnessManifest
from quant_platform.robustness.models import RobustnessStage, is_terminal_robustness_stage
from quant_platform.robustness.promotion import PromotionDecision
from quant_platform.robustness.regimes import RegimeReport
from quant_platform.robustness.selection import SelectionReport
from quant_platform.robustness.sensitivity import SensitivityReport
from quant_platform.robustness.series import ReturnSeriesBundle
from quant_platform.robustness.source import SourceVerificationReport
from quant_platform.robustness.stability import FoldStabilityReport
from quant_platform.robustness.stress import StressReport

_UNSAFE_DECODE_ERRORS: tuple[type[Exception], ...] = (
    ArtifactNotFoundError, ArtifactCorruptionError, SchemaVersionError, RobustnessError, KeyError, ValueError, TypeError,
)
"""Every failure mode a claimed-complete artifact can legitimately hit --
identical in spirit to `backtesting.resume._UNSAFE_DECODE_ERRORS`. Every
report's own `from_json_dict` re-runs `__post_init__`'s structural
checks, which raise ONE of `RobustnessError`'s many subclasses
(`BootstrapError`, `StabilityAnalysisError`, `StressAnalysisError`, ...)
depending on which report is being decoded -- catching the common base
here, rather than enumerating every subclass, correctly demotes a
tampered artifact of ANY of these report kinds to "needs rerun" without
this module needing to track every report module's own specific
exception type."""

_STAGE_ARTIFACT_KINDS: dict[RobustnessStage, tuple[str, ...]] = {
    RobustnessStage.SOURCE_VERIFIED: ("source_verification_report",),
    RobustnessStage.SERIES_BUILT: ("return_series_bundle",),
    RobustnessStage.BOOTSTRAP_COMPLETED: ("bootstrap_report", "downside_analysis_report"),
    RobustnessStage.STABILITY_COMPLETED: ("fold_stability_report",),
    RobustnessStage.STRESS_COMPLETED: ("stress_report",),
    RobustnessStage.REGIMES_COMPLETED: (),
    RobustnessStage.SELECTION_COMPLETED: ("selection_report",),
    RobustnessStage.PROMOTION_EVALUATED: ("promotion_decision",),
    RobustnessStage.VERIFIED: ("verification_report",),
    RobustnessStage.COMPLETED: ("robustness_report",),
}
"""Which `manifests.ARTIFACT_KINDS` keys a given stage is expected to
have populated once it has been reached. `REGIMES_COMPLETED` has none:
`RegimeReport` is optional (Section 11 -- regime coverage can be entirely
skipped for a source lacking sufficient history), so its presence is
never a precondition for trusting later stages; when present it is still
verified opportunistically via `_OPTIONAL_STAGE_ARTIFACT_KINDS`."""

_OPTIONAL_STAGE_ARTIFACT_KINDS: dict[RobustnessStage, tuple[str, ...]] = {
    RobustnessStage.REGIMES_COMPLETED: ("regime_report",),
}

_DECODERS: dict[str, type] = {
    "source_verification_report": SourceVerificationReport,
    "return_series_bundle": ReturnSeriesBundle,
    "bootstrap_report": BootstrapReport,
    "downside_analysis_report": DownsideAnalysisReport,
    "fold_stability_report": FoldStabilityReport,
    "sensitivity_report": SensitivityReport,
    "stress_report": StressReport,
    "regime_report": RegimeReport,
    "selection_report": SelectionReport,
    "promotion_decision": PromotionDecision,
    "verification_report": None,  # type: ignore[dict-item]
    "robustness_report": None,  # type: ignore[dict-item]
}
"""`verification_report`/`robustness_report` are decoded by `verification.
py`/`reporting.py` respectively (this module only needs to confirm their
bytes are readable, not their full internal schema) -- `None` here means
"presence/readability only", not "no decoder exists"."""

_ROBUSTNESS_STAGE_ORDER: tuple[RobustnessStage, ...] = (
    RobustnessStage.CREATED, RobustnessStage.SOURCE_VERIFIED, RobustnessStage.SERIES_BUILT, RobustnessStage.BOOTSTRAP_COMPLETED,
    RobustnessStage.STABILITY_COMPLETED, RobustnessStage.STRESS_COMPLETED, RobustnessStage.REGIMES_COMPLETED,
    RobustnessStage.SELECTION_COMPLETED, RobustnessStage.PROMOTION_EVALUATED, RobustnessStage.VERIFIED, RobustnessStage.COMPLETED,
)


def can_resume(manifest: RobustnessManifest | None) -> bool:
    if manifest is None:
        return False
    return not is_terminal_robustness_stage(manifest.stage)


def _artifact_readable(manifest: RobustnessManifest, *, kind: str, artifact_store: MLArtifactStore) -> bool:
    reference = manifest.artifact(kind)
    if reference is None:
        return False
    try:
        raw = artifact_store.read_artifact(reference.content_hash)
        decoder = _DECODERS[kind]
        if decoder is not None:
            decoder.from_json_dict(parse_json_strict(raw.decode("utf-8")))  # type: ignore[attr-defined]
    except _UNSAFE_DECODE_ERRORS:
        return False
    return True


def verify_completed_robustness_stages(manifest: RobustnessManifest, *, artifact_store: MLArtifactStore) -> RobustnessStage:
    """Walks `_ROBUSTNESS_STAGE_ORDER` forward from `CREATED`, re-reading
    and re-decoding every MANDATORY artifact each stage is expected to
    have produced. Returns the LAST stage whose own artifact(s) (and
    every earlier stage's) verified successfully -- never trusts
    `manifest.stage`'s own claim past that point."""
    verified_through = RobustnessStage.CREATED
    for stage in _ROBUSTNESS_STAGE_ORDER[1:]:
        required_kinds = _STAGE_ARTIFACT_KINDS.get(stage, ())
        required_ok = all(_artifact_readable(manifest, kind=kind, artifact_store=artifact_store) for kind in required_kinds)
        if not required_ok:
            break
        optional_kinds = _OPTIONAL_STAGE_ARTIFACT_KINDS.get(stage, ())
        optional_ok = all(
            _artifact_readable(manifest, kind=kind, artifact_store=artifact_store) for kind in optional_kinds if manifest.artifact(kind) is not None
        )
        if not optional_ok:
            break
        verified_through = stage
    return verified_through


def require_robustness_resumable(manifest: RobustnessManifest | None, *, robustness_id: str) -> RobustnessManifest:
    if manifest is None:
        raise RobustnessResumeError(
            f"No robustness manifest exists for robustness_id={robustness_id!r} -- nothing to resume",
            context={"robustness_id": robustness_id},
        )
    if not can_resume(manifest):
        raise RobustnessResumeError(
            f"Robustness run {robustness_id!r} already reached a terminal stage {manifest.stage.value!r} -- it cannot "
            "be resumed or restarted in place",
            context={"robustness_id": robustness_id, "stage": manifest.stage.value},
        )
    return manifest


def resolve_resume_start_stage(manifest: RobustnessManifest, *, artifact_store: MLArtifactStore) -> RobustnessStage:
    """Where the runner's pipeline should resume from: the LAST stage
    whose artifacts independently re-verify, per `verify_completed_
    robustness_stages` -- which may be earlier than `manifest.stage`
    itself claims if an artifact was corrupted, truncated, or tampered
    with after being written."""
    return verify_completed_robustness_stages(manifest, artifact_store=artifact_store)


__all__ = [
    "can_resume",
    "require_robustness_resumable",
    "resolve_resume_start_stage",
    "verify_completed_robustness_stages",
]
