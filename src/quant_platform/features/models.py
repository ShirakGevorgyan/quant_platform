"""Feature metadata: the immutable, JSON-serializable description of what a
feature is, what it needs, and when it is allowed to become available --
completely separate from the callable that actually computes it (see
`interfaces.FeatureDefinition`).

Keeping `FeatureSpec` pure metadata (no callables, fully JSON round-
trippable) is what makes `FeatureRegistry.fingerprint` deterministic and
storable in a research dataset manifest: two `FeatureSpec` instances with
identical fields always produce the identical fingerprint contribution,
independent of Python object identity, callable source location, or
process.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from quant_platform.core.exceptions import FeatureError
from quant_platform.core.types import Timeframe


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise FeatureError(f"Expected a JSON array, got {type(value).__name__}")
    return value


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FeatureError(f"Expected a JSON object, got {type(value).__name__}")
    return value


class FeatureCategory(Enum):
    PRICE = "price"
    TEMPORAL = "temporal"
    MULTI_TIMEFRAME = "multi_timeframe"
    CROSS_ASSET = "cross_asset"
    MACRO = "macro"


class MissingPolicyKind(Enum):
    """See `quant_platform.features.missing` for the actual application
    logic. This enum is metadata only -- part of `FeatureSpec`, hence part
    of the registry fingerprint -- so a feature's declared missing-data
    contract is itself versioned."""

    PRESERVE_NULL = "preserve_null"
    FORWARD_FILL_MAX_AGE = "forward_fill_max_age"
    CONSTANT_FILL = "constant_fill"
    TRAINING_STATISTIC_FILL = "training_statistic_fill"
    DROP_ROW = "drop_row"


@dataclass(frozen=True, slots=True)
class MissingPolicySpec:
    """Immutable, JSON-serializable missing-value policy declaration
    attached to a `FeatureSpec`. `training_statistic_fill` never computes
    its own statistic here -- see `missing.apply_missing_policy`'s
    docstring for why the fitted value must always be supplied externally,
    fitted only on a training partition."""

    kind: MissingPolicyKind = MissingPolicyKind.PRESERVE_NULL
    max_age_bars: int | None = None
    constant_value: float | None = None
    statistic: str = "mean"
    add_missing_indicator: bool = False

    def __post_init__(self) -> None:
        if self.kind is MissingPolicyKind.FORWARD_FILL_MAX_AGE and self.max_age_bars is None:
            raise ValueError("max_age_bars is required when kind=FORWARD_FILL_MAX_AGE")
        if self.kind is MissingPolicyKind.CONSTANT_FILL and self.constant_value is None:
            raise ValueError("constant_value is required when kind=CONSTANT_FILL")
        if self.statistic not in ("mean", "median"):
            raise ValueError(f"statistic must be 'mean' or 'median', got {self.statistic!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "max_age_bars": self.max_age_bars,
            "constant_value": self.constant_value,
            "statistic": self.statistic,
            "add_missing_indicator": self.add_missing_indicator,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> MissingPolicySpec:
        return cls(
            kind=MissingPolicyKind(raw["kind"]),
            max_age_bars=None if raw.get("max_age_bars") is None else int(str(raw["max_age_bars"])),
            constant_value=None if raw.get("constant_value") is None else float(str(raw["constant_value"])),
            statistic=str(raw.get("statistic", "mean")),
            add_missing_indicator=bool(raw.get("add_missing_indicator", False)),
        )


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Pure metadata for one feature. Immutable and fully JSON-serializable
    -- this is what participates in `FeatureRegistry.fingerprint` and gets
    embedded (per computed feature) into a research dataset manifest's
    lineage section.

    `(name, version)` is the feature's identity. Bumping `version` (rather
    than mutating an already-registered spec in place, which
    `FeatureRegistry.register` refuses) is the only sanctioned way to
    change a feature's definition -- this is what makes "modifying feature
    configuration changes the dataset ID" true by construction: the
    dataset builder's fingerprint is derived from the exact
    `(name, version)` set and parameters used, never from a mutable
    feature object.
    """

    name: str
    version: str
    description: str
    category: FeatureCategory
    required_inputs: tuple[str, ...]
    """Raw column names (e.g. `("close",)`, `("high", "low", "close")`) this
    feature's `compute` reads directly from `FeatureContext.base_df` (or, for
    cross-asset/macro features, the relevant aux-data table). Distinct from
    `feature_dependencies` below -- this is about raw input columns, not
    other features."""
    source_symbols: tuple[str, ...]
    source_timeframe: Timeframe
    output_dtype: str
    lookback_bars: int
    warmup_bars: int
    availability_delay: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    null_policy: MissingPolicySpec = field(default_factory=MissingPolicySpec)
    deterministic_params: dict[str, object] = field(default_factory=dict)
    feature_dependencies: tuple[str, ...] = ()
    """Names of OTHER registered features (not raw OHLCV columns) this
    feature's `compute` reads via `FeatureContext.resolved_features`. Used
    by `FeatureRegistry.resolve_dependency_order` for topological
    ordering/cycle detection -- see its module docstring."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureSpec.name must not be empty")
        if not self.version:
            raise ValueError("FeatureSpec.version must not be empty")
        if self.lookback_bars < 0:
            raise ValueError(f"lookback_bars must be >= 0, got {self.lookback_bars}")
        if self.warmup_bars < 0:
            raise ValueError(f"warmup_bars must be >= 0, got {self.warmup_bars}")
        if self.availability_delay < pd.Timedelta(0):
            raise ValueError(f"availability_delay must be >= 0, got {self.availability_delay}")
        if self.name in self.feature_dependencies:
            raise ValueError(f"Feature {self.name!r} cannot depend on itself")

    @property
    def qualified_name(self) -> str:
        return f"{self.name}@{self.version}"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category.value,
            "required_inputs": list(self.required_inputs),
            "source_symbols": list(self.source_symbols),
            "source_timeframe": self.source_timeframe.value,
            "output_dtype": self.output_dtype,
            "lookback_bars": self.lookback_bars,
            "warmup_bars": self.warmup_bars,
            "availability_delay_seconds": self.availability_delay.total_seconds(),
            "null_policy": self.null_policy.to_json_dict(),
            "deterministic_params": self.deterministic_params,
            "feature_dependencies": list(self.feature_dependencies),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureSpec:
        return cls(
            name=str(raw["name"]),
            version=str(raw["version"]),
            description=str(raw["description"]),
            category=FeatureCategory(raw["category"]),
            required_inputs=tuple(str(s) for s in _as_list(raw.get("required_inputs") or [])),
            source_symbols=tuple(str(s) for s in _as_list(raw["source_symbols"])),
            source_timeframe=Timeframe(raw["source_timeframe"]),
            output_dtype=str(raw["output_dtype"]),
            lookback_bars=int(str(raw["lookback_bars"])),
            warmup_bars=int(str(raw["warmup_bars"])),
            availability_delay=pd.Timedelta(seconds=float(str(raw["availability_delay_seconds"]))),
            null_policy=MissingPolicySpec.from_json_dict(_as_dict(raw["null_policy"])),
            deterministic_params=_as_dict(raw["deterministic_params"]),
            feature_dependencies=tuple(str(s) for s in _as_list(raw.get("feature_dependencies") or [])),
        )

    def fingerprint(self) -> str:
        """A deterministic sha256 of this spec's JSON representation, with
        keys sorted so field-declaration order never affects the result."""
        payload = json.dumps(self.to_json_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["FeatureCategory", "FeatureSpec", "MissingPolicyKind", "MissingPolicySpec"]
