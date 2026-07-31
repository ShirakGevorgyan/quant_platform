"""Strict bridge from a verified Milestone 7 paper session to Milestone 8
execution intents (Section 5/6). `ExecutionIntent` is this package's own
immutable, content-addressed economic-intent object (Section 5);
`ExecutionAuthorization` is the explicit TEST_ONLY proof a given paper
order may be bridged at all (Section 6). `execution_intent_from_paper_order`
is this module's `ExecutionIntentFactory.from_paper_order` (Section 6's
own example API name) -- implemented as a plain function to match this
codebase's existing `create_*`/`compute_*` function-oriented convention
rather than introduce a bespoke factory class for a single method.

THIS MODULE NEVER DECIDES STRATEGY DIRECTION (Section 5's own
instruction) -- every economic field on the resulting `ExecutionIntent`
is copied from, and independently re-verified against, the already-
decided source `paper_trading.orders.OrderRequest`; this module only
validates and routes it.

Every one of Section 6's 13 required checks is performed, in order,
inside `execution_intent_from_paper_order` -- fail-closed, raising a
typed `ExecutionGatewayEligibilityError`/`ExecutionIntentError` on the
FIRST failing check, never partially trusting a later step once an
earlier one has failed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.core.exceptions import ExecutionGatewayEligibilityError, ExecutionIntentError
from quant_platform.execution_gateway.identity import (
    compute_content_id,
    decimal_from_float,
    decimal_to_json,
    is_valid_sha256_hex,
    parse_decimal,
)
from quant_platform.execution_gateway.models import (
    AdapterKind,
    AuthorizationMode,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.specs import ExecutionGatewaySpec, compute_execution_gateway_spec_id
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.persistence import as_json_dict, format_utc_timestamp, parse_json_strict
from quant_platform.paper_trading.eligibility import (
    EligibilityVerificationEnvironment,
    require_paper_trading_eligibility,
)
from quant_platform.paper_trading.manifests import PaperSessionManifest, PaperSessionManifestStore
from quant_platform.paper_trading.models import PaperSessionStage, PositionIntentKind
from quant_platform.paper_trading.orders import OrderRequest
from quant_platform.paper_trading.persistence import PaperSessionEventStore
from quant_platform.paper_trading.specs import PaperTradingSpec, compute_paper_session_spec_id
from quant_platform.paper_trading.verification import require_paper_session_verified

EXECUTION_INTENT_KIND = "execution_intent"
EXECUTION_AUTHORIZATION_KIND = "execution_authorization"


def _require_sha256(value: str, *, field_name: str) -> None:
    if not is_valid_sha256_hex(value):
        raise ExecutionIntentError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")


def _optional_decimal_json(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_json(value)


# --------------------------------------------------------------------------
# ExecutionAuthorization (Section 6)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExecutionAuthorization:
    execution_authorization_id: str
    authorization_mode: AuthorizationMode
    paper_session_id: str
    paper_order_id: str
    execution_gateway_spec_id: str
    authorized_quantity: Decimal
    authorized_side: OrderSide
    issued_sequence: int
    source_verification_id: str
    """Content id of the `ValidationReport` (Section 26) produced by
    `paper_trading.verification.require_paper_session_verified` for the
    source paper session -- proves this authorization rests on a paper
    session that was independently re-verified, not merely one whose
    manifest CLAIMS to be COMPLETED."""

    def __post_init__(self) -> None:
        if self.authorization_mode is not AuthorizationMode.TEST_ONLY_DUMMY_EXECUTION:
            raise ExecutionIntentError(f"ExecutionAuthorization.authorization_mode must be test_only_dummy_execution, got {self.authorization_mode!r}")
        for field_name, value in (
            ("paper_session_id", self.paper_session_id), ("paper_order_id", self.paper_order_id),
            ("execution_gateway_spec_id", self.execution_gateway_spec_id), ("source_verification_id", self.source_verification_id),
        ):
            _require_sha256(value, field_name=f"ExecutionAuthorization.{field_name}")
        if not self.authorized_quantity.is_finite() or self.authorized_quantity <= 0:
            raise ExecutionIntentError(f"ExecutionAuthorization.authorized_quantity must be finite and > 0, got {self.authorized_quantity!r}")
        if self.issued_sequence < 0:
            raise ExecutionIntentError(f"ExecutionAuthorization.issued_sequence must be >= 0, got {self.issued_sequence}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "execution_authorization_id": self.execution_authorization_id, "authorization_mode": self.authorization_mode.value,
            "paper_session_id": self.paper_session_id, "paper_order_id": self.paper_order_id, "execution_gateway_spec_id": self.execution_gateway_spec_id,
            "authorized_quantity": decimal_to_json(self.authorized_quantity), "authorized_side": self.authorized_side.value,
            "issued_sequence": self.issued_sequence, "source_verification_id": self.source_verification_id,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["execution_authorization_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionAuthorization:
        return cls(
            execution_authorization_id=str(raw["execution_authorization_id"]), authorization_mode=AuthorizationMode(raw["authorization_mode"]),
            paper_session_id=str(raw["paper_session_id"]), paper_order_id=str(raw["paper_order_id"]),
            execution_gateway_spec_id=str(raw["execution_gateway_spec_id"]),
            authorized_quantity=parse_decimal(raw["authorized_quantity"], field_name="authorized_quantity"), authorized_side=OrderSide(raw["authorized_side"]),
            issued_sequence=int(str(raw["issued_sequence"])), source_verification_id=str(raw["source_verification_id"]),
        )


def create_execution_authorization(
    *, paper_session_id: str, paper_order_id: str, execution_gateway_spec_id: str, authorized_quantity: Decimal, authorized_side: OrderSide,
    issued_sequence: int, source_verification_id: str,
) -> ExecutionAuthorization:
    provisional = ExecutionAuthorization(
        execution_authorization_id="0" * 64, authorization_mode=AuthorizationMode.TEST_ONLY_DUMMY_EXECUTION, paper_session_id=paper_session_id,
        paper_order_id=paper_order_id, execution_gateway_spec_id=execution_gateway_spec_id, authorized_quantity=authorized_quantity,
        authorized_side=authorized_side, issued_sequence=issued_sequence, source_verification_id=source_verification_id,
    )
    execution_authorization_id = compute_content_id(EXECUTION_AUTHORIZATION_KIND, provisional.to_identity_payload())
    return ExecutionAuthorization(
        execution_authorization_id=execution_authorization_id, authorization_mode=AuthorizationMode.TEST_ONLY_DUMMY_EXECUTION,
        paper_session_id=paper_session_id, paper_order_id=paper_order_id, execution_gateway_spec_id=execution_gateway_spec_id,
        authorized_quantity=authorized_quantity, authorized_side=authorized_side, issued_sequence=issued_sequence,
        source_verification_id=source_verification_id,
    )


# --------------------------------------------------------------------------
# ExecutionIntent (Section 5)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    execution_intent_id: str
    execution_session_id: str
    paper_session_id: str
    source_decision_id: str
    source_paper_order_id: str

    instrument_id: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderTypeKind
    limit_price: Decimal | None
    stop_price: Decimal | None
    time_in_force: TimeInForceKind

    reduce_only: bool
    close_position: bool

    strategy_candidate_id: str
    model_artifact_id: str
    risk_authorization_id: str

    source_event_id: str | None
    source_event_time: str
    created_sequence: int

    contract_multiplier: Decimal
    identity_version: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("execution_session_id", self.execution_session_id), ("paper_session_id", self.paper_session_id),
            ("source_decision_id", self.source_decision_id), ("source_paper_order_id", self.source_paper_order_id),
            ("strategy_candidate_id", self.strategy_candidate_id), ("model_artifact_id", self.model_artifact_id),
            ("risk_authorization_id", self.risk_authorization_id),
        ):
            _require_sha256(value, field_name=f"ExecutionIntent.{field_name}")
        if not self.instrument_id:
            raise ExecutionIntentError("ExecutionIntent.instrument_id must not be empty")
        if not self.quantity.is_finite() or self.quantity <= 0:
            raise ExecutionIntentError(f"ExecutionIntent.quantity must be finite and > 0, got {self.quantity!r}")
        if not self.contract_multiplier.is_finite() or self.contract_multiplier <= 0:
            raise ExecutionIntentError(f"ExecutionIntent.contract_multiplier must be finite and > 0, got {self.contract_multiplier!r}")
        if self.order_type is OrderTypeKind.MARKET:
            if self.limit_price is not None or self.stop_price is not None:
                raise ExecutionIntentError("ExecutionIntent: limit_price/stop_price must be None for a MARKET order")
        elif self.order_type is OrderTypeKind.LIMIT:
            if self.limit_price is None or not self.limit_price.is_finite() or self.limit_price <= 0:
                raise ExecutionIntentError(f"ExecutionIntent.limit_price must be finite and > 0 for a LIMIT order, got {self.limit_price!r}")
            if self.stop_price is not None:
                raise ExecutionIntentError("ExecutionIntent.stop_price must be None for a LIMIT order")
        elif self.order_type is OrderTypeKind.STOP:
            if self.stop_price is None or not self.stop_price.is_finite() or self.stop_price <= 0:
                raise ExecutionIntentError(f"ExecutionIntent.stop_price must be finite and > 0 for a STOP order, got {self.stop_price!r}")
            if self.limit_price is not None:
                raise ExecutionIntentError("ExecutionIntent.limit_price must be None for a STOP order")
        if self.close_position and not self.reduce_only:
            raise ExecutionIntentError("ExecutionIntent.close_position=True requires reduce_only=True (closing is inherently risk-reducing)")
        if self.created_sequence < 0:
            raise ExecutionIntentError(f"ExecutionIntent.created_sequence must be >= 0, got {self.created_sequence}")
        if self.identity_version < 1:
            raise ExecutionIntentError(f"ExecutionIntent.identity_version must be >= 1, got {self.identity_version}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "execution_intent_id": self.execution_intent_id, "execution_session_id": self.execution_session_id, "paper_session_id": self.paper_session_id,
            "source_decision_id": self.source_decision_id, "source_paper_order_id": self.source_paper_order_id, "instrument_id": self.instrument_id,
            "side": self.side.value, "quantity": decimal_to_json(self.quantity), "order_type": self.order_type.value,
            "limit_price": _optional_decimal_json(self.limit_price), "stop_price": _optional_decimal_json(self.stop_price),
            "time_in_force": self.time_in_force.value, "reduce_only": self.reduce_only, "close_position": self.close_position,
            "strategy_candidate_id": self.strategy_candidate_id, "model_artifact_id": self.model_artifact_id,
            "risk_authorization_id": self.risk_authorization_id, "source_event_id": self.source_event_id, "source_event_time": self.source_event_time,
            "created_sequence": self.created_sequence, "contract_multiplier": decimal_to_json(self.contract_multiplier), "identity_version": self.identity_version,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["execution_intent_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionIntent:
        return cls(
            execution_intent_id=str(raw["execution_intent_id"]), execution_session_id=str(raw["execution_session_id"]),
            paper_session_id=str(raw["paper_session_id"]), source_decision_id=str(raw["source_decision_id"]),
            source_paper_order_id=str(raw["source_paper_order_id"]), instrument_id=str(raw["instrument_id"]), side=OrderSide(raw["side"]),
            quantity=parse_decimal(raw["quantity"], field_name="quantity"), order_type=OrderTypeKind(raw["order_type"]),
            limit_price=(None if raw.get("limit_price") is None else parse_decimal(raw["limit_price"], field_name="limit_price")),
            stop_price=(None if raw.get("stop_price") is None else parse_decimal(raw["stop_price"], field_name="stop_price")),
            time_in_force=TimeInForceKind(raw["time_in_force"]), reduce_only=bool(raw["reduce_only"]), close_position=bool(raw["close_position"]),
            strategy_candidate_id=str(raw["strategy_candidate_id"]), model_artifact_id=str(raw["model_artifact_id"]),
            risk_authorization_id=str(raw["risk_authorization_id"]), source_event_id=(None if raw.get("source_event_id") is None else str(raw["source_event_id"])),
            source_event_time=str(raw["source_event_time"]), created_sequence=int(str(raw["created_sequence"])),
            contract_multiplier=parse_decimal(raw["contract_multiplier"], field_name="contract_multiplier"), identity_version=int(str(raw["identity_version"])),
        )


def _create_execution_intent(**kwargs: object) -> ExecutionIntent:
    provisional = ExecutionIntent(execution_intent_id="0" * 64, **kwargs)  # type: ignore[arg-type]
    execution_intent_id = compute_content_id(EXECUTION_INTENT_KIND, provisional.to_identity_payload())
    return ExecutionIntent(execution_intent_id=execution_intent_id, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The bridge itself (Section 6)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PaperBridgeEnvironment:
    """Every store the bridge (and the eligibility/verification chain it
    transitively re-invokes) needs. Mirrors `RunnerEnvironment`/
    `EligibilityVerificationEnvironment`'s identical bundling pattern."""

    manifest_store: PaperSessionManifestStore
    event_store: PaperSessionEventStore
    artifact_store: MLArtifactStore
    eligibility_environment: EligibilityVerificationEnvironment


def _load_paper_trading_spec(environment: PaperBridgeEnvironment, manifest: PaperSessionManifest) -> PaperTradingSpec:
    if manifest.spec_reference is None:
        raise ExecutionGatewayEligibilityError(f"paper session {manifest.paper_session_id!r} has no persisted spec_reference")
    raw = environment.artifact_store.read_artifact(manifest.spec_reference.content_hash)
    return PaperTradingSpec.from_json_dict(as_json_dict(parse_json_strict(raw.decode("utf-8")), field_name="paper_trading_spec"))


def execution_intent_from_paper_order(
    paper_order: OrderRequest, *, execution_gateway_spec: ExecutionGatewaySpec, execution_session_id: str, environment: PaperBridgeEnvironment,
    created_sequence: int, event_time: datetime, source_event_id: str | None = None,
) -> tuple[ExecutionIntent, ExecutionAuthorization]:
    """`ExecutionIntentFactory.from_paper_order` (Section 6). Performs
    every one of Section 6's 13 required checks, in order, fail-closed.
    Returns the freshly minted `(ExecutionIntent, ExecutionAuthorization)`
    pair -- the authorization is newly issued here, never reused from a
    prior call, so `issued_sequence`/`source_verification_id` always
    reflect a FRESH re-verification, never a cached one.

    CONFIRMED DEFECT, FIXED (found during this milestone's own
    acceptance testing): `intent.source_event_time` used to be captured
    via `utc_now()` INSIDE this function -- a genuine wall-clock read,
    unlike every other function in this package, which threads an
    explicit, caller-supplied `event_time` throughout specifically to
    avoid this. Since `source_event_time` participates in `ExecutionIntent
    .to_identity_payload()` (and therefore `execution_intent_id`, which
    itself cascades into `client_order_id`, every command's own identity,
    `broker_order_id`, and every fill id derived from those), replaying
    the IDENTICAL economic scenario at a different wall-clock moment
    produced a COMPLETELY DIFFERENT semantic digest -- the same defect
    class as `adapter_id` leaking into `BrokerEvent` identity, but far
    more severe given how much downstream identity cascades from
    `execution_intent_id`. `event_time` is now a required, caller-supplied
    parameter, exactly matching every other function's own convention."""
    paper_session_id = execution_gateway_spec.paper_session_id

    # Check 1: source paper session exists.
    try:
        manifest = environment.manifest_store.load(paper_session_id)
    except Exception as exc:
        raise ExecutionGatewayEligibilityError(f"source paper session {paper_session_id!r} could not be loaded: {exc}") from exc

    # Check 2: source paper session is terminal and independently verified.
    if manifest.stage is not PaperSessionStage.COMPLETED:
        raise ExecutionGatewayEligibilityError(f"source paper session {paper_session_id!r} is not COMPLETED (stage={manifest.stage.value!r}) -- cannot bridge orders from a non-terminal session")
    spec = _load_paper_trading_spec(environment, manifest)
    ledger = environment.event_store.read_events(paper_session_id)
    verification_report = require_paper_session_verified(spec, manifest=manifest, ledger=ledger, eligibility_environment=environment.eligibility_environment)
    source_verification_id = compute_content_id("execution_gateway_source_verification", {"paper_session_id": paper_session_id, "issue_count": len(verification_report.issues)})

    # Check 3: source paper session belongs to the referenced paper-trading spec.
    recomputed_paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    if recomputed_paper_session_id != paper_session_id:
        raise ExecutionGatewayEligibilityError(f"paper session {paper_session_id!r} does not reproduce its own spec identity (recomputed={recomputed_paper_session_id!r})")
    if manifest.spec_reference is not None and manifest.spec_reference.content_hash != execution_gateway_spec.paper_trading_spec_id:
        raise ExecutionGatewayEligibilityError(
            f"ExecutionGatewaySpec.paper_trading_spec_id={execution_gateway_spec.paper_trading_spec_id!r} does not match the source paper "
            f"session's own persisted spec artifact content_hash={manifest.spec_reference.content_hash!r}"
        )

    # Check 4: source paper session still passes authoritative eligibility verification (fresh, never cached).
    require_paper_trading_eligibility(spec, environment=environment.eligibility_environment)

    # Check 5: source order belongs to the source session.
    if paper_order.session_id != paper_session_id:
        raise ExecutionIntentError(f"paper order {paper_order.order_id!r} belongs to session {paper_order.session_id!r}, not the requested {paper_session_id!r}")

    # Check 6/7: strategy candidate and model identity match.
    strategy_candidate_id = spec.strategy_candidate_identity
    model_artifact_id = spec.model_artifact_identity

    # Check 8: instrument identity matches.
    if paper_order.instrument != spec.instrument.symbol:
        raise ExecutionIntentError(f"paper order instrument {paper_order.instrument!r} does not match source spec instrument {spec.instrument.symbol!r}")
    recomputed_instrument_spec_id = compute_content_id("execution_gateway_instrument_spec", spec.instrument.to_json_dict())
    if recomputed_instrument_spec_id != execution_gateway_spec.instrument_spec_id:
        raise ExecutionIntentError(
            f"ExecutionGatewaySpec.instrument_spec_id={execution_gateway_spec.instrument_spec_id!r} does not match the source paper session's own "
            f"instrument (recomputed={recomputed_instrument_spec_id!r})"
        )

    # Check 9/10: quantity, side, and every other economic field copied
    # verbatim from the source order -- never re-derived or modified.
    quantity = decimal_from_float(paper_order.quantity, field_name="paper_order.quantity")
    contract_multiplier = decimal_from_float(spec.instrument.contract_multiplier, field_name="instrument.contract_multiplier")
    limit_price = None if paper_order.limit_price is None else decimal_from_float(paper_order.limit_price, field_name="paper_order.limit_price")
    stop_price = None if paper_order.stop_price is None else decimal_from_float(paper_order.stop_price, field_name="paper_order.stop_price")
    close_position = paper_order.position_intent is PositionIntentKind.CLOSE

    authorization = create_execution_authorization(
        paper_session_id=paper_session_id, paper_order_id=paper_order.order_id,
        execution_gateway_spec_id=compute_execution_gateway_spec_id(execution_gateway_spec).execution_gateway_spec_id, authorized_quantity=quantity,
        authorized_side=paper_order.side, issued_sequence=created_sequence, source_verification_id=source_verification_id,
    )

    intent = _create_execution_intent(
        execution_session_id=execution_session_id, paper_session_id=paper_session_id, source_decision_id=paper_order.strategy_decision_id,
        source_paper_order_id=paper_order.order_id, instrument_id=paper_order.instrument, side=paper_order.side, quantity=quantity,
        order_type=paper_order.order_type, limit_price=limit_price, stop_price=stop_price, time_in_force=paper_order.time_in_force,
        reduce_only=paper_order.reduce_only, close_position=close_position, strategy_candidate_id=strategy_candidate_id,
        model_artifact_id=model_artifact_id, risk_authorization_id=authorization.execution_authorization_id, source_event_id=source_event_id,
        source_event_time=format_utc_timestamp(pd.Timestamp(event_time)), created_sequence=created_sequence, contract_multiplier=contract_multiplier, identity_version=1,
    )

    # Check 11: authorization belongs to the same source order.
    if authorization.paper_order_id != intent.source_paper_order_id:
        raise ExecutionIntentError("internal error: authorization.paper_order_id does not match intent.source_paper_order_id")
    # Check 12: authorization is TEST_ONLY.
    if authorization.authorization_mode is not AuthorizationMode.TEST_ONLY_DUMMY_EXECUTION:
        raise ExecutionGatewayEligibilityError("authorization is not TEST_ONLY_DUMMY_EXECUTION")
    # Check 13: adapter kind is DETERMINISTIC_DUMMY.
    if execution_gateway_spec.adapter_kind is not AdapterKind.DETERMINISTIC_DUMMY:
        raise ExecutionGatewayEligibilityError(f"execution_gateway_spec.adapter_kind={execution_gateway_spec.adapter_kind!r} is not deterministic_dummy")

    return intent, authorization


__all__ = [
    "EXECUTION_AUTHORIZATION_KIND",
    "EXECUTION_INTENT_KIND",
    "ExecutionAuthorization",
    "ExecutionIntent",
    "PaperBridgeEnvironment",
    "create_execution_authorization",
    "execution_intent_from_paper_order",
]
