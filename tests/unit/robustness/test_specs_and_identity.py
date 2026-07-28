"""Milestone 6, Section 16: `RobustnessSpec`/`compute_robustness_identity`
identity determinism and immutable-spec validation. Two independently
constructed specs with identical field values must produce the same
`robustness_id`; any field change must change it. Frozen dataclasses are
exercised for their `__post_init__` fail-closed validation, not merely
trusted."""

from __future__ import annotations

import pytest

from quant_platform.core.exceptions import RobustnessValidationError
from quant_platform.core.types import Timeframe
from quant_platform.robustness.models import (
    BootstrapMethodKind,
    MultipleTestingCorrectionKind,
    ReturnSeriesKind,
)
from quant_platform.robustness.specs import (
    DEFAULT_PERTURBATIONS,
    DEFAULT_PROMOTION_GATES,
    DEFAULT_REGIME_DEFINITIONS,
    DEFAULT_STRESS_SCENARIOS,
    BootstrapSpec,
    PerturbationSpec,
    PromotionPolicySpec,
    RobustnessSpec,
    StabilityThresholds,
    compute_robustness_identity,
)


def _spec(**overrides: object) -> RobustnessSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "source_backtest_id": "a" * 64, "dataset_content_id": "b" * 64, "split_plan_fingerprint": "c" * 64,
        "instrument_identity": "XAUUSD", "bar_interval": Timeframe.H1, "return_series_kind": ReturnSeriesKind.STITCHED_BAR_NET,
        "bootstrap_spec": BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=500, confidence_level=0.95, block_length=10),
        "seed": 0, "multiple_testing_correction": MultipleTestingCorrectionKind.BENJAMINI_HOCHBERG, "strategy_family_id": None,
        "minimum_fold_count": 3, "minimum_trade_count": 30, "minimum_effective_sample_size": 30,
        "stability_thresholds": StabilityThresholds(
            minimum_profitable_fold_fraction=0.5, maximum_single_fold_profit_concentration=0.6,
            maximum_single_trade_profit_concentration=0.4, maximum_single_direction_profit_concentration=0.7,
        ),
        "stress_scenarios": DEFAULT_STRESS_SCENARIOS, "regime_definitions": DEFAULT_REGIME_DEFINITIONS,
        "promotion_policy": PromotionPolicySpec(gates=DEFAULT_PROMOTION_GATES),
    }
    defaults.update(overrides)
    return RobustnessSpec(**defaults)  # type: ignore[arg-type]


class TestRobustnessIdentityDeterminism:
    def test_identical_specs_produce_identical_robustness_id(self) -> None:
        assert compute_robustness_identity(_spec()).robustness_id == compute_robustness_identity(_spec()).robustness_id

    def test_json_round_trip_preserves_identity(self) -> None:
        spec = _spec()
        roundtripped = RobustnessSpec.from_json_dict(spec.to_json_dict())
        assert compute_robustness_identity(spec).robustness_id == compute_robustness_identity(roundtripped).robustness_id

    @pytest.mark.parametrize(
        ("label", "override"),
        [
            ("source_backtest_id", {"source_backtest_id": "d" * 64}),
            ("dataset_content_id", {"dataset_content_id": "e" * 64}),
            ("split_plan_fingerprint", {"split_plan_fingerprint": "f" * 64}),
            ("instrument_identity", {"instrument_identity": "EURUSD"}),
            ("bar_interval", {"bar_interval": Timeframe.H4}),
            ("return_series_kind", {"return_series_kind": ReturnSeriesKind.BENCHMARK_RELATIVE}),
            ("bootstrap_method", {"bootstrap_spec": BootstrapSpec(method=BootstrapMethodKind.MOVING_BLOCK, repetitions=500, confidence_level=0.95, block_length=10)}),
            ("bootstrap_repetitions", {"bootstrap_spec": BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=1000, confidence_level=0.95, block_length=10)}),
            ("bootstrap_confidence_level", {"bootstrap_spec": BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=500, confidence_level=0.90, block_length=10)}),
            ("bootstrap_block_length", {"bootstrap_spec": BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=500, confidence_level=0.95, block_length=20)}),
            ("seed", {"seed": 1}),
            ("multiple_testing_correction", {"multiple_testing_correction": MultipleTestingCorrectionKind.BONFERRONI}),
            ("strategy_family_id", {"strategy_family_id": "1" * 64}),
            ("minimum_fold_count", {"minimum_fold_count": 5}),
            ("minimum_trade_count", {"minimum_trade_count": 50}),
            ("minimum_effective_sample_size", {"minimum_effective_sample_size": 50}),
            (
                "stability_thresholds",
                {
                    "stability_thresholds": StabilityThresholds(
                        minimum_profitable_fold_fraction=0.6, maximum_single_fold_profit_concentration=0.6,
                        maximum_single_trade_profit_concentration=0.4, maximum_single_direction_profit_concentration=0.7,
                    ),
                },
            ),
        ],
        ids=lambda p: p if isinstance(p, str) else "override",
    )
    def test_changing_any_identity_relevant_field_changes_robustness_id(self, label: str, override: dict[str, object]) -> None:
        baseline = compute_robustness_identity(_spec()).robustness_id
        changed = compute_robustness_identity(_spec(**override)).robustness_id
        assert baseline != changed, f"changing {label!r} did not change robustness_id"

    def test_schema_version_does_not_affect_identity(self) -> None:
        """`compute_robustness_identity` drops `schema_version` from the
        identity payload -- a schema bump alone must never change an
        already-computed robustness_id for otherwise-identical settings."""
        payload_a = _spec().to_identity_payload()
        payload_b = _spec(schema_version=1).to_identity_payload()
        assert payload_a == payload_b
        assert "schema_version" not in payload_a


class TestUnorderedCollectionFieldsDoNotAffectIdentity:
    """Release-audit regression (Milestone 6 final audit, Section 1):
    `stress_scenarios`/`regime_definitions`/`perturbations`/`promotion_
    policy.gates` are semantically UNORDERED SETS -- each element's own
    uniqueness key (name/dimension/axis) is already enforced by
    `__post_init__`, and no downstream code anywhere depends on declared
    order. Before the fix, `RobustnessSpec.to_json_dict()` serialized
    these fields in caller-supplied order, so two specs describing the
    EXACT SAME set of scenarios/regimes/perturbations/gates in a
    different order silently produced DIFFERENT `robustness_id`s --
    breaking the platform's own documented "robustness_id is a pure
    function of every field" guarantee, and inconsistent with
    `multiple_testing.StrategyFamily.to_identity_payload`'s own,
    already-correct `sorted(...)` handling of its candidate-id tuples.
    These tests fail against the pre-fix code (verified directly before
    applying the fix) and pass after it."""

    def test_stress_scenarios_reordered_produces_identical_id(self) -> None:
        forward = compute_robustness_identity(_spec(stress_scenarios=DEFAULT_STRESS_SCENARIOS)).robustness_id
        reversed_id = compute_robustness_identity(_spec(stress_scenarios=tuple(reversed(DEFAULT_STRESS_SCENARIOS)))).robustness_id
        assert forward == reversed_id

    def test_regime_definitions_reordered_produces_identical_id(self) -> None:
        forward = compute_robustness_identity(_spec(regime_definitions=DEFAULT_REGIME_DEFINITIONS)).robustness_id
        reversed_id = compute_robustness_identity(_spec(regime_definitions=tuple(reversed(DEFAULT_REGIME_DEFINITIONS)))).robustness_id
        assert forward == reversed_id

    def test_perturbations_reordered_produces_identical_id(self) -> None:
        forward = compute_robustness_identity(_spec(perturbations=DEFAULT_PERTURBATIONS)).robustness_id
        reversed_id = compute_robustness_identity(_spec(perturbations=tuple(reversed(DEFAULT_PERTURBATIONS)))).robustness_id
        assert forward == reversed_id

    def test_perturbation_relative_deltas_reordered_produces_identical_id(self) -> None:
        from quant_platform.robustness.models import PerturbationAxisKind

        forward_spec = PerturbationSpec(axis=PerturbationAxisKind.PROBABILITY_THRESHOLD, relative_deltas=(-0.1, -0.05, 0.05, 0.1))
        reversed_spec = PerturbationSpec(axis=PerturbationAxisKind.PROBABILITY_THRESHOLD, relative_deltas=(0.1, 0.05, -0.05, -0.1))
        forward = compute_robustness_identity(_spec(perturbations=(forward_spec,))).robustness_id
        reversed_id = compute_robustness_identity(_spec(perturbations=(reversed_spec,))).robustness_id
        assert forward == reversed_id

    def test_promotion_policy_gates_reordered_produces_identical_id(self) -> None:
        forward = compute_robustness_identity(_spec(promotion_policy=PromotionPolicySpec(gates=DEFAULT_PROMOTION_GATES))).robustness_id
        reversed_id = compute_robustness_identity(_spec(promotion_policy=PromotionPolicySpec(gates=tuple(reversed(DEFAULT_PROMOTION_GATES))))).robustness_id
        assert forward == reversed_id

    def test_genuinely_different_stress_scenario_set_still_changes_identity(self) -> None:
        """Sanity counterpart: canonicalizing ORDER must not accidentally
        canonicalize away a real CONTENT difference."""
        narrowed = tuple(s for s in DEFAULT_STRESS_SCENARIOS if s.name != "combined_adverse")
        baseline = compute_robustness_identity(_spec()).robustness_id
        narrowed_id = compute_robustness_identity(_spec(stress_scenarios=narrowed)).robustness_id
        assert baseline != narrowed_id


class TestToJsonDictPreservesDeclaredOrder:
    """Release-audit regression, found DURING the audit itself (a
    regression in the audit's OWN first fix for `TestUnorderedCollection
    FieldsDoNotAffectIdentity` above): that first fix sorted `stress_
    scenarios`/`regime_definitions`/`perturbations`/`promotion_policy.
    gates`/`PerturbationSpec.relative_deltas` directly inside `to_json_
    dict`, not only inside `to_identity_payload`. `to_json_dict` is the
    DURABLE, round-tripped representation: `sensitivity.py`/`stress.py`/
    `regimes.py` each build their result tuples by iterating the
    corresponding `RobustnessSpec` field POSITIONALLY, and `robustness.
    verification.verify_robustness` reloads a spec from its persisted
    `to_json_dict` output (via `from_json_dict`) to independently
    RECOMPUTE those same reports for comparison against what the forward
    pass persisted. Sorting inside `to_json_dict` meant a reloaded spec's
    fields came back in a DIFFERENT order than however the forward pass
    originally declared them, so the recomputed reports' result tuples
    came back reordered too -- three spurious CRITICAL "recomputed result
    does not match the persisted artifact" findings on every real run
    (`stress_report_mismatch`, `sensitivity_report_mismatch`, `regime_
    report_mismatch`), caught by `tests/integration/
    test_robustness_real_model_acceptance.py` failing. These tests fail
    against that first (over-eager) fix and pass against the corrected
    one, where canonicalization lives ONLY in `to_identity_payload`."""

    def test_to_json_dict_preserves_reversed_stress_scenario_order(self) -> None:
        reversed_scenarios = tuple(reversed(DEFAULT_STRESS_SCENARIOS))
        spec = _spec(stress_scenarios=reversed_scenarios)
        assert [s["name"] for s in spec.to_json_dict()["stress_scenarios"]] == [s.name for s in reversed_scenarios]

    def test_to_json_dict_preserves_reversed_regime_definition_order(self) -> None:
        reversed_regimes = tuple(reversed(DEFAULT_REGIME_DEFINITIONS))
        spec = _spec(regime_definitions=reversed_regimes)
        assert [r["dimension"] for r in spec.to_json_dict()["regime_definitions"]] == [r.dimension.value for r in reversed_regimes]

    def test_to_json_dict_preserves_reversed_perturbation_order(self) -> None:
        reversed_perturbations = tuple(reversed(DEFAULT_PERTURBATIONS))
        spec = _spec(perturbations=reversed_perturbations)
        assert [p["axis"] for p in spec.to_json_dict()["perturbations"]] == [p.axis.value for p in reversed_perturbations]

    def test_to_json_dict_preserves_reversed_relative_deltas_order(self) -> None:
        from quant_platform.robustness.models import PerturbationAxisKind

        reversed_spec = PerturbationSpec(axis=PerturbationAxisKind.PROBABILITY_THRESHOLD, relative_deltas=(0.1, 0.05, -0.05, -0.1))
        assert reversed_spec.to_json_dict()["relative_deltas"] == [0.1, 0.05, -0.05, -0.1]

    def test_to_json_dict_preserves_reversed_promotion_gate_order(self) -> None:
        reversed_gates = tuple(reversed(DEFAULT_PROMOTION_GATES))
        spec = _spec(promotion_policy=PromotionPolicySpec(gates=reversed_gates))
        gates_json = spec.to_json_dict()["promotion_policy"]["gates"]  # type: ignore[index]
        assert [g["name"] for g in gates_json] == [g.name for g in reversed_gates]

    def test_full_json_round_trip_preserves_declared_order_everywhere(self) -> None:
        """The end-to-end property that actually matters: a spec built,
        persisted, and reloaded (exactly what `verify_robustness` does)
        must come back with every collection field in the SAME declared
        order it started with -- not merely the same set."""
        reversed_scenarios = tuple(reversed(DEFAULT_STRESS_SCENARIOS))
        reversed_regimes = tuple(reversed(DEFAULT_REGIME_DEFINITIONS))
        reversed_perturbations = tuple(reversed(DEFAULT_PERTURBATIONS))
        reversed_gates = tuple(reversed(DEFAULT_PROMOTION_GATES))
        spec = _spec(
            stress_scenarios=reversed_scenarios, regime_definitions=reversed_regimes, perturbations=reversed_perturbations,
            promotion_policy=PromotionPolicySpec(gates=reversed_gates),
        )
        roundtripped = RobustnessSpec.from_json_dict(spec.to_json_dict())
        assert [s.name for s in roundtripped.stress_scenarios] == [s.name for s in reversed_scenarios]
        assert [r.dimension for r in roundtripped.regime_definitions] == [r.dimension for r in reversed_regimes]
        assert [p.axis for p in roundtripped.perturbations] == [p.axis for p in reversed_perturbations]
        assert [g.name for g in roundtripped.promotion_policy.gates] == [g.name for g in reversed_gates]


class TestExplicitEmptyPerturbationsSurvivesJsonRoundTrip:
    """Release-audit regression (Milestone 6 final audit, Section 1):
    `RobustnessSpec.perturbations=()` is a legal, distinct spec value
    (`compute_sensitivity_report` gracefully produces `axis_results=()`
    for it -- "run with zero declared sensitivity axes"). Before the fix,
    `RobustnessSpec.from_json_dict` used `raw.get("perturbations") or
    [...]`, which treats a persisted empty list the same as an ENTIRELY
    ABSENT key and silently substitutes `DEFAULT_PERTURBATIONS` for both --
    so serializing an explicit `perturbations=()` spec and reloading it
    (as every crash/resume and verification path does) silently produced a
    DIFFERENT spec with 3 unrequested perturbation axes and therefore a
    DIFFERENT `robustness_id`. This test fails against the pre-fix code
    (verified directly before applying the fix) and passes after it."""

    def test_explicit_empty_perturbations_round_trips_as_empty(self) -> None:
        spec = _spec(perturbations=())
        assert spec.to_json_dict()["perturbations"] == []
        roundtripped = RobustnessSpec.from_json_dict(spec.to_json_dict())
        assert roundtripped.perturbations == ()

    def test_explicit_empty_perturbations_round_trip_preserves_robustness_id(self) -> None:
        spec = _spec(perturbations=())
        roundtripped = RobustnessSpec.from_json_dict(spec.to_json_dict())
        assert compute_robustness_identity(spec).robustness_id == compute_robustness_identity(roundtripped).robustness_id

    def test_omitted_perturbations_key_still_defaults(self) -> None:
        """Backward compatibility: a payload written before this field
        existed (key wholly absent, not merely empty) must still default
        to DEFAULT_PERTURBATIONS, not fail closed."""
        spec = _spec()
        raw = dict(spec.to_json_dict())
        del raw["perturbations"]
        restored = RobustnessSpec.from_json_dict(raw)
        assert restored.perturbations == DEFAULT_PERTURBATIONS


class TestBootstrapSpecValidation:
    def test_repetitions_below_100_rejected(self) -> None:
        with pytest.raises(RobustnessValidationError, match="repetitions"):
            BootstrapSpec(method=BootstrapMethodKind.IID, repetitions=99, confidence_level=0.95)

    def test_block_length_required_for_stationary(self) -> None:
        with pytest.raises(RobustnessValidationError, match="block_length"):
            BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=500, confidence_level=0.95, block_length=None)

    def test_block_length_forbidden_for_iid(self) -> None:
        with pytest.raises(RobustnessValidationError, match="block_length"):
            BootstrapSpec(method=BootstrapMethodKind.IID, repetitions=500, confidence_level=0.95, block_length=10)

    def test_confidence_level_must_be_in_unit_interval(self) -> None:
        with pytest.raises(RobustnessValidationError):
            BootstrapSpec(method=BootstrapMethodKind.IID, repetitions=500, confidence_level=1.0)


class TestRobustnessSpecValidation:
    def test_stress_scenarios_must_include_base_cost(self) -> None:
        non_base_only = tuple(s for s in DEFAULT_STRESS_SCENARIOS if s.name != "base_cost")
        with pytest.raises(RobustnessValidationError, match="base_cost"):
            _spec(stress_scenarios=non_base_only)

    def test_regime_definitions_must_not_be_empty(self) -> None:
        with pytest.raises(RobustnessValidationError):
            _spec(regime_definitions=())

    def test_duplicate_regime_dimensions_rejected(self) -> None:
        with pytest.raises(RobustnessValidationError, match="unique"):
            _spec(regime_definitions=(DEFAULT_REGIME_DEFINITIONS[0], DEFAULT_REGIME_DEFINITIONS[0]))

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(RobustnessValidationError):
            _spec(seed=-1)
