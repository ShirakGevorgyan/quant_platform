from __future__ import annotations

import os
import subprocess
import sys

import pytest

from quant_platform.core.exceptions import InvalidSeedError
from quant_platform.ml.seeds import MAX_SEED, SeedConfiguration, SeedDomain, derive_seed


class TestDeriveSeed:
    def test_deterministic_same_process(self) -> None:
        assert derive_seed(42, SeedDomain.MODEL_INIT) == derive_seed(42, SeedDomain.MODEL_INIT)

    def test_different_domains_produce_different_seeds(self) -> None:
        assert derive_seed(42, SeedDomain.MODEL_INIT) != derive_seed(42, SeedDomain.CROSS_VALIDATION)

    def test_different_master_seeds_produce_different_derived_seeds(self) -> None:
        assert derive_seed(1, SeedDomain.GLOBAL) != derive_seed(2, SeedDomain.GLOBAL)

    def test_accepts_string_domain(self) -> None:
        assert derive_seed(42, "model_init") == derive_seed(42, SeedDomain.MODEL_INIT)

    def test_result_within_valid_range(self) -> None:
        for domain in SeedDomain:
            result = derive_seed(0, domain)
            assert 0 <= result <= MAX_SEED

    def test_rejects_negative_master_seed(self) -> None:
        with pytest.raises(InvalidSeedError):
            derive_seed(-1, SeedDomain.GLOBAL)

    def test_rejects_out_of_range_master_seed(self) -> None:
        with pytest.raises(InvalidSeedError):
            derive_seed(MAX_SEED + 1, SeedDomain.GLOBAL)

    def test_rejects_bool_master_seed(self) -> None:
        with pytest.raises(InvalidSeedError):
            derive_seed(True, SeedDomain.GLOBAL)  # type: ignore[arg-type]

    def test_rejects_non_int_master_seed(self) -> None:
        with pytest.raises(InvalidSeedError):
            derive_seed(1.5, SeedDomain.GLOBAL)  # type: ignore[arg-type]

    def test_rejects_empty_domain_name(self) -> None:
        with pytest.raises(InvalidSeedError):
            derive_seed(1, "")

    def test_deterministic_across_processes(self) -> None:
        """Proves derivation does NOT depend on Python's randomized
        `hash()` (which varies per-process via PYTHONHASHSEED unless
        disabled) -- run in a fresh subprocess with a random hash seed
        and confirm the result matches this process's."""
        script = (
            "from quant_platform.ml.seeds import derive_seed, SeedDomain; "
            "print(derive_seed(42, SeedDomain.MODEL_INIT))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": "random"},
        )
        expected = derive_seed(42, SeedDomain.MODEL_INIT)
        assert int(result.stdout.strip()) == expected


class TestSeedConfiguration:
    def test_round_trip(self) -> None:
        config = SeedConfiguration(master_seed=123)
        assert SeedConfiguration.from_json_dict(config.to_json_dict()) == config

    def test_rejects_invalid_seed(self) -> None:
        with pytest.raises(InvalidSeedError):
            SeedConfiguration(master_seed=-1)

    def test_derive_matches_module_function(self) -> None:
        config = SeedConfiguration(master_seed=7)
        assert config.derive(SeedDomain.GLOBAL) == derive_seed(7, SeedDomain.GLOBAL)

    def test_random_for_is_reproducible(self) -> None:
        config = SeedConfiguration(master_seed=7)
        r1 = config.random_for(SeedDomain.DATA_OPERATIONS)
        r2 = config.random_for(SeedDomain.DATA_OPERATIONS)
        assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]

    def test_numpy_generator_for_is_reproducible(self) -> None:
        config = SeedConfiguration(master_seed=7)
        g1 = config.numpy_generator_for(SeedDomain.DATA_OPERATIONS)
        g2 = config.numpy_generator_for(SeedDomain.DATA_OPERATIONS)
        assert (g1.random(5) == g2.random(5)).all()

    def test_fingerprint_deterministic(self) -> None:
        assert SeedConfiguration(master_seed=1).fingerprint() == SeedConfiguration(master_seed=1).fingerprint()

    def test_fingerprint_changes_with_master_seed(self) -> None:
        assert SeedConfiguration(master_seed=1).fingerprint() != SeedConfiguration(master_seed=2).fingerprint()

    def test_has_no_descriptive_field_to_change(self) -> None:
        """Structural guarantee: `SeedConfiguration` has exactly two
        fields (`master_seed`, `schema_version`) -- there is no notes/
        description field whose change could accidentally leave the
        fingerprint (and thus experiment identity) unchanged while a
        human believes something changed, or vice versa."""
        field_names = set(SeedConfiguration.__dataclass_fields__)
        assert field_names == {"master_seed", "schema_version"}


def test_importing_seeds_module_does_not_consume_global_random_state() -> None:
    """Importing `ml.seeds` must not, by itself, draw from or reseed the
    global `random`/`numpy.random` state: seed both generators, draw a
    baseline sequence; reseed identically, import the module, then draw
    again -- the two sequences must be identical, proving the import
    consumed nothing from either generator in between."""
    script = (
        "import random, numpy as np\n"
        "random.seed(123); np.random.seed(123)\n"
        "baseline = ([random.random() for _ in range(3)], list(np.random.random(3)))\n"
        "random.seed(123); np.random.seed(123)\n"
        "import quant_platform.ml.seeds\n"
        "after_import = ([random.random() for _ in range(3)], list(np.random.random(3)))\n"
        "print(baseline == after_import)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True", result.stderr
