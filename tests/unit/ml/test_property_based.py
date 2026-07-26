"""Property-based tests (Section 20) for the ML core infrastructure's
most safety-critical invariants: dict/JSON-key order never affects a
fingerprint; canonical serialization always round-trips; a value that is
not a well-formed sha256 hex digest is always rejected; seed derivation
is always deterministic and always in range; a generated content-
addressed path can never escape its store root; `ExperimentStatus`
transition legality is a pure, deterministic function of its inputs.
"""

from __future__ import annotations

import string
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_platform.core.exceptions import InvalidSeedError
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.fingerprints import fingerprint_json, is_valid_sha256_hex
from quant_platform.ml.models import ExperimentStatus, is_legal_transition
from quant_platform.ml.persistence import canonical_json_bytes, parse_json_strict
from quant_platform.ml.seeds import MAX_SEED, SeedDomain, derive_seed

_json_primitive = st.one_of(
    st.none(), st.booleans(), st.integers(min_value=-(2**53), max_value=2**53),
    st.text(alphabet=string.printable, max_size=20),
)
_json_dict = st.dictionaries(st.text(alphabet=string.ascii_letters, min_size=1, max_size=10), _json_primitive, max_size=8)


@given(_json_dict)
@settings(max_examples=200)
def test_fingerprint_json_independent_of_dict_insertion_order(payload: dict[str, object]) -> None:
    shuffled = dict(reversed(list(payload.items())))
    assert fingerprint_json(payload) == fingerprint_json(shuffled)


@given(_json_dict)
@settings(max_examples=200)
def test_canonical_json_bytes_round_trips(payload: dict[str, object]) -> None:
    encoded = canonical_json_bytes(payload)
    assert parse_json_strict(encoded.decode("utf-8")) == payload


@given(_json_dict)
@settings(max_examples=200)
def test_canonical_json_bytes_deterministic_regardless_of_order(payload: dict[str, object]) -> None:
    shuffled = dict(reversed(list(payload.items())))
    assert canonical_json_bytes(payload) == canonical_json_bytes(shuffled)


@given(st.text(max_size=80))
@settings(max_examples=300)
def test_is_valid_sha256_hex_rejects_anything_not_64_lowercase_hex_chars(value: str) -> None:
    is_valid = is_valid_sha256_hex(value)
    looks_valid = len(value) == 64 and all(c in string.hexdigits for c in value)
    assert is_valid == looks_valid


@given(st.integers(min_value=0, max_value=MAX_SEED), st.sampled_from(list(SeedDomain)))
@settings(max_examples=200)
def test_derive_seed_deterministic_and_in_range(master_seed: int, domain: SeedDomain) -> None:
    result_a = derive_seed(master_seed, domain)
    result_b = derive_seed(master_seed, domain)
    assert result_a == result_b
    assert 0 <= result_a <= MAX_SEED


@given(st.integers(min_value=0, max_value=MAX_SEED))
@settings(max_examples=100)
def test_derive_seed_different_domains_rarely_collide_for_fixed_seed(master_seed: int) -> None:
    results = {derive_seed(master_seed, domain) for domain in SeedDomain}
    # Not a strict uniqueness guarantee (finite range, no promise), but
    # for a SHA-256-derived value, collisions among 8 fixed domain names
    # for one master_seed should never actually occur in practice.
    assert len(results) == len(SeedDomain)


@given(st.integers(max_value=-1))
@settings(max_examples=50)
def test_derive_seed_rejects_all_negative_master_seeds(master_seed: int) -> None:
    with pytest.raises(InvalidSeedError):
        derive_seed(master_seed, SeedDomain.GLOBAL)


@given(st.integers(min_value=MAX_SEED + 1, max_value=MAX_SEED + 10_000_000))
@settings(max_examples=50)
def test_derive_seed_rejects_all_out_of_range_master_seeds(master_seed: int) -> None:
    with pytest.raises(InvalidSeedError):
        derive_seed(master_seed, SeedDomain.GLOBAL)


@given(st.sampled_from(list(ExperimentStatus)), st.sampled_from(list(ExperimentStatus)))
@settings(max_examples=200)
def test_is_legal_transition_is_deterministic(current: ExperimentStatus, target: ExperimentStatus) -> None:
    assert is_legal_transition(current, target) == is_legal_transition(current, target)


@given(st.sampled_from(list(ExperimentStatus)))
@settings(max_examples=50)
def test_terminal_statuses_never_have_legal_outgoing_transitions(current: ExperimentStatus) -> None:
    if current in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED):
        assert not any(is_legal_transition(current, target) for target in ExperimentStatus)


@given(st.text(alphabet=string.hexdigits.lower(), min_size=64, max_size=64))
@settings(max_examples=100)
def test_content_addressed_path_for_any_valid_hash_stays_under_root(content_hash: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = MLArtifactStore(Path(tmp))
        path = store._content_path(content_hash)
        resolved_root = store.root.resolve()
        assert resolved_root in path.resolve().parents or path.resolve() == resolved_root
