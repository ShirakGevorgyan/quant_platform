from __future__ import annotations

from quant_platform.ml.fingerprints import combine_fingerprints, fingerprint_json, is_valid_sha256_hex


class TestFingerprintJson:
    def test_deterministic(self) -> None:
        assert fingerprint_json({"a": 1, "b": 2}) == fingerprint_json({"b": 2, "a": 1})

    def test_sensitive_to_value_change(self) -> None:
        assert fingerprint_json({"a": 1}) != fingerprint_json({"a": 2})

    def test_is_64_char_hex(self) -> None:
        fp = fingerprint_json({"a": 1})
        assert len(fp) == 64
        assert is_valid_sha256_hex(fp)


class TestCombineFingerprints:
    def test_order_sensitive(self) -> None:
        a, b = "a" * 64, "b" * 64
        assert combine_fingerprints(a, b) != combine_fingerprints(b, a)

    def test_deterministic(self) -> None:
        a, b = "a" * 64, "b" * 64
        assert combine_fingerprints(a, b) == combine_fingerprints(a, b)


class TestIsValidSha256Hex:
    def test_valid(self) -> None:
        assert is_valid_sha256_hex("a" * 64)
        assert is_valid_sha256_hex("A" * 64)  # case-insensitive per implementation

    def test_wrong_length(self) -> None:
        assert not is_valid_sha256_hex("a" * 63)
        assert not is_valid_sha256_hex("a" * 65)

    def test_non_hex_chars(self) -> None:
        assert not is_valid_sha256_hex("z" * 64)

    def test_empty_string(self) -> None:
        assert not is_valid_sha256_hex("")
