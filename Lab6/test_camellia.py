import pytest

from unittest.mock import patch

from camellia import Camellia


@pytest.mark.parametrize("key_size", [16, 24, 32])
def test_generate_camellia_key_parametrized(key_size):
    key = Camellia.generate_camellia_key(key_size)
    assert isinstance(key, bytes)
    assert len(key) == key_size


def test_generate_camellia_key_invalid_size():
    with pytest.raises(ValueError):
        Camellia.generate_camellia_key(10)


def test_camellia_encrypt_decrypt_roundtrip():
    key = Camellia.generate_camellia_key(16)
    plaintext = b"example plaintext for camellia"

    ciphertext = Camellia.camellia_encrypt(key, plaintext)
    assert isinstance(ciphertext, bytes)
    assert len(ciphertext) > len(plaintext)

    decrypted = Camellia.camellia_decrypt(key, ciphertext)
    assert decrypted == plaintext


def test_camellia_encrypt_invalid_key_length():
    key = b"short"
    plaintext = b"data"
    with pytest.raises(ValueError):
        Camellia.camellia_encrypt(key, plaintext)


def test_camellia_encrypt_with_stub():
    with patch("os.urandom", return_value=b"0" * 16):
        key = b"k" * 16
        plaintext = b"test"
        ciphertext = Camellia.camellia_encrypt(key, plaintext)
        assert ciphertext[:16] == b"0" * 16


def test_camellia_decrypt_invalid_key_length():
    key = b"short"
    ciphertext = b"\x00" * 32
    with pytest.raises(ValueError):
        Camellia.camellia_decrypt(key, ciphertext)
