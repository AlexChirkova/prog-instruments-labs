import os
import pytest
import tempfile

from cryptography.hazmat.primitives.asymmetric import rsa
from unittest.mock import mock_open, patch

from rsa import RSA


@pytest.mark.parametrize("key_length", [16, 32, 64])
def test_encrypt_different_key_lengths(key_length):
    private_key, public_key = RSA.generate_rsa_keys()
    symmetric_key = b"x" * key_length

    encrypted = RSA.encrypt_symmetric_key(symmetric_key, public_key)
    decrypted = RSA.decrypt_symmetric_key(encrypted, private_key)

    assert decrypted == symmetric_key


def test_generate_rsa_keys_types():
    private_key, public_key = RSA.generate_rsa_keys()
    assert isinstance(private_key, rsa.RSAPrivateKey)
    assert isinstance(public_key, rsa.RSAPublicKey)


def test_serialize_and_get_private_key_roundtrip():
    private_key, _ = RSA.generate_rsa_keys()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "private.pem")
        RSA.serialize_private_key(private_key, path)

        assert os.path.exists(path)

        loaded_private = RSA.get_private_key(path)
        assert isinstance(loaded_private, rsa.RSAPrivateKey)
        assert (
            loaded_private.private_numbers().public_numbers
            == private_key.private_numbers().public_numbers
        )


def test_serialize_private_key_mock():
    private_key, _ = RSA.generate_rsa_keys()

    with patch("builtins.open", mock_open()) as mock_file:
        RSA.serialize_private_key(private_key, "test.pem")
        mock_file.assert_called_once_with("test.pem", "wb")


def test_serialize_and_get_public_key_roundtrip():
    private_key, public_key = RSA.generate_rsa_keys()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "public.pem")
        RSA.serialize_public_key(public_key, path)

        assert os.path.exists(path)

        loaded_public = RSA.get_public_key(path)
        assert isinstance(loaded_public, rsa.RSAPublicKey)
        assert loaded_public.public_numbers() == public_key.public_numbers()


def test_encrypt_decrypt_symmetric_key_roundtrip():
    private_key, public_key = RSA.generate_rsa_keys()
    symmetric_key = b"test_symmetric_key_32bytes!!"
    encrypted = RSA.encrypt_symmetric_key(symmetric_key, public_key)
    assert isinstance(encrypted, bytes)
    assert encrypted != symmetric_key

    decrypted = RSA.decrypt_symmetric_key(encrypted, private_key)
    assert decrypted == symmetric_key
