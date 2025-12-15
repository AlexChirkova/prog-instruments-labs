import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding


class Camellia:
    @staticmethod
    def generate_camellia_key(size: int) -> bytes:
        """
        Generating a random key for Camellia encryption
        :param size: size of key in bytes
        :return: random key
        """
        if size not in {16, 24, 32}:
            raise ValueError(
                "The size of key must be 16, 24, or 32 bytes for Camellia-128, 192, 256."
            )
        return os.urandom(size)

    @staticmethod
    def camellia_encrypt(key: bytes, plaintext: bytes) -> bytes:
        """
        Encrypts text using the Camellia algorithm in CBC mode
        :param key:the encryption key is 16, 24, or 32 bytes (128, 192, or 256 bits) long
        :param plaintext:text for encryption in the form of bytes
        :return: iv + encrypted text
        """
        if len(key) not in {16, 24, 32}:
            raise ValueError(
                "The key must be 16, 24, or 32 bytes for Camellia-128, 192, 256."
            )
        iv = os.urandom(16)

        padder = padding.PKCS7(128).padder()
        padded_plaintext = padder.update(plaintext) + padder.finalize()

        cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
        return iv + ciphertext

    @staticmethod
    def camellia_decrypt(key: bytes, ciphertext: bytes) -> bytes:
        """
        Decrypting cipher text using the Camellia algorithm in CBC mode
        :param key:the decryption key is 16, 24, or 32 bytes (128, 192, or 256 bits) long
        :param ciphertext:encrypted text
        :return:decrypted text
        """
        if len(key) not in {16, 24, 32}:
            raise ValueError(
                "The key must be 16, 24, or 32 bytes for Camellia-128, 192, 256."
            )
        iv = ciphertext[:16]
        actual_ciphertext = ciphertext[16:]

        cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
        return plaintext
