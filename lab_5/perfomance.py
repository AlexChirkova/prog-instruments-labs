import logging

import constants as const

from camellia import *
from file_worker import *
from rsa import *


logger = logging.getLogger(__name__)


class Performance:
    @staticmethod
    def generate_keys() -> None:
        try:
            asymmetric_keys = RSA.generate_rsa_keys()
            logger.info("Asymmetric keys were generated successfully.")
            FileWorker.serialize_private_key(
                asymmetric_keys[0], const.PATH_TO_PRIVATE_KEY
            )
            FileWorker.serialize_public_key(
                asymmetric_keys[1], const.PATH_TO_PUBLIC_KEY
            )
            logger.info("Asymmetric keys were wrote in file successfully.")
            size = ""
            while size not in {"16", "24", "32"}:
                size = input("Key length must be 16, 24, or 32 bytes: ")

            symmetric_key = Camellia.generate_camellia_key(int(size))
            logger.info("Symmetric key was generated successfully.")
            FileWorker.write_txt_file(
                RSA.encrypt_symmetric_key(symmetric_key, asymmetric_keys[1]),
                const.PATH_TO_SYM_KEY,
            )
            logger.info("Symmetric key was wrote in file successfully.")

        except Exception as e:
            logger.error(f"Error: {e}")

    @staticmethod
    def encrypt_text() -> None:
        try:
            if not (
                FileWorker.read_txt_file(const.PATH_TO_SYM_KEY)
                and FileWorker.read_txt_file(const.PATH_TO_PRIVATE_KEY)
                and FileWorker.read_txt_file(const.PATH_TO_PUBLIC_KEY)
            ):
                logger.error("There is no key in the directory.")
                return

            symmetric_key = RSA.decrypt_symmetric_key(
                FileWorker.read_txt_file(const.PATH_TO_SYM_KEY),
                FileWorker.get_private_key(const.PATH_TO_PRIVATE_KEY),
            )
            plaintext = FileWorker.read_txt_file(const.PATH_TO_PLAINTEXT)
            ciphertext = Camellia.camellia_encrypt(symmetric_key, plaintext)
            logger.info("Text was encrypted successfully.")
            FileWorker.write_txt_file(ciphertext, const.PATH_TO_CIPHERTEXT)
            logger.info("Encrypted text was wrote to the file successfully.")

        except Exception as e:
            logger.error(f"Error: {e}")

    @staticmethod
    def decrypt_text() -> None:
        try:
            if not (
                FileWorker.read_txt_file(const.PATH_TO_SYM_KEY)
                and FileWorker.read_txt_file(const.PATH_TO_PRIVATE_KEY)
                and FileWorker.read_txt_file(const.PATH_TO_PUBLIC_KEY)
            ):
                logger.info("There is no key in the directory.")
                return

            symmetric_key = RSA.decrypt_symmetric_key(
                FileWorker.read_txt_file(const.PATH_TO_SYM_KEY),
                FileWorker.get_private_key(const.PATH_TO_PRIVATE_KEY),
            )
            ciphertext = FileWorker.read_txt_file(const.PATH_TO_CIPHERTEXT)
            encrypted_text = Camellia.camellia_decrypt(symmetric_key, ciphertext)
            logger.info("Text was decrypted successfully.")
            FileWorker.write_txt_file(encrypted_text, const.PATH_TO_ENCRYPTED_TEXT)
            logger.info("Decrypted text was wrote to the file successfully.")

        except Exception as e:
            logger.error(f"Error: {e}")
