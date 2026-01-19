import constants as const

from camellia import *
from file_worker import *
from rsa import *


class Performance:
    @staticmethod
    def generate_keys() -> None:
        try:
            asymmetric_keys = RSA.generate_rsa_keys()
            FileWorker.serialize_private_key(
                asymmetric_keys[0], const.PATH_TO_PRIVATE_KEY
            )
            FileWorker.serialize_public_key(
                asymmetric_keys[1], const.PATH_TO_PUBLIC_KEY
            )
            print("Asymmetric keys were generated end wrote in file successfully.")
            size = ""
            while size not in {"16", "24", "32"}:
                size = input("Key length must be 16, 24, or 32 bytes: ")

            symmetric_key = Camellia.generate_camellia_key(int(size))
            FileWorker.write_txt_file(
                RSA.encrypt_symmetric_key(symmetric_key, asymmetric_keys[1]),
                const.PATH_TO_SYM_KEY,
            )
            print("Symmetric key was generated end wrote in file successfully.")

        except Exception as e:
            print(f"Error: {e}")

    @staticmethod
    def encrypt_text() -> None:
        try:
            if not (
                FileWorker.read_txt_file(const.PATH_TO_SYM_KEY)
                and FileWorker.read_txt_file(const.PATH_TO_PRIVATE_KEY)
                and FileWorker.read_txt_file(const.PATH_TO_PUBLIC_KEY)
            ):
                print("You have to generate keys at first.")
                return

            symmetric_key = RSA.decrypt_symmetric_key(
                FileWorker.read_txt_file(const.PATH_TO_SYM_KEY),
                FileWorker.get_private_key(const.PATH_TO_PRIVATE_KEY),
            )
            plaintext = FileWorker.read_txt_file(const.PATH_TO_PLAINTEXT)
            ciphertext = Camellia.camellia_encrypt(symmetric_key, plaintext)
            FileWorker.write_txt_file(ciphertext, const.PATH_TO_CIPHERTEXT)
            print("Text was encrypted and wrote to the file.")

        except Exception as e:
            print(f"Error: {e}")

    @staticmethod
    def decrypt_text() -> None:
        try:
            if not (
                FileWorker.read_txt_file(const.PATH_TO_SYM_KEY)
                and FileWorker.read_txt_file(const.PATH_TO_PRIVATE_KEY)
                and FileWorker.read_txt_file(const.PATH_TO_PUBLIC_KEY)
            ):
                print("You must have keys to decrypt text.")
                return

            symmetric_key = RSA.decrypt_symmetric_key(
                FileWorker.read_txt_file(const.PATH_TO_SYM_KEY),
                FileWorker.get_private_key(const.PATH_TO_PRIVATE_KEY),
            )
            ciphertext = FileWorker.read_txt_file(const.PATH_TO_CIPHERTEXT)
            encrypted_text = Camellia.camellia_decrypt(symmetric_key, ciphertext)
            FileWorker.write_txt_file(encrypted_text, const.PATH_TO_ENCRYPTED_TEXT)
            print("Text was decrypted and wrote to the file.")

        except Exception as e:
            print(f"Error: {e}")
