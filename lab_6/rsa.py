from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes


class RSA:
    @staticmethod
    def generate_rsa_keys() -> tuple:
        """
        Generating an RSA key pair (private and public)
        :return: RSA private and public keys
        """
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def serialize_private_key(private_key: rsa.RSAPrivateKey, file_path: str) -> None:
        """
        Serializing the RSA private key to a file
        :param private_key: the RSA private key
        :param file_path: path to the file to save the key
        :return: None
        """
        with open(file_path, "wb") as file:
            file.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

    @staticmethod
    def serialize_public_key(public_key: rsa.RSAPublicKey, file_path: str) -> None:
        """
        Serializing the RSA public key to a file
        :param public_key: the RSA public key
        :param file_path: path to the file to save the key
        :return:None
        """
        with open(file_path, "wb") as file:
            file.write(
                public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )

    @staticmethod
    def get_private_key(file_path: str) -> rsa.RSAPrivateKey:
        """
        Getting the RSA private key from a file
        :param file_path: the path to the file that contains the key
        :return: the RSA private key
        """
        with open(file_path, "rb") as file:
            private_key = serialization.load_pem_private_key(file.read(), password=None)
        return private_key

    @staticmethod
    def get_public_key(file_path: str) -> rsa.RSAPublicKey:
        """
        Getting the RSA public key from a file
        :param file_path: the path to the file that contains the key
        :return: the RSA public key
        """
        with open(file_path, "rb") as file:
            public_key = serialization.load_pem_public_key(file.read())
        return public_key

    @staticmethod
    def encrypt_symmetric_key(
        symmetric_key: bytes, public_key: rsa.RSAPublicKey
    ) -> bytes:
        """
        Encrypting the symmetric key using the RSA public key
        :param symmetric_key: symmetric encryption key
        :param public_key: RSA public key for encryption
        :return: encrypted symmetric key
        """
        c_symmetric_key = public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return c_symmetric_key

    @staticmethod
    def decrypt_symmetric_key(
        c_symmetric_key: bytes, private_key: rsa.RSAPrivateKey
    ) -> bytes:
        """
        Decrypting a symmetric key using an RSA private key
        :param c_symmetric_key: encrypted symmetric key
        :param private_key: RSA private key for decryption
        :return: decrypted symmetric key
        """
        dc_symmetric_key = private_key.decrypt(
            c_symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return dc_symmetric_key
