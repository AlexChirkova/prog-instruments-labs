from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes


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
