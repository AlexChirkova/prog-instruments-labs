from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


class FileWorker:
    @staticmethod
    def read_txt_file(file_path: str) -> bytes:
        """
        Reading a text file as bytes.
        :param file_path: path to the text file
        :return: content of text file as bytes
        """
        try:
            with open(file_path, "rb") as file:
                return file.read()
        except Exception as e:
            print(f"Error: {e}")
            return b""

    @staticmethod
    def write_txt_file(data: bytes, file_path: str) -> None:
        """
        Writing bytes to a file
        :param data: data to write into the file as bytes
        :param file_path: the path to the file to save the data
        :return: None
        """
        try:
            with open(file_path, "wb") as file:
                file.write(data)
        except Exception as e:
            print(f"Error: {e}")

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
