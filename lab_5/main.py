import argparse
import logging

from perfomance import *


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)


if __name__ == "__main__":
    logging.info("Enter in the program")
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-gen", "--generation", help="Запускает режим генерации ключей")
    group.add_argument("-enc", "--encryption", help="Запускает режим шифрования")
    group.add_argument("-dec", "--decryption", help="Запускает режим дешифрования")

    args = parser.parse_args()

    if args.generation is not None:
        Performance.generate_keys()

    elif args.encryption is not None:
        Performance.encrypt_text()

    else:
        Performance.decrypt_text()

    logging.info("End of the program")