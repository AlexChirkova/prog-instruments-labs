import re
import csv

from const import *
from checksum import *


def is_row_valid(row: dict) -> bool:
    """
    Проверяет строку на валидность.
    :param row: словарь вида 'тип_данных': 'значение'
    :return: соответствует ли строка паттерну
    """
    for data_type in row.keys():
        if not re.fullmatch(REGEX[data_type], row[data_type]):
            return False
    return True


def find_invalid_rows(file_path: str, delimiter: str = ";") -> List[int]:
    invalid_rows = []

    with open(file_path, "r", encoding="utf-16") as file:
        rows = csv.DictReader(file, delimiter=delimiter)

        for i, row in enumerate(rows):
            if is_row_valid(row):
                continue
            invalid_rows.append(i)
    return invalid_rows


def main() -> None:
    """
    Находит невалидные строки,
    получает их контрольную сумму
    :return: None
    """
    invalid_rows = find_invalid_rows(CSV_PATH)
    checksum = calculate_checksum(invalid_rows)
    print(checksum)
    serialize_result(VARIANT, checksum)


if __name__ == "__main__":
    main()
