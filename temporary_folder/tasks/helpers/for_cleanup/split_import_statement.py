import re


SEPARATORS = ["import", "as"]


def get_alias_names(alias_string):
    cleaned = re.sub(r"[()]", "", alias_string).replace("\n", "").split(",")
    cleaned = [name.strip() for name in cleaned if name.strip()]

    return cleaned


def split_import_statement(import_statement):
    words = import_statement.split()
    words.reverse()

    separator_detected = False
    specifiers = ""
    base = []
    for word in words:
        if word in SEPARATORS and not separator_detected:
            separator_detected = True
            base.append(word)
        elif separator_detected:
            base.append(word)
        else:
            specifiers = word + " " + specifiers

    base.reverse()
    base = " ".join(base).strip()

    specifiers = get_alias_names(specifiers)

    return base, specifiers


if __name__ == "__main__":
    import_statements = [
        "import os",
        "import source",
        "import random",
        "import sys",
        "import numpy as np",
        "from math import (\nsqrt,\nceil,\nfloor\n)",
        "from datetime import datetime, timedelta",
        "import sklearn",
        "from pandas.core import series as pd_series",
    ]

    extracted_data = []
    for statement in import_statements:
        extracted_data.append(split_import_statement(statement))
    for data in extracted_data:
        print(data)
