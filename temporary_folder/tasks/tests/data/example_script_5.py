import os
import sys
import json
from math import (
    sqrt as square_root,
    pow as power,
    ceil as ceiling,
    floor
)
from random import randint, shuffle as mix
from datetime import datetime as date, timedelta

import xml.etree.ElementTree as ET

from collections import defaultdict as defdict, OrderedDict as odict

from urllib.parse import urlparse


def parse_url(url):
    result = urlparse(url)
    return result


def calculate_square_root(number):
    return square_root(number)


def generate_random_number():
    return randint(1, 100)


if __name__ == "__main__":
    url = "https://www.example.com"
    number = 16

    parsed_url = parse_url(url)
    root_value = calculate_square_root(number)
    random_number = generate_random_number()

    print("Parsed URL:", parsed_url)
    print("Square root of", number, "is", root_value)
    print("Random number:", random_number)

    current_time = date.now()
    future_time = current_time + timedelta(days=1)
    print("Current time:", current_time)
    print("Future time:", future_time)
