import sklearn
import numpy as np
import csv

FILENAME = ""


def preprocess(file_to_open):
    with open(file_to_open, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    return None


if __name__ == "__main__":
    preprocess(FILENAME)
