import sklearn
import numpy as np
import pandas as pd
import csv

FILENAME = ""


def preprocess(file_to_open):
    training_data = pandas.read_csv(FILENAME)
    with open(file_to_open, "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    print(training_data);
    return None


if __name__ == "__main__":
    preprocess(FILENAME)
