import sklearn
import numpy as np
import pandas as pd

FILENAME = "dev_raw.csv"


def preprocess(file_to_open):
    training_data = pd.read_csv(FILENAME, header=None)

    print(training_data);
    return None


if __name__ == "__main__":
    preprocess(FILENAME)
