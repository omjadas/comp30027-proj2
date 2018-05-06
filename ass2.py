import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

TRAIN = "train_raw.csv"
DEV = "dev_raw.csv"
TEST = "test_raw.csv"


def main():
    training_data = preprocess(TRAIN)
    dev_data = preprocess(DEV)
    test_data = preprocess(TEST)

    train(training_data, dev_data, test_data)
    return None


def preprocess(file_path):
    data = pd.read_csv(TRAIN, header=None)
    data = data[data.iloc[:, -1] != '?']
    data = data[[2, 6]]
    return data


def train(training_data, dev_data, test_data):

    vectoriser = TfidfVectorizer(stop_words='english')
    train_vector = vectoriser.fit_transform(training_data.iloc[:, 1])
    dev_vector = vectoriser.transform(dev_data.iloc[:, 1])
    # test_vector = vectoriser.transform(test_data.iloc[:,1])

    print(train_vector.shape)
    print(dev_vector.shape)

    # clf.fit(training_data.iloc[:, -1], new_data)
    return None


def predict():
    pass


if __name__ == "__main__":
    main()
