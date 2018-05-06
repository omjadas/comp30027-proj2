import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

TRAIN = "train_raw.csv"
DEV = "dev_raw.csv"
TEST = "test_raw.csv"

CLASSES = []


def main():
    training_data = preprocess(TRAIN)
    dev_data = preprocess(DEV)
    test_data = preprocess(TEST, test=True)

    training_data["age"] = training_data["age"].map(ranges)

    train(training_data, dev_data, test_data)
    return None


def ranges(age):
    if age <= 16:
        return "14-16"
    elif age <= 26:
        return "24-26"
    elif age <= 36:
        return "34-36"
    elif age <= 46:
        return "44-46"
    return "?"


def preprocess(file_path, test=False):
    data = pd.read_csv(file_path, header=None)
    data = data[[2, 6]]
    data.columns = ["age", "text"]
    return data


def train(training_data, dev_data, test_data):
    vectoriser = TfidfVectorizer(stop_words='english')
    train_vector = vectoriser.fit_transform(training_data["text"])
    dev_vector = vectoriser.transform(dev_data["text"])
    test_vector = vectoriser.transform(test_data.iloc[:, 1])

    clf = MultinomialNB().fit(train_vector, training_data["age"])

    predictions = clf.predict(dev_vector)

    print(predictions)
    return None


def evaluate():
    pass


def predict():
    pass


if __name__ == "__main__":
    main()
