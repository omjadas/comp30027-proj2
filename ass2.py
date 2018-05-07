import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

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
        return json.dumps([14, 16])
    elif age <= 26:
        return json.dumps([24, 26])
    elif age <= 36:
        return json.dumps([34, 36])
    elif age <= 46:
        return json.dumps([44, 46])
    return None


def preprocess(file_path, test=False):
    data = pd.read_csv(file_path, header=None)
    data = data[[2, 6]]
    data.columns = ["age", "text"]
    return data


def train(training_data, dev_data, test_data):
    # vectoriser = TfidfVectorizer(stop_words="english")
    # train_vector = vectoriser.fit_transform(training_data["text"])
    # dev_vector = vectoriser.transform(dev_data["text"])
    # test_vector = vectoriser.transform(test_data["text"])

    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()), ])

    # clf = MultinomialNB().fit(train_vector, training_data["age"])
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None)), ])
    # text_clf = SGDClassifier(loss='hinge', penalty='l2',
    #                          alpha=1e-3, random_state=42,
    # max_iter=5, tol=None).fit(train_vector, training_data["age"])
    clf.fit(training_data["text"], training_data["age"])
    predictions = clf.predict(dev_data["text"])
    text_clf.fit(training_data["text"], training_data["age"])
    predictions2 = text_clf.predict(dev_data["text"])

    print(evaluate(predictions, dev_data))
    print(evaluate(predictions2, dev_data))
    return None


def evaluate(predictions, data):
    correct = 0
    total = 0
    for i in predictions:
        if json.loads(i)[0] <= data.iloc[total, 0] <= json.loads(i)[1]:
            correct += 1
        total += 1
    return correct / total


def predict():
    pass


if __name__ == "__main__":
    main()
