import json
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

TRAIN = "./data/train_raw.csv"
DEV = "./data/dev_raw.csv"
TEST = "./data/test_raw.csv"

stemmer = SnowballStemmer("english", ignore_stopwords=True)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def main():
    training_data = preprocess(TRAIN)
    dev_data = preprocess(DEV)
    test_data = preprocess(TEST, test=True)

    train(training_data, dev_data, test_data)
    return None


def ranges(age):
    if 14 <= age <= 16:
        return "14, 16"
    elif 24 <= age <= 26:
        return "24, 26"
    elif 34 <= age <= 36:
        return "34, 36"
    elif 44 <= age <= 46:
        return "44, 46"
    return "?"


def preprocess(file_path, test=False):
    data = pd.read_csv(file_path, header=None)
    data = data[[2, 6]]
    data.columns = ["age", "text"]
    if not test:
        data["age"] = data["age"].map(ranges)
        data = data[data.age != "?"]
    return data


def train(training_data, dev_data, test_data):
    nb_clf = Pipeline([('vect', StemmedCountVectorizer(stop_words="english")),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultinomialNB()), ])

    svm_clf = Pipeline([('vect', CountVectorizer(stop_words="english")),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                              alpha=1e-3, random_state=42,
                                              max_iter=5, tol=None,
                                              n_jobs=6)), ])

    lr_clf = Pipeline([('vect', StemmedCountVectorizer(ngram_range=(1, 2))),
                       ('tfidf', TfidfTransformer(use_idf=True)),
                       ('clf', LogisticRegression()), ])

    parameters = {'vect__stop_words': ("english", None),
                  'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False), }

    # gs_clf = GridSearchCV(lr_clf, parameters, n_jobs=6)
    # gs_clf.fit(training_data["text"], training_data["age"])

    # nb_clf.fit(training_data["text"], training_data["age"])
    # svm_clf.fit(training_data["text"], training_data["age"])
    lr_clf.fit(training_data["text"], training_data["age"])

    # print("NB: {}".format(nb_clf.score(dev_data["text"], dev_data["age"])))
    # print("SVM: {}".format(svm_clf.score(dev_data["text"], dev_data["age"])))
    print("LR: {}".format(lr_clf.score(dev_data["text"], dev_data["age"])))
    # print("GS: {}".format(gs_clf.score(dev_data["text"], dev_data["age"])))

    # for param_name in sorted(parameters.keys()):
    #     print("{}: {}".format(param_name, gs_clf.best_params_[param_name]))

    # print(evaluate(predictions, dev_data))
    # print(evaluate(predictions2, dev_data))
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
