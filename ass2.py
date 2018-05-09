import json
import time
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

TRAIN = "./data/train_raw.csv"
DEV = "./data/dev_raw.csv"
TEST = "./data/test_raw.csv"

stemmer = SnowballStemmer("english", ignore_stopwords=True)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def main():
    start = time.time()

    training_data = preprocess(TRAIN)
    dev_data = preprocess(DEV)
    test_data = preprocess(TEST, test=True)

    train(training_data, dev_data, test_data)

    end = time.time()
    # print("\n{} seconds".format(end - start))
    return None


def ranges(age):
    if 14 <= age <= 16:
        return "14-16"
    elif 24 <= age <= 26:
        return "24-26"
    elif 34 <= age <= 36:
        return "34-36"
    elif 44 <= age <= 46:
        return "44-46"
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
    parameters = {"vect__ngram_range": [(1, 1), (1, 2)],
                  "vect__stop_words": ("english", None),
                  "tfidf__use_idf": (True, False)}

    nb_clf = Pipeline([("vect", StemmedCountVectorizer(stop_words="english")),
                       ("tfidf", TfidfTransformer()),
                       ("clf", MultinomialNB()), ])

    svm_clf = Pipeline([("vect", StemmedCountVectorizer(stop_words="english")),
                        ("tfidf", TfidfTransformer()),
                        ("clf", SGDClassifier(loss="hinge", penalty="l2",
                                              alpha=1e-3, random_state=42,
                                              max_iter=5, tol=None,
                                              n_jobs=6)), ])

    lr_clf = Pipeline([("vect", StemmedCountVectorizer(ngram_range=(1, 2))),
                       ("tfidf", TfidfTransformer(use_idf=True)),
                       ("clf", LogisticRegression()), ])

    gs_clf = GridSearchCV(lr_clf, parameters, n_jobs=6)

    # fit(nb_clf, training_data)
    # fit(svm_clf, training_data)
    fit(lr_clf, training_data)
    # fit(gs_clf, training_data)

    # score(nb_clf, "NB", training_data, dev_data)
    # score(svm_clf, "SVM", training_data, dev_data)
    # score(lr_clf, "LR", training_data, dev_data)
    # gs_score(gs_clf, parameters, training_data, dev_data)

    predictions = lr_clf.predict(test_data["text"])
    output(predictions)
    return None


def fit(classifier, training_data):
    classifier.fit(training_data["text"], training_data["age"])
    return None


def score(classifier, classifier_name, training_data, dev_data):
    score = classifier.score(dev_data["text"], dev_data["age"])
    print("{}: {}".format(classifier_name, score))
    return None


def gs_score(classifier, parameters, training_data, dev_data):
    score(classifier, "GS", training_data, dev_data)
    for param_name in sorted(parameters.keys()):
        print("{}: {}".format(param_name, classifier.best_params_[param_name]))
    return None


def output(predictions):
    """Output the predictions to stdout in the form specified on Kaggle"""

    print("Id,Prediction")
    i = 1
    for prediction in predictions:
        print("3{},{}".format(i, prediction))
        i += 1
    return None


if __name__ == "__main__":
    main()
