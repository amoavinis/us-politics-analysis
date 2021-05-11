import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def evaluate(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")


def search_classifiers():

        # read data
        df = pd.read_pickle("data/preprocessed_gender_classifier_dataset.pkl")

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(df["processed_text"], df["gender"], random_state=0)

        # clfs to try
        clfs = [
                LogisticRegression(max_iter=1000, random_state=0),
                RandomForestClassifier(random_state=0),
                AdaBoostClassifier(
                        LogisticRegression(max_iter=1000, random_state=0),
                        random_state=0),
                ExtraTreesClassifier(random_state=0),
                GradientBoostingClassifier(random_state=0),
                MultinomialNB()]
        
        # train and evaluate
        print("Fitting classifiers on tweet text")
        for clf in clfs:
                pipeline = make_pipeline(
                        TfidfVectorizer(),
                        clf)
                print(f"\nFitting: {str(clf)}")
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


def search_features():

        # read data
        df = pd.read_pickle("data/preprocessed_gender_classifier_dataset.pkl")

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
                df[["processed_text", "processed_description", "processed_both"]],
                df["gender"],
                random_state=0)

        # define columns for training and testing
        train_columns = ["processed_text", "processed_description", "processed_both"]
        test_columns = ["processed_text", "processed_description", "processed_both"]

        # initialize accuracy
        accuracy = np.zeros((3,3))

        # train and test
        for i_train, train_col in enumerate(train_columns):
            clf = make_pipeline(
                TfidfVectorizer(),
                LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
            clf.fit(X_train[train_col], y_train)
        
            for j_test, test_col in enumerate(test_columns):
                y_pred = clf.predict(X_test[test_col])
                accuracy[i_train, j_test] = round(accuracy_score(y_test, y_pred), 3)
        
        # visualize results
        trained_on = ["Tweet text", "User description", "Both"]
        tested_on = ["Tweet text", "User description", "Both"]

        fig, ax = plt.subplots()
        im = ax.imshow(accuracy)

        ax.set_xticks(np.arange(len(tested_on)))
        ax.set_yticks(np.arange(len(trained_on)))

        ax.set_xticklabels(tested_on)
        ax.set_yticklabels(trained_on)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(trained_on)):
            for j in range(len(tested_on)):
                text = ax.text(j, i, accuracy[i, j], ha="center", va="center", color="w")

        ax.set_title("Gender prediction evaluation (accuracy)")
        ax.set_xlabel("Testing")
        ax.set_ylabel("Training")

        fig.tight_layout()
        plt.show()


def train_and_save_best():

        # read data
        df = pd.read_pickle("data/preprocessed_gender_classifier_dataset.pkl")

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
                df[["processed_text", "processed_description", "processed_both"]],
                df["gender"],
                random_state=0)
        
        # train
        train_col = "processed_both"

        clf = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))

        print(f"Training {str(clf)} on {train_col}")
        clf.fit(X_train[train_col], y_train)

        # verify accuracy
        test_columns = ["processed_text", "processed_description", "processed_both"]

        for test_col in test_columns:
            y_pred = clf.predict(X_test[test_col])
            print(f"Accuracy on {test_col}: {round(accuracy_score(y_test, y_pred), 3)}")

        # save classifier
        with open("pretrained-models/gender/clf.pkl", "wb") as f:
            pickle.dump(clf, f, protocol=4)


if __name__ == "__main__":
        
        search_classifiers()
        # Tfidf + SGDClassifier gives good accuracy and fast inference

        search_features()
        # training on tweet text + user description gives best accuracy and robustness

        train_and_save_best()
        # train Tfidf + SGDClassifier on tweet text + user description
