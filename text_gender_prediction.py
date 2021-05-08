import pickle
import autosklearn.classification

import pandas as pd

from autosklearn.metrics import make_scorer
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


df = pd.read_pickle("data/preprocessed_gender_classifier_dataset.pkl")

X_train, X_test, y_train, y_test = train_test_split(df["processed_text"], df["gender"], random_state=42)

# Experiment 1
clf = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000, random_state=0))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\n{clf}")
evaluate(y_test, y_pred)
# Accuracy: 0.5920159680638722

# Experiment 2
clf = make_pipeline(
        TfidfVectorizer(),
        RandomForestClassifier(random_state=0))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\n{clf}")
evaluate(y_test, y_pred)
# Accuracy: 0.5944111776447106

# Experiment 3
clf = make_pipeline(
        TfidfVectorizer(),
        AdaBoostClassifier(
            LogisticRegression(max_iter=1000, random_state=0),
            random_state=0))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\n{clf}")
evaluate(y_test, y_pred)
# Accuracy: 0.5389221556886228

# Experiment 4
clf = make_pipeline(
        TfidfVectorizer(),
        ExtraTreesClassifier(random_state=0))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\n{clf}")
evaluate(y_test, y_pred)
# Accuracy: 0.5904191616766467

# Experiment 5
clf = make_pipeline(
        TfidfVectorizer(),
        GradientBoostingClassifier(random_state=0))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\n{clf}")
evaluate(y_test, y_pred)
# Accuracy: 0.5800399201596806

# Experiment 6
tfidf = TfidfVectorizer()
X_train_trans = tfidf.fit_transform(X_train)
clf = HistGradientBoostingClassifier(random_state=0)
clf.fit(X_train_trans.toarray(), y_train)
X_test_trans = tfidf.transform(X_test)
print(f"\n{clf}")
y_pred = clf.predict(X_test_trans.toarray())
evaluate(y_test, y_pred)
Accuracy: 0.5912175648702594

# Experiment 7
clf = make_pipeline(
        TfidfVectorizer(),
        TruncatedSVD(n_components=512, random_state=0),
        LogisticRegression(max_iter=1000, random_state=0))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\n{clf}")
evaluate(y_test, y_pred)
# Accuracy: 0.5964071856287425
with open("pretrained-models/gender/clf.pkl", "wb") as f:
    pickle.dump(clf, f, protocol=4)

# Experiment 8
clf = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\n{clf}")
evaluate(y_test, y_pred)
# Accuracy: 0.5992015968063872

# Experiment 9
# Auto-sklearn
preprocess = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=512, random_state=0))
X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)
# auto-sklearn
autoclf = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=round(3600 * 1),
    per_run_time_limit=round(3600 * 1 / 10),
    n_jobs=4,
    memory_limit=3072,
    metric=make_scorer(name="Accuracy", score_func=accuracy_score),
    seed=1)
autoclf.fit(X_train, y_train, feat_type=["Numerical"] * X_train.shape[1])
y_pred = autoclf.predict(X_test)
evaluate(y_test, y_pred)
# Accuracy score: 0.603193612774451
