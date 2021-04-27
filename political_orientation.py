import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split


TRAINED_VECTORIZER_PATH = Path('pretrained-models/political-orientation/vectorizer.pkl')
TRAINED_CLASSIFIER = Path('pretrained-models/political-orientation/clf.pkl')

df = pd.read_csv('data/processed-balanced.csv')
df.dropna(axis=0, inplace=True)

x, y = df['processed_msg'], df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)

if TRAINED_VECTORIZER_PATH.exists():
    vectorizer = joblib.load(str(TRAINED_VECTORIZER_PATH))
else:
    vectorizer = TfidfVectorizer(min_df=5)
    x_train = vectorizer.fit_transform(x_train)
    joblib.dump(vectorizer, str(TRAINED_VECTORIZER_PATH))

x_test = vectorizer.transform(x_test)

if TRAINED_CLASSIFIER.exists():
    clf = joblib.load(str(TRAINED_CLASSIFIER))
else:
    clf = LogisticRegression(random_state=0)
    clf.fit(x_train.toarray(), y_train)
    joblib.dump(clf, str(TRAINED_CLASSIFIER))

y_pred = clf.predict(x_test.toarray())

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("F1-score: {}".format(f1_score(y_test, y_pred)))
