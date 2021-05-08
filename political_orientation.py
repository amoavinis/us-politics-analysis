import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


MODEL_PATH = Path('pretrained-models/political-orientation/model.pkl')

df = pd.read_csv('data/processed-balanced.csv')
df.dropna(axis=0, inplace=True)

x, y = df['processed_msg'], df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)

model = make_pipeline(TfidfVectorizer(min_df=5),
                      LogisticRegression(random_state=0))

if MODEL_PATH.exists():
    print("Loading pretrained model at: {}".format(str(MODEL_PATH)))
    model = joblib.load(str(MODEL_PATH))
else:
    model.fit(x_train, y_train)
    joblib.dump(model, str(MODEL_PATH))

y_pred = model.predict(x_test)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("F1-score: {}".format(f1_score(y_test, y_pred)))
