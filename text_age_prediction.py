import xgboost
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
import pickle
import os
import pandas as pd

def get_pipeline():
    vec = TfidfVectorizer()
    scaler = MaxAbsScaler()
    xgb = xgboost.XGBClassifier(n_estimators=2000, use_label_encoder=False, n_jobs=8)
    pipeline = Pipeline(steps=[('tfidf', vec), ('scaler', scaler), ('xgb', xgb)])

    return pipeline

data = pd.read_csv("data/preprocessed-user-age-dataset.csv")

X, y = data['text'], data['age']
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7) 
print("Loaded and split data")

pipeline = get_pipeline()

if os.path.exists('pretrained-models/user-age/text_age_pipeline.pkl'):
    pipeline.fit(x_train, y_train, xgb__eval_metric='logloss')
    pickle.dump(pipeline, open('pretrained-models/user-age/text_age_pipeline.pkl', 'wb'))
    print("Training completed")
else:
    pipeline = pickle.load(open('pretrained-models/user-age/text_age_pipeline.pkl', 'rb'))
    print("Pipeline loaded from pickle file")

y_train_pred = pipeline.predict(x_train)
y_pred = pipeline.predict(x_test)

train_f1 = f1_score(y_train, y_train_pred, average='macro')
train_accuracy = accuracy_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_pred, average='macro')
test_accuracy = accuracy_score(y_test, y_pred)

print("Training F1:", round(train_f1, 2))
print("Training accuracy:", round(train_accuracy, 2))
print("Testing F1:", round(test_f1, 2))
print("Testing accuracy:", round(test_accuracy, 2))
