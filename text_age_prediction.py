from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import joblib
import pickle
import numpy as np
import pandas as pd
import os

def convert_age(n):
    if n<25:
        return 0
    elif n<45:
        return 1
    elif n<65:
        return 2
    else:
        return 3

def get_pipeline():
    vec = TfidfVectorizer()
    svd = TruncatedSVD(n_components=512, random_state=0)
    scaler = MinMaxScaler((-1, 1))
    model = MLPClassifier((100, 30), momentum=0.99, max_iter=500, random_state=0)
    pipeline = Pipeline(steps=[('tfidf', vec), ('svd', svd), ('scaler', scaler), ('mlp', model)])

    return pipeline


data = pd.read_csv('data/user-age-dataset.csv')

X, y = data['text'], data['age']
y = [convert_age(n) for n in y]
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0) 
print("Loaded and split data")

pipeline = get_pipeline()

if not os.path.exists('pretrained-models/user-age/text_age_pipeline.pkl'):
    pipeline.fit(x_train, y_train)
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
