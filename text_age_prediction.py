from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
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


data = pd.read_csv('data/user-age-dataset.csv')

X, y = data['text'], data['age']
y = [convert_age(n) for n in y]
x_train, x_test, y_train, y_test = train_test_split(X, y) 
print("Loaded and split data")

vec = TfidfVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

svd = TruncatedSVD(n_components=512)
x_train = svd.fit_transform(x_train)
x_test = svd.transform(x_test)
print("TFIDF transformation and SVD reduction completed")

scaler = MinMaxScaler((-1, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Scaling completed")

if not os.path.exists('pretrained-models/user-age/text_age_classifier.pkl'):
    model = MLPClassifier((100, 30), momentum=0.99, max_iter=500, random_state=0)
    model.fit(x_train, y_train)
    pickle.dump(model, open('pretrained-models/user-age/text_age_classifier.pkl', 'wb'))
    print("Model trained")
else:
    model = pickle.load(open('pretrained-models/user-age/text_age_classifier.pkl', 'rb'))
    print("Model loaded from pickle file")

y_train_pred = model.predict(x_train)
y_pred = model.predict(x_test)

train_f1 = f1_score(y_train, y_train_pred, average='macro')
train_accuracy = accuracy_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_pred, average='macro')
test_accuracy = accuracy_score(y_test, y_pred)

print("Training F1:", round(train_f1, 2))
print("Training accuracy:", round(train_accuracy, 2))
print("Testing F1:", round(test_f1, 2))
print("Testing accuracy:", round(test_accuracy, 2))
