import pandas as pd
import pickle
import joblib

from tqdm import tqdm

from polarity import detect_sentiment, detect_subjectivity


def apply_gender(df):
    df['gender'] = df['text'].progress_apply(lambda text: gender_model.predict([text])[0])
    df.to_pickle('data/inferences/tweets-with-gender.pkl')


def apply_political(df):
    df['political'] = df['text'].progress_apply(lambda text: political_model.predict([text])[0])
    df.to_pickle('data/inferences/tweets-with-political.pkl')


def apply_age(df):
    df['age'] = df['text'].progress_apply(lambda text: age_model.predict([text])[0])
    df.to_pickle('data/inferences/tweets-with-age.pkl')


def apply_sentiment(df):
    df = detect_sentiment(df)
    df.to_pickle('data/inferences/tweets-with-sentiment.pkl')
   

def apply_subjectivity(df):
    df = detect_subjectivity(df)
    df.to_pickle('data/inferences/tweets-with-subj.pkl')


if __name__ == "__main__":
    tqdm.pandas()

    df = pd.read_pickle('data/tweets-us-all.pkl')
    
    # Load the 3 models
    with open('pretrained-models/gender/clf.pkl', 'rb') as f:
        gender_model = pickle.load(f)

    with open('pretrained-models/political-orientation/model.pkl', 'rb') as f:
        political_model = pickle.load(f)

    with open('pretrained-models/user-age/text_age_pipeline.pkl', 'rb') as f:
        age_model = pickle.load(f)
    
    print("Political orientation predictions...")
    apply_political(df)

    print("Gender predictions...")
    apply_gender(df)

    print("Sentiment predictions...")
    apply_sentiment(df)

    print("Subjectivity predictions...")
    apply_subjectivity(df)
    
    print("Age predictions...")
    apply_age(df)
