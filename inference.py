import pandas as pd
import pickle
import joblib

from polarity import detect_sentiment, detect_subjectivity


if __name__ == "__main__":
    df = pd.read_pickle('tweets.pkl')

    df = df.sample(20, random_state=0)
    
    # Load the 3 models
    with open('pretrained-models/gender/clf.pkl', 'rb') as f:
        gender_model = pickle.load(f)

    # political orientation model is saved via joblib instead of pickle
    with open('pretrained-models/political-orientation/model.pkl', 'rb') as f:
        political_model = joblib.load(f)

    with open('pretrained-models/user-age/text_age_pipeline.pkl', 'rb') as f:
        age_model = pickle.load(f)

    # We use the 0 index to return just prediction instead of array
    df['gender'] = df['text'].apply(lambda text: gender_model.predict([text])[0])
    df['political'] = df['text'].apply(lambda text: political_model.predict([text])[0])
    df['age'] = df['text'].apply(lambda text: age_model.predict([text])[0])

    # Populate with sentiment
    df = detect_sentiment(df)
    # Populate with subjectivity
    df = detect_subjectivity(df)

    print(df)
