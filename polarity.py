import nltk
import string
import preprocessor
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


nltk.download('vader_lexicon')


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def compound_to_sentiment(compound_score):
    # Reference: https://github.com/cjhutto/vaderSentiment#about-the-scoring
    if compound_score >= 0.05:
        return 'pos'
    elif compound_score > -0.05 and compound_score < 0.05:
        return 'neu'
    elif compound_score <= -0.05:
        return 'neg'

def detect_sentiment(df, tweet_col='text', processed_col='cleaned_text'):
    analyzer = SentimentIntensityAnalyzer()

    if processed_col not in df.columns:
        df['cleaned_text'] = df['text']\
            .apply(preprocessor.clean)\
            .str.lower()\
            .apply(remove_punctuations)
    else:
        df = df.rename(columns={processed_col: 'cleaned_text'})
    
    df['raw_sentiment'] = df['cleaned_text'].apply(lambda tweet: analyzer.polarity_scores(tweet))
    df['total_sentiment'] = df['raw_sentiment']\
                                .apply(lambda score_dict: score_dict['compound'])\
                                .apply(lambda compound: compound_to_sentiment(compound))
    return df


def detect_subjectivity(df, tweet_col='text', processed_col='cleaned_text'):
    if processed_col not in df.columns:
        df['cleaned_text'] = df['text']\
            .apply(preprocessor.clean)\
            .str.lower()\
            .apply(remove_punctuations)
    else:
        df = df.rename(columns={processed_col: 'cleaned_text'})
    
    df['subjectivity'] = df['cleaned_text'].apply(lambda tweet: TextBlob(tweet).subjectivity)
    return df


# Example use
if __name__ == '__main__':
    
    df = pd.read_json('~/Desktop/tweets-us-29sept-19oct.json')
    
    df = df.sample(20)
    
    df = detect_sentiment(df)
    df = detect_subjectivity(df)
    
    print(df)
