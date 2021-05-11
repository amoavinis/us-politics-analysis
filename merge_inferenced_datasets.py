import pandas as pd
from pandas.core.frame import DataFrame


if __name__ == "__main__":
    political = pd.read_pickle('data/inferences/tweets-with-political.pkl')
    sentiment = pd.read_pickle('data/inferences/tweets-with-sentiment.pkl')
    subj = pd.read_pickle('data/inferences/tweets-with-subj.pkl')
    gender = pd.read_pickle('data/inferences/tweets-with-gender.pkl')
    age = pd.read_pickle('data/inferences/tweets-with-age.pkl')
    
    df_merge = political\
                .merge(sentiment[['ID', 'raw_sentiment', 'total_sentiment']], how='inner', on='ID')\
                .merge(subj[['ID', 'subjectivity']], how='inner', on='ID')\
                .merge(gender[['ID', 'gender']], how='inner', on='ID')\
                .merge(age[['ID', 'age']], how='inner', on='ID')
                
    
    print(df_merge.columns)
    df_merge.to_pickle('data/inferences/tweets-inferenced-all-but-topics.pkl')
