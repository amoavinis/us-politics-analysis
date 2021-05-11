import pandas as pd


if __name__ == "__main__":
    political = pd.read_pickle('data/inferences/tweets-with-political.pkl')
    sentiment = pd.read_pickle('data/inferences/tweets-with-sentiment.pkl')
    subj = pd.read_pickle('data/inferences/tweets-with-subj.pkl')
    
    df_merge = political\
                .merge(sentiment[['ID', 'raw_sentiment', 'total_sentiment']], how='inner', on='ID')\
                .merge(subj[['ID', 'subjectivity']], how='inner', on='ID')
    
    print(df_merge.columns)
