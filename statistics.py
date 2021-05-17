import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_pickle('data/inferences/dataset-inferenced.pkl')

    # Political distribution
    print(df['political'].value_counts())

    # Gender distribution
    print(df['gender'].value_counts())

    # Age distribution
    print(df['age'].value_counts())

    # Sentiment distribution
    print(df['total_sentiment'].value_counts())

    # Subjectivity distribution
    ax = df['subjectivity'].plot.hist()
    plt.show()


    ## df[df['gender']==1]['political'].value_counts()