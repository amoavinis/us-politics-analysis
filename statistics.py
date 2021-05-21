import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    
    # For gender find political orientation
    print("Females:\n", df[df['gender']==1]['political'].value_counts())
    print("Males:\n", df[df['gender']==0]['political'].value_counts())

    # For gender find age
    print("Females:\n", df[df['gender']==1]['age'].value_counts())
    print("Males:\n", df[df['gender']==0]['age'].value_counts())

    # For gender find sentiment (polarity)
    print("Females:\n", df[df['gender']==1]['total_sentiment'].value_counts())
    print("Males:\n", df[df['gender']==0]['total_sentiment'].value_counts())

    # For gender find subjectivity
    print("Females:\n", df[df['gender']==1]['subjectivity'].value_counts())
    print("Males:\n", df[df['gender']==0]['subjectivity'].value_counts())

    
    # For political find gender
    print("Republicans:\n", df[df['political']==1]['gender'].value_counts())
    print("Democrats:\n", df[df['political']==0]['gender'].value_counts())

    # For political find age
    print("Republicans:\n", df[df['political']==1]['age'].value_counts())
    print("Democrats:\n", df[df['political']==0]['age'].value_counts())

    # For political find sentiment (polarity)
    print("Republicans:\n", df[df['political']==1]['total_sentiment'].value_counts())
    print("Democrats:\n", df[df['political']==0]['total_sentiment'].value_counts())

    # For political find subjectivity
    print("Republicans:\n", df[df['political']==1]['subjectivity'].value_counts())
    print("Democrats:\n", df[df['political']==0]['subjectivity'].value_counts())

    # Topics stats
    df['topic'] = df['topic_distribution'].apply(lambda x: np.argmax(x))

    for k in range(0, 6):
        print(df[df['period'] == k]['topic'].value_counts())
        print("Sum: ", df[df['period'] == k]['topic'].count())

    # Topic 4 of period 4 content based stats
    print('political')
    print(df[(df['period'] == 4) & (df['topic'] == 4)]['political'].value_counts())
    print('total_sentiment')
    print(df[(df['period'] == 4) & (df['topic'] == 4)]['total_sentiment'].value_counts())
    print('subjectivity')
    print(df[(df['period'] == 4) & (df['topic'] == 4)]['subjectivity'].value_counts())
    print('gender')
    print(df[(df['period'] == 4) & (df['topic'] == 4)]['gender'].value_counts())

