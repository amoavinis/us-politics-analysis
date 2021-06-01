import networkx as nx
import pandas as pd


def print_stats():

    df = pd.read_pickle("data/tweets-us-all-with-retweets.pkl")
    
    total_tweets = len(df)
    total_users = len(df["user_id"].unique())
    avg_tweets_per_user = total_tweets / total_users

    print(f"\nTotal number of tweets in the dataset: {total_tweets}")

    print(f"\nTotal number of users in the dataset: {total_users}")

    print(f"\nAverage number of tweets per user: {avg_tweets_per_user:.2f}")


if __name__ == "__main__":

    print_stats()