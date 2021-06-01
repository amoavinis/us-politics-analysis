import networkx as nx
import pandas as pd


def get_edge_list():
    
    # get edge list from tweets that are retweets
    df_retweets = pd.read_pickle("data/tweets-us-all-with-retweets.pkl")
    df_retweets = df_retweets.dropna(subset=["retweet_graph_edge"])
    edge_list = df_retweets["retweet_graph_edge"].tolist()
    print(f"\nWe found {len(edge_list)} graph edges.")

    return edge_list


def export_sample_graph(n_nodes):

    # get edges
    edge_list = get_edge_list()

    # build graph from edge list
    G = nx.Graph()
    G.add_edges_from(edge_list[:n_nodes])

    # save for gephi
    nx.write_gexf(G, "data/sample-retweet-graph.gexf")


def get_most_retweeted_users(n):

    # get edges
    edge_list = get_edge_list()

    # build graph from edge list
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # get most retweetes users
    most_retweeted = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:n]
    
    print("\nMost retweeted users (user_id, retweets):")
    for user_id, retweets in most_retweeted:
        print(user_id, ",", retweets)


if __name__ == "__main__":

    export_sample_graph(n_nodes=10000)

    get_most_retweeted_users(n=40)
