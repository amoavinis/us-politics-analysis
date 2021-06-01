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


if __name__ == "__main__":

    export_sample_graph(n_nodes=10000)
