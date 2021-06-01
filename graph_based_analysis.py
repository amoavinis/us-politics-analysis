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


# Most retweeted users (user_id, retweets):
#
# 939091 , 38210
# 30354991 , 12539
# 226222147 , 6220
# 216776631 , 5478
# 1205226529455632385 , 5282
# 32871086 , 4766
# 1640929196 , 4682
# 255812611 , 4666
# 1243560408025198593 , 3998
# 39344374 , 3798
# 292929271 , 3645
# 14247236 , 3623
# 592730371 , 3066
# 803694179079458816 , 2998
# 1073047860260814848 , 2976
# 357606935 , 2913
# 1108472017144201216 , 2831
# 78523300 , 2579
# 2228878592 , 2537
# 236487888 , 2525
# 90480218 , 2379
# 471677441 , 2307
# 18266688 , 2208
# 963790885937995777 , 2188
# 1058520120 , 2107
# 38495835 , 2089
# 259001548 , 2083
# 288277167 , 2048
# 225265639 , 2033
# 27493883 , 2002
# 1043185714437992449 , 1942
# 1212806053907185664 , 1839
# 970207298 , 1732
# 15976705 , 1670
# 15115280 , 1573
# 91882544 , 1559
# 21059255 , 1556
# 216065430 , 1550
# 1917731 , 1514
# 39349894 , 1477
