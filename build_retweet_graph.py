import networkx as nx
import pandas as pd


df = pd.read_json("preprocessing_us-pres-elections-2020/tweets-us-10nov-30nov-extended.json")

edge_list = df["retweet_graph_edge"].tolist()

G = nx.Graph()
G.add_edges_from(edge_list)
