import networkx as nx
import json
from networkx.readwrite import json_graph
import sys
import re

file_repo = 'graph/stat_graph.txt'
file_cont = 'graph/all_graph.txt'
file_all = 'graph/final.txt'
with open(file_repo, 'r', encoding='utf-8') as stat:
    ss = stat.read()
    G = json_graph.node_link_graph(eval(ss))
with open(file_cont, 'r', encoding='utf-8') as file:
    ss = file.read()
    H = json_graph.node_link_graph(eval(ss))
G.add_nodes_from(H.nodes(data=True))
G.add_edges_from(H.edges())
with open(file_all, 'w', encoding='utf-8') as file:
    file.write(str(json_graph.node_link_data(G)))