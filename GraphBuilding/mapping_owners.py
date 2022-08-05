import networkx as nx
import json
from networkx.readwrite import json_graph
import sys
import re

file_cont_stat = 'repo_contribution.txt'
file_graph_stat = 'all_graph.txt'
G = nx.Graph()
with open(file_cont_stat, 'r', encoding='utf-8') as stat:
    linenum = 0
    for line in stat:
        blk = re.search('\s', line).start()
        full_name = line[:blk]
        line = line[line.find('['):]
        contributers = json.loads(line)
        G.add_node(full_name, is_repo = 1)
        for contributer in contributers:
            G.add_node(contributer[0], is_repo = 0)
            G.add_edge(full_name, contributer[0], weight=contributer[1])
        linenum += 1
with open(file_graph_stat, 'w', encoding='utf-8') as file:
    file.write(str(json_graph.node_link_data(G)))