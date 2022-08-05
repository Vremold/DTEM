import networkx as nx
import json
from networkx.readwrite import json_graph
import sys

file_repo_stat = 'src/repo_statistics.txt'  # 仓库建图
file_graph_stat = 'graph/stat_graph.txt'
G = nx.Graph()
with open(file_repo_stat, 'r', encoding='utf-8') as stat:
    linenum = 0
    for line in stat:
        repo = json.loads(line)
        try:
            if 'full_name' in repo.keys():
                G.add_node(repo['full_name'], description = repo['description'], topics = repo['topics'], stargazers_count = repo['stargazers_count'], watchers = repo['watchers'], fork = repo['fork'], forks = repo['forks'], is_repo = 1)
                if repo['description'] is None:
                    G.nodes[repo['full_name']]['description'] = ''
                else:
                    G.nodes[repo['full_name']]['description'] = repo['description']
        except KeyError:
            print(linenum)
            sys.exit(1)
        linenum += 1
with open(file_graph_stat, 'w', encoding='utf-8') as file:
    file.write(str(json_graph.node_link_data(G)))