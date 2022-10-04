# 统计嵌入节点编号

import sys
import networkx as nx

embfile = sys.argv[1]
emb2Dfile = sys.argv[2]
targetfile = sys.argv[3]
with open(embfile, 'r') as fil:
    with open(targetfile + '_number.txt', 'w') as wrt:
        i = 0
        for line in fil:
            if i != 0:
                wrt.write(line[:line.find(' ')] + ' ')
            i += 1

# 整张图可视化

G = nx.Graph()
i = 0
with open(targetfile + '_number.txt', 'r') as mapp:
    m = mapp.read()
    mm = m.split()
    mm = [int(x) for x in mm]
G.add_nodes_from(mm)
with open(emb2Dfile, 'r') as emb:
    for p in emb:
        pxy = p.split()
        ind = mm[i]
        if ind < 400000:
            G.nodes[ind]['is_repo'] = 1
        else:
            G.nodes[ind]['is_repo'] = 0
        G.nodes[ind]['viz'] = {}
        G.nodes[ind]['viz']['position'] = {}
        G.nodes[ind]['viz']['position']['x'] = float(pxy[0]) * 50
        G.nodes[ind]['viz']['position']['y'] = float(pxy[1]) * 50
        G.nodes[ind]['viz']['position']['z'] = 0.0
        i += 1

nx.write_gexf(G, targetfile + '.gexf')
