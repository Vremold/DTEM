#!/usr/bin/env python

import dgl

hgs = {
    it: dgl.load_graphs(it)[0][0]

    for it in [
        # 'structure_graph.bin', 
        # 'structure_graph_without_feature.bin',
        'structure_graph_with_node_feature.bin', 
        'structure_graph_with_node_feature_only_metapath.bin', 
        'structure_graph_with_node_feature_without_metapath.bin', 
    ]
}

pr_shape = {
    k: v.nodes['pr'].data['feat'].shape
    for k, v in hgs.items()
}

for k, v in pr_shape.items(): 
    print(f'{v}\t{k}')
