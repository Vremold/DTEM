#!/usr/bin/env python3

import traceback

"""
    这个文件是复刻 4.add_node_feature.py 的. 
    那个文件中, 我们将所有需要的 embedding 都拼接到了一起. 
    在这个文件中, 我们将尝试在不改变原有的 embedding 维度的前提下,
    构建一个新的未训练的图. 

    Note: 
        
        输出: ./full_graph/structure_graph_with_random_feature.bin
"""

import time
from datetime import datetime

import os
import sys
import json

import torch
import dgl

from dgl import load_graphs
from dgl.data.utils import save_graphs

from utils import RepositoryFeatureLoader, IssueFeatureLoader, PRFeatureLoader

# 事实证明, 对于开发人员, 还是代码复用(直接复制)效率来得高啊... 
# 下面2行代码完全复制 4.add_node_feature.py. 

structure_graph_file            = "./full_graph/structure_graph.bin"
dst_graph_file                  = "./full_graph/structure_graph_without_feature.bin"

if __name__ == '__main__':

    device = torch.device("cpu")

    # expertise features: [name], [size] and [dim]
    expertise_features = {
        'pr':           (379496, 1536),
        'issue':        (692554, 768),
        'repository':   (50000,  2048),
        'contributor':  (394474, 0),
    }

    hg = load_graphs(structure_graph_file)[0][0]
    for name, size in expertise_features.items(): 
        # size[1] + 256: 预留给 metapath_vec 的位置
        hg.nodes[name].data['feat'] = torch.randn((size[0], size[1] + 256), device=device)

    save_graphs(dst_graph_file, [hg])
