import os
import sys
import json
from tqdm import tqdm
import random
import time

import numpy as np
import torch
from torch.utils.data import random_split
import torch.nn.functional as F
import dgl

from hetsage import HetSAGE
from utils import (GraphLoader)

SPECIAL_SUFFIX = "dump_node_embedding"

class ParameterNamespace():
    def __init__(self, special_suffix):
        if not special_suffix:
            sys.exit(0)
        now = time.strftime("%Y,%m,%d,%H,%M,%S")
        self.checkpoint_dir = f"./checkpoint/{special_suffix}_{now}/"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        
        self.logger = None

        self.graph_path = "../DataPreprocess/full_graph/structure_graph_with_node_feature.bin"

        self.use_gpu = False
        self.device = None
        if self.use_gpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Model parameters
        self.residual = True  # whether to use residual connection after a GNN layer
        self.use_self_loop = True  # whether to include self loop message in GNN
        self.dropout = 0.2  # dropout rate
        self.feat_drop = 0.2 # feature dropout rate
        self.fanouts = [15, 15, 15] # fanout of each GNN layer
        self.num_hidden_layers = len(self.fanouts) - 2 # number of hidden GNN layers other than input and output layer
        self.layer_norm = True  # whether to perform layer normalization before each GNN layer
        
        # Model tensor input and output size
        self.embed_size = 512
        self.in_feats = 512
        self.hidden_size = 512
        self.out_feats = 512

        # parameters for dataloading
        self.negative_samples = 1
        self.lp_batch_size = 512
        self.er_batch_size = 48
        self.ec_batch_size = 24
        self.ec_etype = ("pr", "pr_belong_to_repo", "repository")
        self.er_etype = ("repository", "repo_committed_by_contributor", "contributor")
        
        # training parameters
        self.epochs = 40
        self.lr = 0.0001
        self.lr_decay = 0.95
        self.weight_decay = 0.001
        self.max_grad = 4
        self.omega = 0.1

        # pretrained model path
        # Please replace the path with your own trained model path
        self.trained_model_path = "./bin/cat/model.bin"
        self.node_embedding_dir = f"./node_embedding/{special_suffix}_{now}/"
        if not os.path.exists(self.node_embedding_dir):
            os.mkdir(self.node_embedding_dir)

    def generate_model_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"model_in_epoch{i}")
    def generate_lp_scorer_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"lp_scorer_in_epoch{i}")
    def generate_ec_scorer_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"ec_scorer_in_epoch{i}")
    def generate_er_scorer_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"er_scorer_in_epoch{i}")

if __name__ == "__main__":
    pn = ParameterNamespace(
        special_suffix=SPECIAL_SUFFIX
    )

    # load graph
    graph_loader = GraphLoader(graph_path=pn.graph_path)
    hg, node_feats, edge2ids = graph_loader.load_graph(device=pn.device)
    node_feat_dim_dict = {ntype: node_feats[ntype].shape[1] for ntype in hg.ntypes}

    """
    Move the graph to GPU
    """
    if pn.use_gpu:
        hg = hg.to(pn.device)

    # alias for graph params
    n_ntypes = len(hg.ntypes)
    n_etypes = len(hg.etypes)
    n_edges = hg.number_of_edges()
    n_nodes = hg.number_of_nodes()
    etypes = hg.etypes
    
    model = HetSAGE(
        hg=hg,
        node_feat_dim_dict=node_feat_dim_dict,
        embed_size=pn.embed_size,
        hidden_dim=pn.hidden_size,
        out_dim=pn.out_feats,
        num_hidden_layers=pn.num_hidden_layers,
        feat_drop=pn.feat_drop)
    obj = torch.load(pn.trained_model_path, map_location=pn.device)
    model.load_state_dict(obj)
    model = model.to(pn.device)

    x = model.inference(hg, node_feats, batch_size=512, device=pn.device, num_workers=4)
    torch.save(x, pn.node_embedding_dir + "node_embedding.bin")
