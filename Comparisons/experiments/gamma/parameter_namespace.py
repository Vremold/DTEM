#!/usr/bin/env python3

import torch
import os
import sys
import time

'''
    下面三个 PrameterNamespace 来自 ./{HetGAT,HetGCN}/pretrain_with_lp.py. 

    提取了公共的部分, 剩余不同的部分. 
'''

class GeneralParameterNamespace(): 
    def __init__(self, special_suffix):
        # basic parameters
        if not special_suffix:
            sys.exit(0)
        now = time.strftime("%Y,%m,%d,%H,%M,%S")
        self.checkpoint_dir = f"./checkpoint/{special_suffix}_{now}/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        self.logger = None

        self.graph_interaction_path = "DataPreprocess/full_graph/interaction_graph_nf.bin"
        # self.graph_social_path = "DataPreprocess/full_graph/social_graph.bin"  # DEPRECATED. DON'T USE. 
        
        
        self.use_gpu = True
        self.device = None
        self.device = torch.device("cuda:1") \
            if self.use_gpu \
            else torch.device("cpu")
        
        # parameters for dataloading
        self.negative_samples = 1
        self.lp_batch_size = 128
        self.er_batch_size = 48
        self.ec_batch_size = 24
        self.ec_etype = ("pr", "pr_belong_to_repo", "repository")
        self.er_etype = ("repository", "repo_committed_by_contributor", "contributor")
        
        # training parameters
        self.epochs = 20
        self.lr = 0.0001
        self.lr_decay = 0.95
        self.weight_decay = 0.001
        self.max_grad = 4
        self.omega = 0.1

    def generate_model_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"model_in_epoch{i}")

    def generate_lp_scorer_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"lp_scorer_in_epoch{i}")
        

        
class HetGATParameterNamespace(): 

    def __init__(self):
        # Model parameters
        self.num_heads = 4  # number of heads in multi-head attention
        self.negative_slope = 0.2  # negative slope for LeakyReLU
        self.residual = False  # whether to use residual connection after a GNN layer
        self.use_self_loop = False  # whether to include self loop message in GNN
        self.dropout = 0.2  # dropout rate
        self.feat_drop = 0.2 # dropout rate for feature
        self.attn_drop = 0.2 # dropout rate for attention
        self.fanouts = [15, 15, 15] # fanout of each GNN layer
        self.num_hidden_layers = len(self.fanouts) - 2 # number of hidden GNN layers other than input and output layer
        self.layer_norm = True  # whether to perform layer normalization before each GNN layer
        
        # Model tensor input and output size
        self.embed_size = 512
        self.in_feats = 512
        self.hidden_size = 512
        self.out_feats = 512

class HetGCNParameterNamespace(): 

    def __init__(self): 
        # Model parameters
        self.num_heads = 4  # number of heads in multi-head attention
        self.negative_slope = 0.2  # negative slope for LeakyReLU
        self.residual = True  # whether to use residual connection after a GNN layer
        self.use_self_loop = False  # whether to include self loop message in GNN
        self.dropout = 0.2  # dropout rate
        self.feat_drop = 0.2 # dropout rate for feature
        self.attn_drop = 0.2 # dropout rate for attention
        self.fanouts = [5,5,5] # fanout of each GNN layer 
        # Since we already sampled enough contributors in GAT, 
        # We try not to sample to much contributors here. 
        self.num_hidden_layers = len(self.fanouts) - 2 # number of hidden GNN layers other than input and output layer
        self.layer_norm = True  # whether to perform layer normalization before each GNN layer
        
        # Model tensor input and output size
        self.embed_size = 512
        self.in_feats = 512
        self.hidden_size = 512
        self.out_feats = 512

