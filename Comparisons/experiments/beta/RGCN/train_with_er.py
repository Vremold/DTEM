import os
import sys
import json
from tqdm import tqdm
import random
import time
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from rgcn import RGCN
from model import (EdgeRegresionScorer)
from utils import (GraphLoader, 
                   generate_logits_and_labels_for_er, 
                   prepare_dataloader_for_er,
                   l2_penalty)

SPECIAL_SUFFIX = "train_lp_rgcn_self_loop_15_15_15"
wandb.init(
    project="RGCN_train_er_fanouts_15_15_15_residual"
)

class ParameterNamespace():
    def __init__(self, special_suffix):
        # basic parameters
        if not special_suffix:
            sys.exit(0)
        now = time.strftime("%Y,%m,%d,%H,%M,%S")
        self.checkpoint_dir = f"./checkpoint/{special_suffix}_{now}/"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        
        self.logger = None
        # self.logger = open(f"./log/{special_suffix}_{now}.log", "w", encoding="utf-8")

        self.graph_path = "../DataPreprocess/full_graph/structure_graph_with_node_feature.bin"
        
        self.use_gpu = True
        self.device = None
        if self.use_gpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Model parameters
        self.residual = True  # whether to use residual connection after a GNN layer
        self.use_self_loop = False  # whether to include self loop message in GNN
        self.dropout = 0.2  # dropout rate
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
        self.lr = 0.00005
        self.lr_decay = 1
        self.weight_decay = 0.001
        self.max_grad = 4
        self.omega = 0.1

        # pretrained model path
        # Please replace the path with your own pretrained model path
        self.pretrain_model_path = "/root/wujw/DTEM/GNN/RGCN/pretrained/..."

    def generate_model_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"model_in_epoch{i}")
    def generate_lp_scorer_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"lp_scorer_in_epoch{i}")
    def generate_ec_scorer_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"ec_scorer_in_epoch{i}")
    def generate_er_scorer_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"er_scorer_in_epoch{i}")


def evaluate(model, er_scorer, eval_er_dataloader, node_feats):
    model.eval()
    er_scorer.eval()

    er_losses = []
    er_maes = []
    er_rmses = []

    with torch.no_grad():
        for er_batch in eval_er_dataloader:
            _, subgraph, blocks = er_batch
            subgraph = subgraph.to(pn.device)
            blocks = [b.to(pn.device) for b in blocks]
            x = model(blocks, node_feats)
            
            er_logits = er_scorer(subgraph, x)
            er_labels = subgraph.edata["cr_weight"]

            er_logits, er_labels = generate_logits_and_labels_for_er(er_logits, er_labels, pn.er_etype)
            er_loss = F.smooth_l1_loss(er_logits, er_labels, beta=1)
            er_losses.append(er_loss.item())

            mae = F.l1_loss(er_logits, er_labels, reduction="mean")
            mse = F.mse_loss(er_logits, er_labels, reduction="mean")
            rmse = torch.sqrt(mse)
            er_maes.append(mae.item())
            er_rmses.append(rmse.item())

    wandb.log({
        "validation_er_loss": np.mean(er_losses),
        "validation_er_mae": np.mean(er_maes),
        "validation_er_rmse": np.mean(er_rmses),
    })

    return np.mean(er_losses), np.mean(er_maes), np.mean(er_rmses)

def train(
        model, er_scorer, er_dataloader, eval_er_dataloader, test_er_dataloader,
        node_feats, epochs, lr=0.001):
    print("Start Training:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=pn.logger)

    def get_decay_parameters(module:torch.nn.Module, no_decay_params:list, decay_params:list):
        for name, param in module.named_parameters():
            if "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    no_decay_params = []
    decay_params = []
    get_decay_parameters(model, no_decay_params, decay_params)
    get_decay_parameters(er_scorer, no_decay_params, decay_params)
    print(len(decay_params), len(no_decay_params))
    opt = torch.optim.Adam([
        {
            "params": no_decay_params,
        },
        {
            "params": decay_params,
            "weight_decay": pn.weight_decay
        }
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=pn.lr_decay)

    best_er_loss = 10000000

    for epoch in range(epochs):
        train_er_losses = []

        model.train()
        er_scorer.train()

        b_idx = 0
        for er_batch in er_dataloader:
            # Edge Regression
            er_loss = 0
            if er_batch:
                _, subgraph, blocks = er_batch
                subgraph = subgraph.to(pn.device)
                blocks = [b.to(pn.device) for b in blocks]
                x = model(blocks, node_feats)
                
                er_logits = er_scorer(subgraph, x)
                er_labels = subgraph.edata["cr_weight"]

                er_logits, er_labels = generate_logits_and_labels_for_er(er_logits, er_labels, pn.er_etype)
                er_loss = F.smooth_l1_loss(er_logits, er_labels, beta=1)
                train_er_losses.append(er_loss.item())

            loss = er_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            if b_idx % 50 == 0:
                l2_sum = 0
                for weight in decay_params:
                    l2_sum += l2_penalty(weight)
                wandb.log({
                    "train_er_loss": np.mean(train_er_losses),
                    "weight_L2": l2_sum,
                })
                pass
            b_idx += 1

        scheduler.step()
        # Validation
        val_er_loss, val_er_mae, val_er_rmse = evaluate(model, er_scorer, eval_er_dataloader, node_feats)
        print("Train Epoch: {:01d}, lr: {}, train er loss: {:.4f}, \n              val er loss: {:.4f}, val er mae: {:.4f}, val er rmse: {:.4f}".format(
            epoch, opt.param_groups[0]["lr"], 
            np.mean(train_er_losses), 
            val_er_loss, 
            val_er_mae, val_er_rmse), 
            file=pn.logger)

        if best_er_loss > val_er_loss:
            torch.save(model.state_dict(), pn.generate_model_state_file(epoch))
            torch.save(er_scorer.state_dict(), pn.generate_er_scorer_state_file(epoch))
            print("Model saved", file=pn.logger)
        if best_er_loss > val_er_loss:
            best_er_loss = val_er_loss

    # Test
    test_er_loss, test_er_mae, test_er_rmse = \
        evaluate(model, er_scorer, test_er_dataloader, node_feats)
    print("test er loss: {:.4f}, test er mae: {:.4f}, test er rmse: {:.4f}".format(test_er_loss, test_er_mae, test_er_rmse), file=pn.logger)

if __name__ == "__main__":
    pn = ParameterNamespace(special_suffix=SPECIAL_SUFFIX)

    # load graph
    graph_loader = GraphLoader(graph_path=pn.graph_path)
    hg, node_feats, edge2ids = graph_loader.load_graph(device=pn.device)
    node_feat_dim_dict = {ntype: node_feats[ntype].shape[1] for ntype in hg.ntypes}
    print(hg.edges["repo_committed_by_contributor"].data)
    print(hg.edges["pr_belong_to_repo"].data)

    assert "cr_weight" in hg.edges["repo_committed_by_contributor"].data
    assert "pr_label" in hg.edges["pr_belong_to_repo"].data
    
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
    
    model = RGCN(
        hg=hg,
        node_feat_dim_dict=node_feat_dim_dict,
        embed_size=pn.embed_size,
        hidden_dim=pn.hidden_size,
        out_dim=pn.out_feats,
        num_hidden_layers=pn.num_hidden_layers,
        dropout=pn.dropout,
        self_loop=pn.use_self_loop,
        layer_norm=pn.layer_norm)
    
    # load pretrained weights
    model.load_state_dict(torch.load(pn.pretrain_model_path))
    model = model.to(pn.device)

    er_scorer = EdgeRegresionScorer(in_features=pn.hidden_size).to(pn.device)

    train_er_dataloader, eval_er_dataloader, test_er_dataloader = prepare_dataloader_for_er(
        hg, 
        etype="repo_committed_by_contributor",
        fanouts=pn.fanouts,
        batch_size=pn.er_batch_size,
        use_gpu=pn.use_gpu,
        device=pn.device
    )

    # training
    # Testing if everything is on the right place
    if pn.use_gpu:
        assert train_er_dataloader.device == pn.device
        assert eval_er_dataloader.device == pn.device
        assert test_er_dataloader.device == pn.device
        for key in node_feats:
            assert node_feats[key].device == pn.device
    train(model, er_scorer, train_er_dataloader, eval_er_dataloader, test_er_dataloader,
          node_feats, pn.epochs, pn.lr)