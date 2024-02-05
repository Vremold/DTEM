
import os
import sys
import json
from datetime import datetime
import random
import time


import numpy as np
import torch
from torch.utils.data import random_split
import torch.nn.functional as F
import dgl

import wandb

from hetsage import HetSAGE

from model import (
    LinkPredictionScorer_V2 as LinkPredictionScorer, 
    EdgeClassificationScorer)

from utils import (GraphLoader, 
                   print_detail_for_lp, 
                   generate_logits_and_labels_for_lp, 
                   calculate_lp_accuracy, 
                   prepare_dataloader_for_lp,
                   l2_penalty)

SPECIAL_SUFFIX = "pretrain_with_only_metapath"
wandb.init(
    project="HetSAGE_pretrain_with_only_metapath"
)

class ParameterNamespace():
    def __init__(self, special_suffix):
        # basic parameters
        if not special_suffix:
            sys.exit(0)
        now = time.strftime("%Y,%m,%d,%H,%M,%S")
        self.checkpoint_dir = f"./checkpoint/{special_suffix}_{now}/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        self.logger = None

        self.graph_path = "../DataPreprocess/full_graph/structure_graph_with_node_feature_only_metapath.bin"
        
        self.use_gpu = True
        self.device = None
        if self.use_gpu:
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")

        # Model parameters
        self.residual = True  # whether to use residual connection after a GNN layer
        self.use_self_loop = False  # whether to include self loop message in GNN
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
        self.lr_decay = 0.9
        self.weight_decay = 0.001
        self.max_grad = 4
        self.omega = 0.1

        # pretrained model path

    def generate_model_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"model_in_epoch{i}")
    def generate_lp_scorer_state_file(self, i):
        return os.path.join(self.checkpoint_dir, f"lp_scorer_in_epoch{i}")


def evaluate(model, lp_scorer, eval_dataloader, node_feats):
    model.eval()
    lp_scorer.eval()

    losses = []
    lp_losses = []
    link_prediction_rights = 0
    link_prediction_totals = 0

    # For tongji
    lp_category_rights = {}
    lp_category_totals = {}

    with torch.no_grad():
        for _, positive_graph, negative_graph, blocks in eval_dataloader:
            blocks = [b.to(pn.device) for b in blocks]
            positive_graph = positive_graph.to(pn.device)
            negative_graph = negative_graph.to(pn.device)

            x = model(blocks, node_feats)
            neg_link_prediction_score = lp_scorer(negative_graph, x)
            pos_link_prediction_score = lp_scorer(positive_graph, x)

            lp_scores, lp_labels = generate_logits_and_labels_for_lp(neg_link_prediction_score, pos_link_prediction_score)
            lp_loss = F.binary_cross_entropy(lp_scores, lp_labels)
            lp_losses.append(lp_loss.item())
            loss = lp_loss
            
            losses.append(loss.item())
            link_prediction_rights += calculate_lp_accuracy(lp_scores, lp_labels, lp_category_rights, lp_category_totals)
            link_prediction_totals += len(lp_scores)
    
    wandb.log({
        "validation_lp_loss": np.mean(lp_losses),
        "validation_lp_accu": link_prediction_rights / link_prediction_totals,
    })
    
    return np.mean(losses), link_prediction_rights / link_prediction_totals, lp_category_rights, lp_category_totals

def train(model, lp_scorer, train_dataloader, eval_dataloader, test_dataloader, node_feats, epochs, lr=0.001):
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
    get_decay_parameters(lp_scorer, no_decay_params, decay_params)
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
    
    best_link_prediction_accuracy = 0
    
    print(datetime.now())
    for epoch in range(epochs):
        train_lp_losses = []
        train_loss_list = []

        link_prediction_rights = 0
        link_prediction_totals = 0

        # For tongji
        lp_category_rights = {}
        lp_category_totals = {}

        model.train()
        lp_scorer.train()

        for i, (_, positive_graph, negative_graph, blocks) in enumerate(train_dataloader):
            blocks = [b.to(pn.device) for b in blocks]
            positive_graph = positive_graph.to(pn.device)
            negative_graph = negative_graph.to(pn.device)
            x = model(blocks, node_feats)
            # print(x)
            
            neg_link_prediction_score = lp_scorer(negative_graph, x)
            pos_link_prediction_score = lp_scorer(positive_graph, x)
            
            lp_scores, lp_labels = generate_logits_and_labels_for_lp(neg_link_prediction_score, pos_link_prediction_score)
            
            lp_loss = F.binary_cross_entropy(lp_scores, lp_labels)
            train_lp_losses.append(lp_loss.item())
            
            loss = lp_loss
            train_loss_list.append(loss.item())
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            link_prediction_rights += calculate_lp_accuracy(lp_scores, lp_labels, lp_category_rights, lp_category_totals)
            link_prediction_totals += len(lp_scores)

            if i % 50 == 0:
                l2_sum = 0
                for weight in decay_params:
                    l2_sum += l2_penalty(weight)
                wandb.log({
                    "train_lp_loss": np.mean(train_lp_losses),
                    "weight_L2": l2_sum,
                    "train_lp_accu": link_prediction_rights / link_prediction_totals,
                })
        
        scheduler.step()

        # Validation
        val_loss, val_link_prediction_accuracy, lp_category_rights, lp_category_totals = \
            evaluate(model, lp_scorer, eval_dataloader, node_feats)
        print(str(datetime.now()) + " Train Epoch: {:01d}, lr: {}, train loss: {:.4f}, val loss: {:.4f}, val link prediction accuracy: {:.4f}.".format(
            epoch, opt.param_groups[0]["lr"], np.mean(train_loss_list), val_loss,
            val_link_prediction_accuracy), file=pn.logger)
        print_detail_for_lp(lp_category_rights, lp_category_totals, logger=pn.logger)

        torch.save(model.state_dict(), pn.generate_model_state_file(epoch))
        torch.save(lp_scorer.state_dict(), pn.generate_lp_scorer_state_file(epoch))
        print("Model saved", file=pn.logger)
        if best_link_prediction_accuracy < val_link_prediction_accuracy:
            best_link_prediction_accuracy = val_link_prediction_accuracy

    # Test
    test_loss, test_link_prediction_accuracy, lp_category_rights, lp_category_totals = \
        evaluate(model, lp_scorer, test_dataloader, node_feats)
    print("test loss: {:.4f}, test link prediction accuracy: {:.4f}.".format(test_loss, test_link_prediction_accuracy), file=pn.logger)
    print_detail_for_lp(lp_category_rights, lp_category_totals, logger=pn.logger)

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
    model = model.to(pn.device)

    lp_scorer = LinkPredictionScorer(in_features=pn.out_feats).to(pn.device)

    train_dataloader, val_dataloader, test_dataloader = prepare_dataloader_for_lp(
        hg, 
        fanouts=pn.fanouts,
        batch_size=pn.lp_batch_size,
        negative_samples=pn.negative_samples, 
        use_gpu=pn.use_gpu,
        device=pn.device,
        dropout=0
    )

    # training
    # Testing if everything is on the right place
    if pn.use_gpu:
        assert train_dataloader.device == pn.device
        assert val_dataloader.device == pn.device
        assert test_dataloader.device == pn.device
        for key in node_feats:
            assert node_feats[key].device == pn.device
    train(model, lp_scorer, train_dataloader, val_dataloader, test_dataloader, node_feats, pn.epochs, pn.lr)