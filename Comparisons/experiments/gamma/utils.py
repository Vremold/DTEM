import sys
import os
import random

import torch
import numpy as np
from torch.utils.data import random_split
import dgl
from dgl import load_graphs

class GraphLoader():
    '''
    The Scheme of the heterogenous graph:
        ("contributor", "contributor_belong_to_org", "organization")
        ("contributor", "contributor_commit_repo", "repository")
        ("contributor", "contributor_propose_issue", "issue")
        ("contributor", "contributor_propose_pr", "pr")
        ("contributor", "contributor_star_repo", "repository")
        ("contributor", "contributor_watch_repo", "repository")
        ("issue", "issue_belong_to_repo", "repository")
        ("pr", "pr_belong_to_repo", "repository")
    '''
    def __init__(self, graph_path) -> None:
        self.graph_path = graph_path
        pass
    
    @staticmethod
    def print_hg_info(hg):
        print("################# Basic Information of The Graph #################")
        for etype in hg.canonical_etypes:
            print("Edge", etype, hg.number_of_edges(etype))
        for ntype in hg.ntypes:
            print("Node", ntype, hg.number_of_nodes(ntype))
        print("Total number of nodes", hg.number_of_nodes())
        print("Total number of edges", hg.number_of_edges())
        print("################# End of the Graph Information  #################")

    def load_graph(self, device=torch.device("cpu")):
        hgs, _ = load_graphs(self.graph_path)
        hg = hgs[0].to(device)
        GraphLoader.print_hg_info(hg=hg)

        edge2ids = {}
        for c_etype in hg.canonical_etypes:
            edge2ids[c_etype] = len(edge2ids)
        print(edge2ids)

        node_feats = {}
        for nt in hg.ntypes:
            node_feats[nt] = hg.nodes[nt].data.pop("feat").to(device)
        return hg, node_feats, edge2ids

def print_detail_for_lp(lp_category_rights, lp_category_totals, logger):
    print("### Link Prediction", file=logger)
    for cate in lp_category_totals:
        print("###  cate: {}, right: {}, total: {}".format(cate, lp_category_rights.get(cate, 0), lp_category_totals[cate]), file=logger)

def print_detail_for_ec(ec_category_rights, ec_category_totals, logger):
    print("### Edge Classification") 
    for cate in ec_category_totals:
        print("###  cate: {}, right: {}, total: {}".format(cate, ec_category_rights.get(cate, 0), ec_category_totals[cate]), file=logger)

def calculate_lp_accuracy(logits, labels, lp_category_rights, lp_category_totals):
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    logits = np.where(logits >= 0.5, 1, 0)
    for l, golden in zip(logits, labels):
        if l == golden:
            lp_category_rights[golden] = lp_category_rights.get(golden, 0) + 1
        lp_category_totals[golden] = lp_category_totals.get(golden, 0) + 1
    return np.sum(logits == labels)

def calculate_ec_accuracy(logits, labels, ec_category_rights, ec_category_totals):
    if len(logits.shape) != len(labels.shape):
        logits = torch.argmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    else:
        logits = logits.detach().cpu().numpy()
        logits = np.where(logits >= 0.5, 1, 0)
        labels = labels.detach().cpu().numpy()
    rights = np.sum(logits == labels)
    for l, golden in zip(logits, labels):
        if l == golden:
            ec_category_rights[golden] = ec_category_rights.get(golden, 0) + 1
        ec_category_totals[golden] = ec_category_totals.get(golden, 0) + 1
    return rights

def generate_logits_and_labels_for_lp(neg_lp_score, pos_lp_score):
    neg_lp_scores = torch.cat([neg_lp_score[c_etype] for c_etype in neg_lp_score])
    pos_lp_scores = torch.cat([pos_lp_score[c_etype] for c_etype in pos_lp_score])

    neg_lp_scores = neg_lp_scores.squeeze()
    pos_lp_scores = pos_lp_scores.squeeze()
    lp_scores = torch.cat([neg_lp_scores, pos_lp_scores], dim=-1)
    lp_labels = torch.cat([torch.zeros_like(neg_lp_scores), torch.ones_like(pos_lp_scores)])
    return lp_scores, lp_labels

def generate_logits_and_labels_for_ec(ec_logits, ec_labels, etype):
    return ec_logits[etype], ec_labels[etype]
    new_ec_logits = []
    new_ec_labels = []
    for c_etype in ec_labels:
        new_ec_logits.append(ec_logits[c_etype])
        new_ec_labels.append(ec_labels[c_etype])
    
    return torch.cat(new_ec_logits, dim=0).squeeze() , torch.cat(new_ec_labels, dim=-1)

def generate_logits_and_labels_for_ec_multi_label(ec_logits, edge2ids, device):
    new_ec_logits = []
    new_ec_labels = []
    for c_etype in ec_logits:
        new_ec_logits.append(ec_logits[c_etype])
        new_ec_labels.extend([edge2ids[c_etype]] * ec_logits[c_etype].shape[0])
    
    new_ec_logits = torch.cat(new_ec_logits, dim=0)
    new_ec_labels = torch.LongTensor(new_ec_labels).to(device)
    return new_ec_logits, new_ec_labels

def generate_logits_and_labels_for_er(er_logits, er_labels, etype):
    return er_logits[etype].squeeze(), er_labels[etype]
    new_er_logits = []
    new_er_labels = []
    for c_etype in er_labels:
        new_er_logits.append(er_logits[c_etype])
        new_er_labels.append(er_labels[c_etype])
    return torch.cat(new_er_logits, dim=0).squeeze() , torch.cat(new_er_labels, dim=-1)

def prepare_dataloader_for_er(hg, etype, fanouts, batch_size, use_gpu, device):
    print("In Prepare Dataloader For ER of", etype)
    train_eid_dict, validate_eid_dict, test_eid_dict = {}, {}, {}
    # ? What hg.edges(etype=etype, form="eid") api for 
    total_length = len(hg.edges(etype=etype, form="eid"))
    validate_length = total_length // 10
    test_length = validate_length
    train_length = total_length - validate_length - test_length
    train_eids, validate_eids, test_eids = random_split(hg.edges(etype=etype, form="eid"), [train_length, validate_length, test_length], generator=torch.Generator().manual_seed(42))
    
    if use_gpu:
        """
        Since the graph is on the GPU, we need the sampling dict be on GPU also
        """
        train_eid_dict[etype] = torch.tensor(train_eids, device=device)
        validate_eid_dict[etype] = torch.tensor(validate_eids, device=device)
        test_eid_dict[etype] = torch.tensor(test_eids, device=device)
    else:
        train_eid_dict[etype] = train_eids
        validate_eid_dict[etype] = validate_eids
        test_eid_dict[etype] = test_eids
    
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    sampler = dgl.dataloading.NeighborSampler(fanouts, replace=False)
    """
    exclue="self" 删除在minibatch中出现的边，这样的预测总不会有任何问题
    """
    train_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude="self")
    val_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude="self")
    test_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude="self")

    if use_gpu:
        train_dataloader = dgl.dataloading.DataLoader(hg, train_eid_dict, train_sampler, batch_size=batch_size, shuffle=True, drop_last=False, device=device)
        val_dataloader = dgl.dataloading.DataLoader(hg, validate_eid_dict, val_sampler, batch_size=512, shuffle=True, drop_last=False, device=device)
        test_dataloader = dgl.dataloading.DataLoader(hg, test_eid_dict, test_sampler, batch_size=512, shuffle=True, drop_last=False, device=device)
    else:
        train_dataloader = dgl.dataloading.DataLoader(hg, train_eid_dict, train_sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        val_dataloader = dgl.dataloading.DataLoader(hg, validate_eid_dict, val_sampler, batch_size=512, shuffle=True, drop_last=False, num_workers=4)
        test_dataloader = dgl.dataloading.DataLoader(hg, test_eid_dict, test_sampler, batch_size=512, shuffle=True, drop_last=False, num_workers=4)
    
    print("     Train Dataloader Length:", len(train_dataloader))
    print("Validation Dataloader Length:", len(val_dataloader))
    print("      Test Dataloader Length:", len(test_dataloader))

    return train_dataloader, val_dataloader, test_dataloader

def prepare_dataloader_for_lp(hg, fanouts, batch_size, negative_samples, use_gpu, device, dropout=0):
    print("In Prepare Dataloader For LP")
    etypes = hg.etypes
    train_eid_dict, validate_eid_dict, test_eid_dict = {}, {}, {}
    for etype in etypes:
        # ? What hg.edges(etype=etype, form="eid") api for 
        edge_eids = hg.edges(etype=etype, form="eid")

        sampled_mask = torch.zeros_like(edge_eids).bool().bernoulli_(1 - dropout)
        sampled_edge_ids = torch.masked_select(edge_eids, sampled_mask)

        total_length = len(sampled_edge_ids)
        validate_length = total_length // 10
        test_length = validate_length
        train_length = total_length - validate_length - test_length
        train_eids, validate_eids, test_eids = random_split(sampled_edge_ids, [train_length, validate_length, test_length], generator=torch.Generator().manual_seed(42))
        
        if use_gpu:
            """
            Since the graph is on the GPU, we need the sampling dict be on GPU also
            """
            train_eid_dict[etype] = torch.tensor(train_eids, device=device)
            validate_eid_dict[etype] = torch.tensor(validate_eids, device=device)
            test_eid_dict[etype] = torch.tensor(test_eids, device=device)
        else:
            train_eid_dict[etype] = train_eids
            validate_eid_dict[etype] = validate_eids
            test_eid_dict[etype] = test_eids
    
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    sampler = dgl.dataloading.NeighborSampler(fanouts, replace=False)
    """
    exclue="self" 删除在minibatch中出现的边，这样的预测总不会有任何问题
    """
    train_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(negative_samples), exclude="self")
    val_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(negative_samples), exclude="self")
    test_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(negative_samples), exclude="self")

    if use_gpu:
        train_dataloader = dgl.dataloading.DataLoader(hg, train_eid_dict, train_sampler, batch_size=batch_size, shuffle=True, drop_last=False, device=device)
        val_dataloader = dgl.dataloading.DataLoader(hg, validate_eid_dict, val_sampler, batch_size=512, shuffle=True, drop_last=False, device=device)
        test_dataloader = dgl.dataloading.DataLoader(hg, test_eid_dict, test_sampler, batch_size=512, shuffle=True, drop_last=False, device=device)
    else:
        train_dataloader = dgl.dataloading.DataLoader(hg, train_eid_dict, train_sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        val_dataloader = dgl.dataloading.DataLoader(hg, validate_eid_dict, val_sampler, batch_size=512, shuffle=True, drop_last=False, num_workers=4)
        test_dataloader = dgl.dataloading.DataLoader(hg, test_eid_dict, test_sampler, batch_size=512, shuffle=True, drop_last=False, num_workers=4)
    
    print("     Train Dataloader Length:", len(train_dataloader))
    print("Validation Dataloader Length:", len(val_dataloader))
    print("      Test Dataloader Length:", len(test_dataloader))
    return train_dataloader, val_dataloader, test_dataloader

def prepare_dataloader_for_ec_specific(hg, etype, label_key, fanouts, batch_size, use_gpu, device, drop_rate=0.7):
    print("In Prepare Dataloader For EC of", etype)
    edge_eids = hg.edges(etype=etype, form="eid")
    sampled_mask = torch.ones_like(edge_eids).bool()
    
    print("Before Drop:", len(edge_eids))
    # Randomly drop drop-rate of the edges of the most frequent edge label
    edge_labels = hg.edges[etype].data[label_key].cpu().numpy()
    indices_to_delete = np.where(edge_labels == 1)
    for index in zip(*indices_to_delete):
        if np.random.rand() < drop_rate:
            sampled_mask[index] = False

    sampled_eids = edge_eids[sampled_mask]
    print("After Drop:", len(sampled_eids))

    train_eid_dict, validate_eid_dict, test_eid_dict = {}, {}, {}
    # ? What hg.edges(etype=etype, form="eid") api for 
    total_length = len(sampled_eids)
    validate_length = total_length // 10
    test_length = validate_length
    train_length = total_length - validate_length - test_length
    train_eids, validate_eids, test_eids = random_split(sampled_eids, [train_length, validate_length, test_length], generator=torch.Generator().manual_seed(42))
    
    if use_gpu:
        """
        Since the graph is on the GPU, we need the sampling dict be on GPU also
        """
        train_eid_dict[etype] = torch.tensor(train_eids, device=device)
        validate_eid_dict[etype] = torch.tensor(validate_eids, device=device)
        test_eid_dict[etype] = torch.tensor(test_eids, device=device)
    else:
        train_eid_dict[etype] = train_eids
        validate_eid_dict[etype] = validate_eids
        test_eid_dict[etype] = test_eids
    
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    sampler = dgl.dataloading.NeighborSampler(fanouts, replace=False)
    """
    exclue="self" 删除在minibatch中出现的边，这样的预测总不会有任何问题
    """
    train_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude="self")
    val_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude="self")
    test_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude="self")

    if use_gpu:
        train_dataloader = dgl.dataloading.DataLoader(hg, train_eid_dict, train_sampler, batch_size=batch_size, shuffle=True, drop_last=False, device=device)
        val_dataloader = dgl.dataloading.DataLoader(hg, validate_eid_dict, val_sampler, batch_size=512, shuffle=True, drop_last=False, device=device)
        test_dataloader = dgl.dataloading.DataLoader(hg, test_eid_dict, test_sampler, batch_size=512, shuffle=True, drop_last=False, device=device)
    else:
        train_dataloader = dgl.dataloading.DataLoader(hg, train_eid_dict, train_sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        val_dataloader = dgl.dataloading.DataLoader(hg, validate_eid_dict, val_sampler, batch_size=512, shuffle=True, drop_last=False, num_workers=4)
        test_dataloader = dgl.dataloading.DataLoader(hg, test_eid_dict, test_sampler, batch_size=512, shuffle=True, drop_last=False, num_workers=4)
    
    print("     Train Dataloader Length:", len(train_dataloader))
    print("Validation Dataloader Length:", len(val_dataloader))
    print("      Test Dataloader Length:", len(test_dataloader))

    return train_dataloader, val_dataloader, test_dataloader


def l2_penalty(w):
    return torch.sum(torch.pow(w, 2))
