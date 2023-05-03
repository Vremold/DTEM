import os
import sys
import json

import dgl
import dgl.nn.pytorch as dglnn
import torch

from dgl import load_graphs
from dgl.transforms import AddReverse
from dgl.nn.pytorch import MetaPath2Vec
from torch.optim import SparseAdam
from torch.utils.data import DataLoader

if __name__ == "__main__":
    src_structure_graph = "./full_graph/structure_graph.bin"
    hg = load_graphs(src_structure_graph)[0][0]

    print(hg.etypes)

    metapath_list = [
        ('contributor', 'contributor_propose_pr', 'pr'),
        ('pr', 'pr_belong_to_repo', 'repository'),
        ("repository", "repo_committed_by_contributor", "contributor"),
        ('contributor', 'contributor_follow_contributor', 'contributor'),
        ('contributor', 'contributor_star_repo', 'repository'),
        ("repository", "repo_committed_by_contributor", "contributor"),
        ('contributor', 'contributor_watch_repo', 'repository'),
        ("repository", "repo_committed_by_contributor", "contributor"),
        ('contributor', 'contributor_propose_issue', 'issue'),
        ('issue', 'issue_belong_to_repo', 'repository'),
        ("repository", "repo_committed_by_contributor", "contributor"),
    ]

    model = MetaPath2Vec(
        g=hg,
        metapath=[item[1] for item in metapath_list],
        window_size=3,
        emb_dim=256,
        negative_size=5
    ).to(torch.device("cuda:0"))

    dataloader = DataLoader(dataset=torch.arange(hg.number_of_nodes("contributor")), batch_size=1024, shuffle=True, collate_fn=model.sample)
    optimizer = SparseAdam(model.parameters(), lr=0.01)

    model.train()
    for i in range(5):
        train_losses = []
        for (pos_u, pos_v, neg_v) in dataloader:
            pos_u = pos_u.to(torch.device("cuda:0"))
            pos_v = pos_v.to(torch.device("cuda:0"))
            neg_v = neg_v.to(torch.device("cuda:0"))
            
            loss = model(pos_u, pos_v, neg_v)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    node_embedding = {}
    for ntype in hg.ntypes:
        ntype_nids = torch.LongTensor(model.local_to_global_nid[ntype]).to(torch.device("cuda:0"))
        ntype_embs = model.node_embed(ntype_nids)
        node_embedding[ntype] = ntype_embs
    
    dst_dir = "./cache/full_graph"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    torch.save(node_embedding, os.path.join(dst_dir, "node_metapath_embedding.bin"))