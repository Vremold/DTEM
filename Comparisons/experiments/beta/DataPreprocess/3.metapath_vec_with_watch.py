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
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm

"""
    这个文件修改自 GNN/DataPreprocess/3.build_structure_graph.py

    只运行一次. 
    用 metapath_vec 训练节点的 embedding.
    这个工作的结果将被放到 Comparisons/embedding/beta/node_metapath_embedding_without_watch.bin 中. 

    Comparisons/data/beta/structure_graph_without_watch.bin => Comparisons/embedding/beta/node_metapath_embedding_without_watch.bin

    补充: 
        运行时, 会首先读条(猜测是加载数据), 然后使用GPU训练. 
"""

if __name__ == "__main__":
    src_structure_graph = "Comparisons/data/beta/structure_graph_with_watch.bin"
    hg = load_graphs(src_structure_graph)[0][0]
    """
        load_graphs(src_structure_graph) = ([
            Graph(
                num_nodes={'contributor': 394474, 'issue': 692554, 'pr': 379496, 'repository': 50000}, 
                num_edges={
                    ('contributor', 'contributor_follow_contributor', 'contributor'): 2286407, 
                    ('contributor', 'contributor_propose_issue', 'issue'): 692554, 
                    ('contributor', 'contributor_propose_pr', 'pr'): 379498, 
                    ('contributor', 'contributor_star_repo', 'repository'): 947423, 
                    ('contributor', 'contributor_watch_repo', 'repository'): 150292, 
                    ('issue', 'issue_belong_to_repo', 'repository'): 692554, 
                    ('pr', 'pr_belong_to_repo', 'repository'): 379498, 
                    ('repository', 'repo_committed_by_contributor', 'contributor'): 161241
                },                                                             
                metagraph=[
                    ('contributor', 'contributor', 'contributor_follow_contributor'), 
                    ('contributor', 'issue', 'contributor_propose_issue'), 
                    ('contributor', 'pr', 'contributor_propose_pr'), 
                    ('contributor', 'repository', 'contributor_star_repo'), 
                    ('contributor', 'repository', 'contributor_watch_repo'), 
                    ('issue', 'repository', 'issue_belong_to_repo'), 
                    ('pr', 'repository', 'pr_belong_to_repo'), 
                    ('repository', 'contributor', 'repo_committed_by_contributor')
                ]
            )
        ], {})
    """

    print(hg.etypes)
    
    metapath_list = [
        ('contributor', 'contributor_propose_pr', 'pr'),                        # N
        ('pr', 'pr_belong_to_repo', 'repository'),                              # S
        ("repository", "repo_committed_by_contributor", "contributor"),         # S
        ('contributor', 'contributor_follow_contributor', 'contributor'),       # N
        ('contributor', 'contributor_star_repo', 'repository'),                 # N*
        ("repository", "repo_committed_by_contributor", "contributor"),         # N
        ('contributor', 'contributor_watch_repo', 'repository'),                # N*
        ("repository", "repo_committed_by_contributor", "contributor"),         # N
        ('contributor', 'contributor_propose_issue', 'issue'),                  # N
        ('issue', 'issue_belong_to_repo', 'repository'),                        # N
        ("repository", "repo_committed_by_contributor", "contributor"),         # N
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
        
        print("Training Epoch " + str(i + 1))
        
        train_losses = []
        for (pos_u, pos_v, neg_v) in tqdm(dataloader):
            pos_u = pos_u.to(torch.device("cuda:0"))
            pos_v = pos_v.to(torch.device("cuda:0"))
            neg_v = neg_v.to(torch.device("cuda:0"))
            
            loss = model(pos_u, pos_v, neg_v)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(train_losses)
        
    node_embedding = {}
    for ntype in hg.ntypes:
        ntype_nids = torch.LongTensor(model.local_to_global_nid[ntype]).to(torch.device("cuda:0"))
        ntype_embs = model.node_embed(ntype_nids)
        node_embedding[ntype] = ntype_embs
    
    dst_dir = "Comparisons/embedding/beta"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    torch.save(node_embedding, os.path.join(dst_dir, "node_metapath_embedding_with_watch.bin"))