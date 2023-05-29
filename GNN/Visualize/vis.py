import os
import sys
import json

import dgl
import torch
import torch.nn as nn

from utils import GraphLoader, NodeID

node_id_reader = NodeID(
    repo_idx_file="../DataPreprocess/full_graph/content/repositories.json",
    contributor_idx_file="../DataPreprocess/full_graph/content/contributors.json",
    issue_idx_file="../DataPreprocess/full_graph/content/issues.json",
    pr_idx_file="../DataPreprocess/full_graph/content/prs.json",
)

hetero_graph_path = "../DataPreprocess/full_graph/structure_graph_with_node_feature.bin"
hg, node_feats, edge2ids = GraphLoader(graph_path=hetero_graph_path).load_graph(device=torch.device("cpu"))

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 30])
# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
dataloader = dgl.dataloading.DataLoader(
    hg, 
    {"repository": torch.LongTensor([16328])},
    sampler,
    batch_size=1,
    shuffle=True,
    drop_last=False,
    device=torch.device("cpu"),
)

edges = {}

for input_nodes, output_nodes, blocks in dataloader:
    for b in blocks:
        # for etype in b:
        #     src_nodes, dst_nodes = b.edges(etype=etype, form="uv")
        #     edges[etype] = (src_nodes.numpy().tolist(), dst_nodes.numpy().tolist())
        eids = b.edata[dgl.EID]
        # print(eids)
        for etype in eids:
            src_nodes, dst_nodes = hg.find_edges(etype=etype, eid=eids[etype])
            print(etype, src_nodes, dst_nodes)
            if etype not in edges:
                edges[etype] = {
                    "src": [],
                    "dst": []
                }
            edges[etype]["src"].extend(src_nodes.numpy().tolist())
            edges[etype]["dst"].extend(dst_nodes.numpy().tolist())
    break

node_gid = {}
refined_edges = []
refined_nodes = []

for cetype in edges:
    etype = cetype
    if isinstance(cetype, tuple):
        etype = cetype[1]
    src_type = None
    dst_type = None
    if etype == "repo_committed_by_contributor":
        src_type = "repo"
        dst_type = "contributor"
    elif etype == "contributor_follow_contributor":
        src_type = "contributor"
        dst_type = "contributor"
    elif etype == "contributor_propose_issue":
        src_type = "contributor"
        dst_type = "issue"
    elif etype == "contributor_propose_pr":
        src_type = "contributor"
        dst_type = "pr"
    elif etype == "contributor_star_repo":
        src_type = "contributor"
        dst_type = "repo"
    elif etype == "contributor_watch_repo":
        src_type = "contributor"
        dst_type = "repo"
    elif etype == "issue_belong_to_repo":
        src_type = "issue"
        dst_type = "repo"
    elif etype == "pr_belong_to_repo":
        src_type = "pr"
        dst_type = "repo"
    else:
        print("What!!", etype)
    
    src_nodes, dst_nodes = edges[cetype]["src"], edges[cetype]["dst"]
    for src, dst in zip(src_nodes, dst_nodes):
        src_name = node_id_reader.get_name_by_id(src, src_type)
        dst_name = node_id_reader.get_name_by_id(dst, dst_type)
        src_key = f"{src_type}_{src}"
        dst_key = f"{dst_type}_{dst}"
        if src_key not in node_gid:
            node_gid[src_key] = len(node_gid)
            refined_nodes.append({
                "id": str(node_gid[src_key]),
                "label": src_name,
                "type": src_type
            })
        if dst_key not in node_gid:
            node_gid[dst_key] = len(node_gid)
            refined_nodes.append({
                "id": str(node_gid[dst_key]),
                "label": dst_name,
                "type": dst_type
            })
        
        refined_edges.append({
            "source": str(node_gid[src_key]),
            "target": str(node_gid[dst_key]),
            "type": etype
        })

with open("sampled_graph.json", "w", encoding="utf-8") as outf:
    json.dump(
        {
            "nodes": refined_nodes,
            "edges": refined_edges,
        }, ensure_ascii=False, indent=4, fp=outf
    )
