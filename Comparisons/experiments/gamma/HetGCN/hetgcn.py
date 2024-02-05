import sys
import tqdm

import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each
import dgl.function as fn
import dgl.nn.pytorch as dglnn

class HetGCN(nn.Module):
    def __init__(
        self,
        hg,
        node_feat_dim_dict,
        embed_size,
        hidden_dim,
        out_dim,
        num_hidden_layers=1,
        dropout=0,
        residual=False,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.residual = residual

        self.layers = torch.nn.ModuleList()
        self.hetero_linear = dglnn.HeteroLinear(node_feat_dim_dict, embed_size, bias=True)
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(
                in_feats=embed_size, 
                out_feats=hidden_dim, 
                norm="both",
                bias=True,
                weight=True,
                activation=F.relu) 
            for etype in hg.etypes},
            aggregate="sum")
        )
        for _ in range(num_hidden_layers):
            self.layers.append(dglnn.HeteroGraphConv({
                etype: dglnn.GraphConv(
                    in_feats=hidden_dim, 
                    out_feats=hidden_dim, 
                    norm="both",
                    bias=True,
                    weight=True,
                    activation=F.relu)
                for etype in hg.etypes},
                aggregate="sum")
            )
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(
                in_feats=hidden_dim, 
                out_feats=out_dim, 
                norm="both",
                bias=True,
                weight=True,
                activation=None)
            for etype in hg.etypes},
            aggregate="sum")
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, blocks, node_feats, node_tids=None):
        if node_tids is None:
            node_tids = blocks[0].srcdata[dgl.NID]
            if not isinstance(node_tids, dict): 
                node_tids = {blocks[0].ntypes[0]: node_tids}
        
        h = self.hetero_linear({
            ntype: node_feats[ntype][node_tids[ntype]] 
            for ntype in node_tids
        })
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            old_h = h
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = apply_each(h, self.dropout)

            # Residual connection
            if self.residual:
                for nt in h:
                    h[nt] += old_h[nt][:block.num_dst_nodes(nt)]
        return h
    

    @torch.no_grad()
    def inference(self, hg, node_feats, batch_size, device, num_workers=4):
        x = self.hetero_linear(node_feats)
        for l, layer in enumerate(self.layers):
            y = {
                k: torch.zeros(
                    hg.number_of_nodes(k),
                    self.hidden_dim if l != len(self.layers) - 1 else self.out_dim,
                )
                for k in hg.ntypes
            }

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                hg,
                {k: torch.arange(hg.number_of_nodes(k)).to(device) for k in hg.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                device=device,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                if isinstance(input_nodes, torch.Tensor):
                    input_nodes  = {hg.ntypes[0]: input_nodes}
                    # output_nodes = {hg.ntypes[0]: output_nodes}

                block = blocks[0].to(device)

                h = {
                    k: x[k][input_nodes[k]]
                    for k in input_nodes.keys()
                }

                old_h = h
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = apply_each(h, self.dropout)

                # Residual connection
                if self.residual:
                    for nt in h:
                        h[nt] += old_h[nt][:block.num_dst_nodes(nt)]

                for k in output_nodes.keys():
                    y[k][output_nodes[k]] = \
                        h[k].cpu() 

            x = { k: v.to(device) for k, v in y.items() }
        return y
