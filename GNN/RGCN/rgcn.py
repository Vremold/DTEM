import sys
import tqdm
import math

import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each
import dgl.function as fn
import dgl.nn.pytorch as dglnn

class RelGraphConv(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        bias=True,
        activation=None,
        self_loop=True,
        dropout=0.0,
        layer_norm=False,
        norm="none") -> None:
        super().__init__()

        if norm not in ["none", "left", "right", "both"]:
            raise ValueError(
                "Invalid norm value. Must be either 'none', 'left', 'right' or 'both'."
            )
    
        self.norm = norm
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # relation weight
        self.weight = nn.Parameter(torch.Tensor(in_feat, out_feat))

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
        
        # layernorm
        if self.layer_norm:
            self.layer_norm_layer = nn.LayerNorm(
                out_feat, elementwise_affine=True
            )
        
        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.bias:
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )
        nn.init.xavier_uniform_(
            self.weight, gain=nn.init.calculate_gain("relu")
        )

    def message(self, edges):
        """Message function."""
        m = torch.matmul(edges.src["h"], self.weight)
        return {"m": m}
    
    def forward(self, g, feat):
        with g.local_scope():
            if isinstance(feat, tuple):
                feat_src, feat_dst = feat
            else:
                feat_src = feat_dst = feat
            # Normalize node embeddings according to the outdegree.
            if self.norm in ['left', 'both']:
                degs = g.out_degrees().to(feat_src).clamp(min=1)
                if self.norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            g.srcdata["h"] = feat_src
            
            # message passing
            g.update_all(self.message, fn.sum("m", "h"))
            h = g.dstdata["h"]

            # Normalize node embeddings according to the indegree.
            if self.norm in ['right', 'both']:
                degs = g.in_degrees().to(feat_dst).clamp(min=1)
                if self.norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                h = h * norm
            
            # apply bias and activation
            h = g.dstdata["h"]
            if self.layer_norm:
                h = self.layer_norm_layer(h)
            if self.bias:
                h = h + self.h_bias
            
            # Kind of like residual connection
            if self.self_loop:
                h = h + feat_dst @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h

class RGCN(nn.Module):
    def __init__(
        self,
        hg,
        node_feat_dim_dict,
        embed_size, 
        hidden_dim,
        out_dim,
        num_hidden_layers=1,
        dropout=0.2,
        self_loop=True,
        layer_norm=False,
        ) -> None:
        super().__init__()

        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # Node initial embedding transformation
        self.hetero_linear = dglnn.HeteroLinear(node_feat_dim_dict, embed_size, bias=True)
        self.layers = torch.nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            etype: RelGraphConv(
                in_feat=self.embed_size,
                out_feat=self.hidden_dim,
                bias=True,
                activation=F.relu,
                self_loop=self.self_loop,
                dropout=self.dropout,
                layer_norm=self.layer_norm,
                norm="right"
            ) 
            for etype in hg.etypes}))

        for _ in range(self.num_hidden_layers):
            self.layers.append(dglnn.HeteroGraphConv({
                etype: RelGraphConv(
                    in_feat=self.hidden_dim,
                    out_feat=self.hidden_dim,
                    bias=True,
                    activation=F.relu,
                    self_loop=self.self_loop,
                    dropout=self.dropout,
                    layer_norm=self.layer_norm,
                    norm="right"
                )
                for etype in hg.etypes
            }))
        
        self.layers.append(dglnn.HeteroGraphConv({
            etype: RelGraphConv(
                in_feat=self.hidden_dim,
                out_feat=self.out_dim,
                bias=True,
                activation=None,
                self_loop=self.self_loop,
                dropout=0,
                layer_norm=self.layer_norm,
                norm="right"
            )
            for etype in hg.etypes
        }))

    # 记录一个bug，当图中的某一类型的节点只有出边时，message passing不会更新这个节点的表示
    # 于是这个节点也不会在我们网络的返回结果中，但是采样子图中存在这个类型的节点，后面的应用就会出错
    # 具体体现在LPscorer中，可能需要allow_zero_in_degree来处理
    def forward(self, blocks, node_feats, node_tids=None):
        if node_tids is None:
            node_tids = blocks[0].srcdata[dgl.NID]

        h = self.hetero_linear({ntype: node_feats[ntype][node_tids[ntype]] for ntype in node_tids})
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
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
                {k: torch.arange(hg.number_of_nodes(k), device=device) for k in hg.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                device=device,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)

                h = {
                    k: x[k][input_nodes[k]].to(device)
                    for k in input_nodes.keys()
                }
                h = layer(block, h)

                for k in output_nodes.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y
