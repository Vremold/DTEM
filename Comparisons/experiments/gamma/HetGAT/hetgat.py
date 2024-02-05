import sys
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import apply_each
import dgl.nn.pytorch as dglnn

class HetGAT(torch.nn.Module):
    def __init__(
        self,
        hg, 
        node_feat_dim_dict,
        embed_size,
        hidden_dim,
        out_dim, 
        num_heads=4,
        num_hidden_layers=1,
        residual=True,
        feat_drop=0.2,
        attn_drop=0.2,
        negative_slope=0.2,
        edge_feat_dim=None,
    ) -> None:
        super().__init__()

        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.residual = residual
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope

        # Node initial embedding transformation
        self.hetero_linear = dglnn.HeteroLinear(node_feat_dim_dict, embed_size, bias=True)
        self.layers = torch.nn.ModuleList()
        
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.GATConv(
                in_feats=self.embed_size,
                out_feats=self.hidden_dim // self.num_heads,
                num_heads=self.num_heads,
                feat_drop=self.feat_drop,
                attn_drop=self.attn_drop,
                residual=self.residual,
                activation=F.relu,
                negative_slope=self.negative_slope,
                allow_zero_in_degree=True,
                bias=True
            )
            for etype in hg.etypes
        }))

        for _ in range(self.num_hidden_layers):
            self.layers.append(dglnn.HeteroGraphConv({
                etype: dglnn.GATConv(
                    in_feats=self.hidden_dim,
                    out_feats=self.hidden_dim // self.num_heads,
                    num_heads=self.num_heads,
                    feat_drop=self.feat_drop,
                    attn_drop=self.attn_drop,
                    residual=self.residual,
                    activation=F.relu,
                    negative_slope=self.negative_slope,
                    allow_zero_in_degree=True,
                    bias=True
                )
                for etype in hg.etypes
            }))
        
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.GATConv(
                in_feats=self.hidden_dim,
                out_feats=self.out_dim // self.num_heads,
                num_heads=self.num_heads,
                feat_drop=self.feat_drop,
                attn_drop=self.attn_drop,
                residual=self.residual,
                activation=None,
                negative_slope=self.negative_slope,
                allow_zero_in_degree=True,
                bias=True
            )
            for etype in hg.etypes
        }))

    def forward(self, blocks, node_feats, node_tids=None, edge_feats=None):
        if node_tids is None:
            node_tids = blocks[0].srcdata[dgl.NID]
        h = self.hetero_linear({ntype: node_feats[ntype][node_tids[ntype]] for ntype in node_tids})
        
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
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
                block = blocks[0].to(device)

                h = {
                    k: x[k][input_nodes[k]].to(device)
                    for k in input_nodes.keys()
                }
                h = layer(block, h)
                h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))

                for k in output_nodes.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = { k: v.to(device) for k, v in y.items() }
        return y