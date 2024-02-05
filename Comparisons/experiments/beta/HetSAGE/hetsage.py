import sys
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import apply_each
import dgl.nn.pytorch as dglnn

class HetSAGE(torch.nn.Module):
    def __init__(
        self,
        hg, 
        node_feat_dim_dict,
        embed_size,
        hidden_dim,
        out_dim, 
        num_hidden_layers=1,
        feat_drop=0.2,
    ) -> None:
        super().__init__()

        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.feat_drop = feat_drop

        # Node initial embedding transformation
        self.hetero_linear = dglnn.HeteroLinear(node_feat_dim_dict, embed_size, bias=True)
        self.layers = torch.nn.ModuleList()
        
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(
                in_feats=self.embed_size,
                out_feats=self.hidden_dim,
                aggregator_type="pool",
                feat_drop=self.feat_drop,
                bias=True,
                activation=F.relu,
                norm=nn.LayerNorm(self.hidden_dim)
            )
            for etype in hg.etypes
        }))

        for _ in range(self.num_hidden_layers):
            self.layers.append(dglnn.HeteroGraphConv({
                etype: dglnn.SAGEConv(
                    in_feats=self.hidden_dim,
                    out_feats=self.hidden_dim,
                    aggregator_type="pool",
                    feat_drop=self.feat_drop,
                    bias=True,
                    activation=F.relu,
                    norm=nn.LayerNorm(self.hidden_dim)
                )
                for etype in hg.etypes
            }))
        
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(
                in_feats=self.hidden_dim,
                out_feats=self.out_dim,
                aggregator_type="pool",
                feat_drop=self.feat_drop,
                bias=True,
                activation=None,
                norm=nn.LayerNorm(self.out_dim)
            )
            for etype in hg.etypes
        }))

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
                {k: torch.arange(hg.number_of_nodes(k)) for k in hg.ntypes},
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