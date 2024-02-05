import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import apply_each
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import  dgl.nn.pytorch as dglnn

class HGTLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads,
        dropout=0.2,
        use_norm=False,
    ):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.n_heads)
        )
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        if G.is_block:
            h_src = h
            h_dst = {
                k: v[: G.number_of_dst_nodes(k)] for k, v in h.items()
            }
        else:
            h_src = h
            h_dst = h
        
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h_src[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h_src[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h_dst[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1)
                    * relation_pri
                    / self.sqrt_dk
                )
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"),
                        fn.sum("m", "t"),
                    )
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="sum",
            )
            
            new_h = {}
            rst = G.dstdata["t"]

            for ntype in rst:
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = rst[ntype].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h_dst[ntype] * (1 - alpha)
                
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(
        self,
        hg, 
        node_feat_dim_dict,
        embed_size,
        hidden_size,
        out_size,
        num_hidden_layers,
        n_heads,
        dropout=0.2,
        layer_norm=True,
    ):
        super(HGT, self).__init__()
        self.embed_size = embed_size
        self.node_dict = {}
        self.edge_dict = {}
        for ntype in hg.ntypes:
            self.node_dict[ntype] = len(self.node_dict)
        for etype in hg.etypes:
            self.edge_dict[etype] = len(self.edge_dict)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.n_heads = n_heads
        self.layer_norm = layer_norm

        self.hetero_linear = dglnn.HeteroLinear(node_feat_dim_dict, embed_size, bias=True)

        # Input layer
        self.layers = nn.ModuleList()
        self.layers.append(
            HGTLayer(
                in_dim=self.embed_size,
                out_dim=self.hidden_size,
                node_dict=self.node_dict,
                edge_dict=self.edge_dict,
                n_heads=self.n_heads,
                dropout=self.dropout,
                use_norm=self.layer_norm,
            )
        )

        # Hidden layers
        for _ in range(self.num_hidden_layers):
            self.layers.append(
                HGTLayer(
                    in_dim=self.hidden_size,
                    out_dim=self.hidden_size,
                    node_dict=self.node_dict,
                    edge_dict=self.edge_dict,
                    n_heads=self.n_heads,
                    dropout=self.dropout,
                    use_norm=self.layer_norm,
                )
            )
        
        # Output layer
        self.layers.append(
            HGTLayer(
                in_dim=self.hidden_size,
                out_dim=self.out_size,
                node_dict=self.node_dict,
                edge_dict=self.edge_dict,
                n_heads=self.n_heads,
                dropout=0,
                use_norm=self.layer_norm,
            )
        )

    def forward(self, blocks, node_feats, node_tids=None):
        if node_tids is None:
            node_tids = blocks[0].srcdata[dgl.NID]
        
        h = self.hetero_linear({ntype: node_feats[ntype][node_tids[ntype]] for ntype in node_tids})
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h
    