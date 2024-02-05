from networkx import relabel_nodes
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLError

from parameter_namespace import \
    GeneralParameterNamespace, \
    HetGATParameterNamespace, \
    HetGCNParameterNamespace
from models.hetgat import HetGAT 
from models.hetgcn import HetGCN

from utils import GraphLoader, prepare_dataloader_for_lp

pn = GeneralParameterNamespace('gamma')
gat_pn = HetGATParameterNamespace()
gcn_pn = HetGCNParameterNamespace()

class Attention(nn.Module): 

    def __init__(self, n_dims, batch_first=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_dims, 1, batch_first=batch_first)
        self.n_dims = n_dims

    def forward(self, q, k, v): 
        assert q.shape[-1] == self.n_dims
        assert k.shape[-1] == self.n_dims
        assert v.shape[-1] == self.n_dims
        output, _  = self.attn(q, k, v)
        return output


class GammaModel(nn.Module): 
    
    def __init__(self, 
        g_inter, g_social, 
        device, 
        node_feat_dim_dict, 
        node_feat_dim_dict2
    ): 
        super().__init__()

        USER_FEATURES, ITEM_FEATURES = 256, 2304
        EMB_FEATURES = 64

        self.g_social = g_social
        self.g_inter  = g_inter
        
        self.m_inter = HetGCN(
            hg=g_inter,
            node_feat_dim_dict=node_feat_dim_dict,
            embed_size=gcn_pn.embed_size,
            hidden_dim=gcn_pn.hidden_size,
            out_dim=gcn_pn.out_feats,
            num_hidden_layers=gcn_pn.num_hidden_layers,
            dropout=gcn_pn.dropout, 
            residual=gcn_pn.residual
        ).to(pn.device)

        self.m_social = HetGCN(
            hg=g_social,
            node_feat_dim_dict={'contributor': gcn_pn.out_feats},
            embed_size=gcn_pn.embed_size,
            hidden_dim=gcn_pn.hidden_size,
            out_dim=gcn_pn.out_feats,
            num_hidden_layers=gcn_pn.num_hidden_layers,
            dropout=gcn_pn.dropout, 
            residual=gcn_pn.residual
        ).to(pn.device)

        self.m_fin = HetGAT(
            hg=g_inter,
            node_feat_dim_dict=node_feat_dim_dict2,
            embed_size=EMB_FEATURES, # pn.embed_size,
            hidden_dim=EMB_FEATURES, # pn.hidden_size,
            out_dim=64,  # pn.out_feats,
            num_heads=gat_pn.num_heads,
            num_hidden_layers=gat_pn.num_hidden_layers,
            residual=gat_pn.residual,
            feat_drop=gat_pn.feat_drop,
            attn_drop=gat_pn.attn_drop,
            negative_slope=gat_pn.negative_slope
        ).to(pn.device)

        self.u_head = nn.Linear(USER_FEATURES, EMB_FEATURES).to(pn.device)
        self.v_head = nn.Linear(ITEM_FEATURES, EMB_FEATURES).to(pn.device)
        self.mock_x_head = nn.Linear(512, EMB_FEATURES).to(pn.device)

        self.HUR = Attention(n_dims=EMB_FEATURES).to(pn.device)
        self.UVR = Attention(n_dims=EMB_FEATURES).to(pn.device)
        self.VUR = Attention(n_dims=EMB_FEATURES).to(pn.device)

        self.sampler = dgl.dataloading.NeighborSampler(gcn_pn.fanouts, replace=False)


    def forward(self, blks, node_feats): 

        g_sub = self.g_social.subgraph(GammaModel.get_cids(blks))
        _, _, blks2 = self.sampler.sample(g_sub, g_sub.nodes())

        g_sub2 = self.g_inter.subgraph({'contributor': GammaModel.get_cids(blks2)})
        _, _, blks3 = self.sampler.sample(g_sub2, {
            'contributor': g_sub2.nodes('contributor')
        })
        
        x_inter = self.m_inter(blks3, node_feats)
        X = self.m_social(blks2, x_inter, {
            n_type: range(len(x_inter[n_type]))
            for n_type in ['contributor']
        })['contributor']

        u, v = {
            node_feats[it][blks[0].srcdata[dgl.NID][it]]
            for it in ['contributor', 'repository']
        }
        u, v = self.u_head(u), self.v_head(v)
        X = self.mock_x_head(X)

        X  = X.unsqueeze(1)
        uu = u.unsqueeze(1)
        vv = v.unsqueeze(1)


        f_uus = self.HUR(uu, X, X)
        e_uv = self.UVR(vv, f_uus, f_uus).squeeze(1)
        e_vu = self.VUR(f_uus, vv, vv).squeeze(1)

        u2 = torch.cat((u, e_vu), dim=1)
        v2 = torch.cat((v, e_uv), dim=1)

        node_feats2 = {k: v for k, v in node_feats.items()}
        node_feats2['contributor'] = u2
        node_feats2['repository']  = v2

        node_tids = blks[0].srcdata[dgl.NID]
        node_tids['contributor'] = range(len(u2))
        node_tids['repository']  = range(len(v2))

        return self.m_fin(blks, node_feats2, node_tids)

    @staticmethod
    def get_cids(blks): 
        data = blks[0].srcdata[dgl.NID]
        if isinstance(data, dict): 
            return data['contributor']
        else: 
            return data


    @torch.no_grad()
    def inference(self, node_feats, num_workers=4): 
        # Not implemented.  
        # You can implement this by yourself
        # by reading  dump_node_embedding.ipynb. 
        raise NotImplementedError


class GammaScorer(nn.Module): 

    def __init__(self, in_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def apply_edges(self, edges):
        score = self.linear(edges.src["x"] * edges.dst["x"])
        return {"score": self.sigmoid(score)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["x"] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(self.apply_edges, etype=etype)
            return edge_subgraph.edata["score"]
  
