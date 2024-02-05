import torch
import torch.nn.functional as F
import dgl

from parameter_namespace import \
        GeneralParameterNamespace, \
        HetGATParameterNamespace, \
        HetGCNParameterNamespace

from utils import GraphLoader, \
    prepare_dataloader_for_lp, \
    print_detail_for_lp, \
    calculate_lp_accuracy

from model_4 import GammaModel, GammaScorer

import pickle
import numpy as np
import time

pn = GeneralParameterNamespace('gamma')
gat_pn = HetGATParameterNamespace()
gcn_pn = HetGCNParameterNamespace()

EMB_FEATURES = 64

def main(): 

    # region 

    # STEP 1. Preparing
    gl = GraphLoader(graph_path=pn.graph_interaction_path)

    gh, node_feats, edge2ids = gl.load_graph(device=pn.device)
    node_feat_dim_dict  = {ntype: node_feats[ntype].shape[1] for ntype in gh.ntypes}

    node_feat_dim_dict2 = {k: v for k, v in node_feat_dim_dict.items()}
    node_feat_dim_dict2['contributor'] = 2 * EMB_FEATURES
    node_feat_dim_dict2['repository']  = 2 * EMB_FEATURES

    g_inter = gh.edge_type_subgraph([
        'contributor_propose_issue',
        'contributor_propose_pr',
        'contributor_star_repo',
        'contributor_watch_repo',
        'issue_belong_to_repo',
        'pr_belong_to_repo',
        'repo_committed_by_contributor'
    ])
    g_social = gh.edge_type_subgraph(['contributor_follow_contributor'])


    # STEP 2. Loading data
    train_dl, val_dl, test_dl = prepare_dataloader_for_lp(
        g_inter, 
        fanouts=gcn_pn.fanouts,
        batch_size=pn.lp_batch_size,
        negative_samples=pn.negative_samples, 
        use_gpu=pn.use_gpu,
        device=pn.device,
        dropout=0
    )

    # STEP 3. model, scorer, optimizer
    model = GammaModel(
        g_inter=g_inter, 
        g_social=g_social, 
        device=pn.device,
        node_feat_dim_dict=node_feat_dim_dict,
        node_feat_dim_dict2=node_feat_dim_dict2
    )
    scorer = GammaScorer(in_features=EMB_FEATURES).to(pn.device)

    no_decay_params = []
    decay_params = []
    get_decay_parameters(model, no_decay_params, decay_params)
    get_decay_parameters(scorer, no_decay_params, decay_params)
    opt = torch.optim.Adam([
        {
            "params": no_decay_params,
        },
        {
            "params": decay_params,
            "weight_decay": pn.weight_decay
        }
    ], lr=pn.lr)

    # STEP 4. Train & Valid
    print('start training.')

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=pn.lr_decay)

    # endregion

    for epoch in range(pn.epochs): 
        epoch += 1
        print(f'[{currentTime()}] epoch {epoch} / {pn.epochs}')

        # TRAIN
        losses, rights, totals = [], 0, 0
        for i, (loss, pred_rights, pred_totals) in enumerate(train(train_dl, model, scorer, node_feats, opt)): 
            losses.append(loss)
            rights += pred_rights
            totals += pred_totals

            if i % 50 == 0: 
                print(f'[{currentTime()}] [train {i:6d}/{len(train_dl)}] loss = {np.mean(losses):.4f}, accuracy = {rights / totals:.4f}')

        scheduler.step()

        # VALID
        val_loss, val_acc, val_cate_rights, val_cate_totals = evaluate(val_dl, model, scorer, node_feats)
        print(f'Train epoch {epoch:2d}, lr = {opt.param_groups[0]["lr"]:.4f}, train_loss = {np.mean(losses):.4f}, val loss = {val_loss:.4f}, accuracy = {val_acc:.4}')
        print_detail_for_lp(val_cate_rights, val_cate_totals, None)

        torch.save(model.state_dict(), pn.generate_model_state_file(epoch))
        torch.save(scorer.state_dict(), pn.generate_lp_scorer_state_file(epoch))
        print('Model saved')

    # TEST
    print('testing')
    test_loss, test_acc, test_cate_rights, test_cate_totals = evaluate(test_dl, model, scorer, node_feats)
    print(f'Test result: lr = {opt.param_groups[0]["lr"]}, test_loss = {np.mean(losses)}, val loss = {val_loss}, accuracy = {val_acc:.4}')
    print(f'test_loss = {test_loss}, accuracy = {test_acc:.4}')
    print_detail_for_lp(test_cate_rights, test_cate_totals, None)



def get_decay_parameters(module:torch.nn.Module, no_decay_params:list, decay_params:list):
    for name, param in module.named_parameters():
        if "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

def go_through(data, model, scorer, node_feats):

    _, pg, ng, blks = data 
    pg, ng = pg.to(pn.device), ng.to(pn.device)
    blks = [b.to(pn.device) for b in blks]

    result = model(blks, node_feats)

    scores, labels = get_scores_and_labels(scorer, pg, ng, result)
    loss = F.binary_cross_entropy(scores, labels)

    return scores, labels, loss

def train(dataloader, model, scorer, node_feats, opt): 
    cate_rights, cate_totals = {}, {}
    model.train()
    scorer.train()

    for data in dataloader: 
        scores, labels, loss = go_through(data, model, scorer, node_feats)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pred_rights = calculate_lp_accuracy(scores, labels, cate_rights, cate_totals)
        pred_totals = len(scores)

        yield loss.item(), pred_rights, pred_totals

def evaluate(dataloader, model, scorer, node_feats):
    model.eval()
    scorer.eval()

    def sub(): 
        with torch.no_grad():
            for i, data in enumerate(dataloader): 
                scores, labels, loss = go_through(data, model, scorer, node_feats)

                pred_rights = calculate_lp_accuracy(scores, labels, cate_rights, cate_totals)
                pred_totals = len(scores)

                yield loss.item(), pred_rights, pred_totals
                if i % 50 == 0: 
                    print(f'[{currentTime()}] [test/valuation {i:6d}/{len(dataloader)}]')
    
    val_losses, val_rights, val_totals = [], 0, 0
    cate_rights, cate_totals = {}, {}
    for loss, pred_rights, pred_totals in sub(): 
        val_losses.append(loss)
        val_rights += pred_rights
        val_totals += pred_totals

    return np.mean(val_losses), val_rights / val_totals, cate_rights, cate_totals

def get_scores_and_labels(scorer, pg, ng, result): 
    def f(graph): # returns an 1d-tensor
        ret = scorer(graph, result)
        # ret = ret[('repository', 'repo_committed_by_contributor', 'contributor')]
        ret = torch.cat([v for v in ret.values()])
        ret = ret.squeeze()
        return ret

    scores = [f(ng), f(pg)]
    labels = torch.cat([
        torch.zeros_like(scores[0]), 
        torch.ones_like(scores[1])
    ])
    scores = torch.cat(scores, dim=-1)

    return scores, labels


def currentTime(): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

if __name__ == '__main__': 
    main()