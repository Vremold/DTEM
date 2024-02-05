#!/usr/bin/env python3 

'''
这是 alpha 对比实验的核心. 
在前面的工作中, 我们获取到了开发者的嵌入向量,
这一步, 我们将利用这些嵌入向量, 训练一个三层的简单模型, 
看看效果如何. 

从这一步开始, 后面的工作都参考学长原有的 "相似开发者推荐" 的工作了. 

@see also: 
    RecommendationTasks/SimDeveloper/train_nn.py
    RecommendationTasks/SimDeveloper/* 
'''

from RecommendationTasks.SimDeveloper.train_nn import \
        Net as SimDeveloperNet, \
        MyDataset as DataSet, \
        DataLoader, \
        collate_fn, metric


from ..general import \
    load_yaml_cfg

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from time import strftime

cfg = load_yaml_cfg()['alpha']
task_cfg = cfg['tasks']['sim_developer']

def main(device=torch.device('cpu')): 

    # STEP 1. Prepare dataset (sample indices and embeddings)
    embs = torch.load(cfg['embedding']['contributor_merged_embedding'])

    def get_dataloader(samples, embedding, batch_size=32, shuffle=True, collate_fn=collate_fn): 
        dataset = DataSet(samples=samples, node_embedding_obj=embedding)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    # dl: data loader
    train_dl = get_dataloader(task_cfg['data']['train_file'], embs)
    eval_dl  = get_dataloader(task_cfg['data']['valid_file'], embs)
    test_dl  = get_dataloader(task_cfg['data']['test_file'],  embs)


    # STEP 2. Prepare model
    in_dim = embs[0].shape[0] # in_dim = 580 = 230 (repository) + 150 (issue) + 200 (api)
    model = SimDeveloperNet(embedding_dim=in_dim).to(device)


    # STEP 3. Prepare optimizer, train & valid
    # (exactly copied from RecommendationTasks/SimDeveloper/train_nn.py, changed some parameter names (BAD MANNER!))
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCELoss()
    best_f1 = 0
    epochs = 60
    # epochs = 1

    for epoch in range(epochs):
        print(f'>>== {strftime("%Y-%m-%d %H:%M:%S")}, epoch {epoch + 1}')
        running_loss = 0.0
        model.train()
        for _, (x, labels) in enumerate(train_dl):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: loss={running_loss / len(train_dl)}")

        # Validation
        model.eval()
        pos_rights = 0
        neg_rights = 0
        pos_totals = 0
        neg_totals = 0

        with torch.no_grad():
            for _, (x, labels) in enumerate(eval_dl):
                x, labels = x.to(device), labels.to(device)
                logits = model(x)
                pred = (logits > 0.5).long()
                pos_right = ((pred == 1) & (labels == 1)).sum().item()
                neg_right = ((pred == 0) & (labels == 0)).sum().item()
                pos_total = (labels == 1).sum().item()
                neg_total = (labels == 0).sum().item()

                pos_rights += pos_right
                neg_rights += neg_right
                pos_totals += pos_total
                neg_totals += neg_total

        precision, recall, f1 = metric(pos_rights, neg_rights, pos_totals, neg_totals)
        print(f"Epoch {epoch+1}: precision={precision}, recall={recall}, f1={f1}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), task_cfg['model']['model_file'])
            print("Model saved.")


    # STEP 4. Test
    model.load_state_dict(torch.load(task_cfg['model']['model_file']))
    model.eval()
    pos_rights = 0
    neg_rights = 0
    pos_totals = 0
    neg_totals = 0
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_dl):
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            pred = (logits > 0.5).long()
            pos_right = ((pred == 1) & (labels == 1)).sum().item()
            neg_right = ((pred == 0) & (labels == 0)).sum().item()
            pos_total = (labels == 1).sum().item()
            neg_total = (labels == 0).sum().item()

            pos_rights += pos_right
            neg_rights += neg_right
            pos_totals += pos_total
            neg_totals += neg_total
    precision, recall, f1 = metric(pos_rights, neg_rights, pos_totals, neg_totals)
    print(f"Test: precision={precision}, recall={recall}, f1={f1}")

main(device=torch.device('cuda:0'))