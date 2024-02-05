"""
Borrowed from microsoft/GraphCodeBERT
"""
import sys
import os
import logging
import pickle
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           RobertaConfig, RobertaModel, RobertaTokenizer)
import numpy as np
from tree_sitter import Language, Parser

# from model import Model
from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)

class Model(torch.nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None):
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.encoder.embeddings.word_embeddings(
                code_inputs)
            nodes_to_token_mask = nodes_mask[:, :,
                                             None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / \
                (nodes_to_token_mask.sum(-1)+1e-10)[:, :, None]
            avg_embeddings = torch.einsum(
                "abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * \
                (~nodes_mask)[:, :, None]+avg_embeddings*nodes_mask[:, :, None]
            return self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]


logger = logging.getLogger(__name__)

cpu_count = 16

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser

# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    code_tokens, dfg = [], []
    if lang == "php":
        code = "<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 project,
                 path
                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.project = project
        self.path = path


def convert_code_to_feature(code, project, path, tokenizer, lang, code_length, data_flow_length):
    # parser
    parser = parsers[lang]

    # extract dataflow
    code_tokens, dfg = extract_dataflow(code=code, parser=parser, lang=lang)
    code_tokens = [tokenizer.tokenize(
        '@ '+x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i-1][1],
                          ori2cur_pos[i-1][1]+len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    # truncating
    code_tokens = code_tokens[:code_length +
                              data_flow_length - 2 - min(len(dfg), data_flow_length)]

    # constructing code_ids input for neural networks
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i + tokenizer.pad_token_id +
                    1 for i in range(len(code_tokens))]
    dfg = dfg[:code_length + data_flow_length - len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for _ in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = code_length + data_flow_length - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length

    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i]
                              for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]

    return InputFeatures(code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg, project, path)


def convert_examples_to_features(item):
    code, project, path, tokenizer, lang, code_length, data_flow_length = item
    return convert_code_to_feature(code, project, path, tokenizer, lang, code_length, data_flow_length)


class TextDataset(Dataset):
    def __init__(self, tokenizer, lang, code_length, data_flow_length, filepath=None, pool=None, cache_file="./cache_javascript_7_6.pkl"):
        self.lang = lang
        self.code_length = code_length
        self.data_flow_length = data_flow_length

        self.examples = list()

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as inf:
                self.examples = pickle.load(inf)
        data = list()
        if type(filepath) == str:
            with open(filepath) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    # print(obj["code"])
                    data.append((obj["code"], obj["project"], obj["path"], tokenizer, lang, code_length, data_flow_length))
        else:
            for item in filepath:
                data.append((item["code"], item["project"], obj["path"], tokenizer, lang, code_length, data_flow_length))

        self.examples = pool.map(
            convert_examples_to_features, tqdm(data, total=len(data)))
        pickle.dump(self.examples, open(cache_file, 'wb'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.code_length+self.data_flow_length,
                              self.code_length+self.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx+node_index, a:b] = True
                attn_mask[a:b, idx+node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index < len(self.examples[item].position_idx):
                    attn_mask[idx+node_index, a+node_index] = True

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                self.examples[item].project,
                self.examples[item].path)


def embed_code_snippet(model, tokenizer, code_file_name, pool, lang, code_length, data_flow_length, batch_size, device):
    code_dataset = TextDataset(
        tokenizer, lang, code_length, data_flow_length, code_file_name, pool, cache_file=code_file_name+".pkl")
    # sys.exit(0)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(
        code_dataset, sampler=code_sampler, batch_size=batch_size, num_workers=4)
    
    project_path_embedding = {}
    project_path_n_funcs   = {}

    model.eval()
    print("Start running the model...")
    for batch in code_dataloader:
        code_inputs = batch[0].to(device)
        attn_mask = batch[1].to(device)
        position_idx = batch[2].to(device)

        projects = batch[3]
        paths    = batch[4]
        with torch.no_grad():
            code_vecs = model(code_inputs=code_inputs,
                              attn_mask=attn_mask, position_idx=position_idx)
            code_vecs = code_vecs.cpu().numpy()
            
            for project, path, vec in zip(projects, paths, code_vecs):
                if project not in project_path_embedding:
                    project_path_embedding[project] = {}
                    project_path_n_funcs[project] = {}
                if path not in project_path_embedding[project]:
                    project_path_embedding[project][path] = np.zeros(vec.shape, dtype=vec.dtype)
                    project_path_n_funcs[project][path] = 0
                project_path_embedding[project][path] += vec
                project_path_n_funcs[project][path] += 1
        
    return project_path_embedding, project_path_n_funcs
