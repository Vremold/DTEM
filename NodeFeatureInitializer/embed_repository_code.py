import os
import json
import pickle
import multiprocessing
import argparse

import torch
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

from embedder_code import embed_code_snippet, Model

parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int, default=1)
parser.add_argument("--n_process", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=512)

args = parser.parse_args()

if __name__ == "__main__":
    cpu_count = args.n_process
    config_name = "microsoft/graphcodebert-base"
    tokenizer_name = "microsoft/graphcodebert-base"
    model_name_or_path = "microsoft/graphcodebert-base"
    code_length = 256
    data_flow_length = 64
    eval_batch_size = args.batch_size

    pool = multiprocessing.Pool(cpu_count)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # model configuration
    config = RobertaConfig.from_pretrained(config_name)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    model = RobertaModel.from_pretrained(model_name_or_path)
    model = Model(model)
    model = model.to(device)

    src_dir = "./RepositoryCodeEmbedding/codes"
    dst_dir = "./RepositoryCodeEmbedding/result"

    task_fnames = set(os.listdir(src_dir))
    finished_task_fnames = set()
    for fname in os.listdir(dst_dir):
        if fname.endswith("path_embedding.pkl"):
            finished_task_fnames.add(fname.split("_project")[0]+".jsonl")
    task_fnames = list(task_fnames - finished_task_fnames)
    task_fnames.sort()

    for filename in task_fnames:
        if "x" in filename:
            continue
        eval_data_file = f"{src_dir}/{filename}"
        dst_file = f"{dst_dir}/{filename}"
        lang = filename.split("_")[0]
        print("Now processing", filename, "...")
        project_path_embedding, project_path_n_funcs = embed_code_snippet(model, tokenizer, eval_data_file, pool, lang, code_length, data_flow_length, eval_batch_size, device)
        filename = filename.split(".")[0]
        with open(f"{dst_dir}/{filename}_project_path_embedding.pkl", "wb") as outf:
            pickle.dump(project_path_embedding, outf)
        with open(f"{dst_dir}/{filename}_project_path_n_funcs.json", "w", encoding="utf-8") as outf:
            json.dump(project_path_n_funcs, outf, ensure_ascii=False)
