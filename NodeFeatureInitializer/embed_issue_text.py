import os
import sys
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from embedder_text import embed_text, MyModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    cpu_count = 16
    pretrained_model_name = "xlm-roberta-base"
    # batch_size = 64
    device = torch.device("cuda:0")

    src_file = "./IssueEmbedding/issue_descriptions.txt"
    dst_file = "./IssueEmbedding/issue_description_embedding.txt"

    model = MyModel(pretrained_bert_model=pretrained_model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    embed_text(model, tokenizer, src_file, dst_file, batch_size=args.batch_size, device=device)