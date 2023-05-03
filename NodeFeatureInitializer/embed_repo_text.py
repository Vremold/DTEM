import os
import sys
import json

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from embedder_text import embed_text, MyModel

if __name__ == "__main__":
    cpu_count = 16
    pretrained_model_name = "xlm-roberta-base"
    batch_size = 16
    device = torch.device("cuda")

    src_file = "./RepositoryEmbedding/repo_descriptions.txt"
    dst_file = "./RepositoryEmbedding/repo_description_embedding.txt"

    model = MyModel(pretrained_bert_model=pretrained_model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    embed_text(model, tokenizer, src_file, dst_file, batch_size=batch_size, device=device)