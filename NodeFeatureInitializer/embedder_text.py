import os
import json
import pickle
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

PADDING_VALUE = 0

class MyModel(torch.nn.Module):
    def __init__(self, pretrained_bert_model) -> None:
        super().__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(pretrained_bert_model, output_hidden_states=True)
    
    def forward(self, input_ids, attention_mask):
        # The explanation of the output: https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.MaskedLMOutput
        bert_output = self.bert(input_ids, attention_mask)
        # bsz * 768
        return bert_output["hidden_states"][-1][:, 0, :]


# https://huggingface.co/xlm-roberta-base
# XLM-RoBERTa is a multilingual version of RoBERTa. It is pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages.
class TextDataset(Dataset):
    def __init__(self, text_filepath, tokenizer, max_length) -> None:
        super().__init__()

        self.data = []
        with open(text_filepath, "r", encoding="utf-8") as inf:
            for line in inf:
                # 需要数据每行为一个python dict，
                # 且必须包含键值 text
                text = json.loads(line)["text"]
                # {"input_ids": [[tenosr element]], "attention_mask": [[tensor element]]}
                encoded_input = tokenizer(text, return_tensors="pt")
                input_ids =      encoded_input["input_ids"][0][:max_length]
                attention_mask = encoded_input["attention_mask"][0][:max_length]
                if len(input_ids) > 512:
                    print("####", len(input_ids))
                # print(len(input_ids))
                # print(len(attention_mask))
                self.data.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def embed_text(model:torch.nn.Module, tokenizer, text_filepath, out_filepath, batch_size, device):

    outf = open(out_filepath, "w", encoding="utf-8")
    text_dataset = TextDataset(text_filepath, tokenizer, max_length=512)
    text_sampler = SequentialSampler(text_dataset)
    text_dataloader = DataLoader(text_dataset, sampler=text_sampler, batch_size=batch_size, collate_fn=collate_fn)
    
    model.eval()

    for batch in text_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            text_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
            outf.write(json.dumps(text_embeddings.cpu().numpy().tolist()) + "\n")

    outf.close()


if __name__ == "__main__":
    model = MyModel(pretrained_bert_model="xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="xlm-roberta-base")
    embed_text(model, tokenizer, text_filepath="./test.jsonl", out_filepath="./test", batch_size=32, device=torch.device("cpu"))