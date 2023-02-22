import os

import sys
import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer, DistilBertForMaskedLM, DistilBertTokenizer
sys.path.append("./examples/python/pytorch/transformers")
from transformer_utils import set_seed, get_nsp_data_loader

BASE_DIR = "examples/python/pytorch/bert"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOADER_DIR = os.path.join(BASE_DIR, "loader")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained(
    'distilbert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['distilbert-base-uncased']


def train(epoch, model, device, loader, optimizer):
    model.train()
    print("----------start training----------\n")
    for i, data in enumerate(loader, 0):
        outputs = model(input_ids=data['input_ids'].to(device, dtype=torch.int64),
                            attention_mask=data['attention_mask'].to(
                                device, dtype=torch.int64),
                            labels=data['input_ids'].to(device, dtype=torch.int64))

        loss = outputs.loss
        if i % 10 == 0:
            print(f"Epoch={epoch}\tbatch={i} \tloss={loss:.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval(epoch, model, device, loader):
    model.eval()
    print("----------start training----------\n")
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            outputs = model(input_ids=data['input_ids'].to(device, dtype=torch.int64), 
                            attention_mask=data['attention_mask'].to(device, dtype=torch.int64), 
                           labels=data['input_ids'].to(device, dtype=torch.int64))

            loss = outputs.loss
            if i % 10 == 0:
                print(f"Epoch={epoch}\tbatch={i} \tloss={loss:.5f}")


if __name__ == "__main__":
    set_seed(123)
    device = torch.device(0)
    torch.cuda.empty_cache()

    model = model.to(device)
    train_loader, test_loader = get_nsp_data_loader(tokenizer, max_input_length)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-5
    )

    """"record train&evaluation time"""
    for epoch in range(1, 2):
        train(epoch, model, device, train_loader, optimizer)

    eval(1, model, device, test_loader)
