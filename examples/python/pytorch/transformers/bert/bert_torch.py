import os

import sys
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
sys.path.append("./examples/python/pytorch/transformers")
from transformer_utils import set_seed, get_transformer_dataloaders, BATCH_SIZE

BASE_DIR = "examples/python/pytorch/bert"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOADER_DIR = os.path.join(BASE_DIR, "loader")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['distilbert-base-uncased']


def train(epoch, model, device, loader, optimizer):
    model.train()
    print("----------start training----------\n")
    for i, data in enumerate(loader, 0):
        labels = data['label'].to(device, dtype=torch.int64)
        outputs = model(input_ids=data['input_ids'].to(device, dtype=torch.int64),
                        attention_mask=data['attention_mask'].to(
                            device, dtype=torch.int64), labels=labels)

        loss = outputs.loss
        if i % 10 == 0:
            print(f"Epoch={epoch}\tbatch={i} \tloss={loss:.8f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval(epoch, model, device, loader):
    print("------------start eval--------------\n")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            outputs = model(input_ids=data['input_ids'].to(device, dtype=torch.int64),
                            attention_mask=data['attention_mask'].to(
                                device, dtype=torch.int64),
                            labels=data['label'].to(device, dtype=torch.int64))
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == data['label'].to(device)).sum()
            total += BATCH_SIZE
    print("-----")
    print(correct)
    print(total)
    print(f"Epoch={epoch}\taccurate={correct/total:.8f}")


if __name__ == "__main__":
    set_seed(123)
    device = torch.device(0)
    torch.cuda.empty_cache()

    model = model.to(device)
    train_loader, test_loader = get_transformer_dataloaders(tokenizer, max_input_length)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-5
    )

    """"record train&evaluation time"""
    for epoch in range(1, 2):
        train(epoch, model, device, train_loader, optimizer)

    eval(1, model, device, test_loader)
