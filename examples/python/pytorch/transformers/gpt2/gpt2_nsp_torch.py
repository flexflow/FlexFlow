import os

import sys
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
sys.path.append("./examples/python/pytorch/transformers")
from transformer_utils import set_seed, get_nsp_data_loader


config = GPT2Config.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2", config=config)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
max_input_length = tokenizer.max_model_input_sizes['distilgpt2']

def train(epoch, model, loader, device, optimizer):
    model.train()
    print("----------start training----------\n")
    for i, data in enumerate(loader, 0):
        outputs = model(input_ids=data['input_ids'].to(device, dtype=torch.int64),
                            attention_mask=data['attention_mask'].to(
                                device, dtype=torch.int64),
                            labels=data['input_ids'].to(device, dtype=torch.int64))

        loss = outputs.loss
        if i % 10 == 0:
            print(f"Epoch={epoch}\tbatch={i} \tloss={loss:.8f}")

        model.zero_grad()
        loss.backward()
        optimizer.step()
        
def eval(epoch, model, device, loader):
    model.eval()
    print("----------start eval----------\n")
    for i, data in enumerate(loader, 0):
        outputs = model(input_ids=data['input_ids'].to(device, dtype=torch.int64),
                            attention_mask=data['attention_mask'].to(
                                device, dtype=torch.int64),
                            labels=data['input_ids'].to(device, dtype=torch.int64))
        
        loss = outputs.loss
        logits = outputs.logits
        #print next token_id
        if i % 10 == 0:
            print(f"Epoch={epoch}\tbatch={i} \tloss={loss:.8f}")

if __name__ == "__main__":
    set_seed(123)
    device = torch.device(0)
    torch.cuda.empty_cache()

    model = model.to(device)
    
    train_dataloader, test_dataloader = get_nsp_data_loader(tokenizer=tokenizer, max_input_length=max_input_length)

    #for the new padding token
    model.resize_token_embeddings(len(tokenizer)) 
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-5
    )
    for epoch in range(1, 2):
        train(epoch, model, train_dataloader, device, optimizer)
    eval(1, model, device, test_dataloader)