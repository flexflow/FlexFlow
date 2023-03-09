import os

import sys
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

config = GPT2Config.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2', config=config)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
max_input_length = tokenizer.max_model_input_sizes['distilgpt2']
save_dir = "./examples/cpp/GPT"

if __name__ == "__main__":
    print('---from ckp---\n');
    
    #store config in a file
    config.save_pretrained(save_dir, False)
    
    #save weights to file
    for name, param in model.named_parameters():
        if name.startswith('transformer.h.'):
            name = 'layer' + name[len('transformer.h.'):]
        param_path = os.path.join(save_dir, "params",name)
        param_np = param.detach().cpu().numpy()
        param_np.tofile(param_path)
        
