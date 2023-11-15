import os, sys, shutil
import torch
# Reproducibility
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
#torch.use_deterministic_algorithms(True)
import torch.nn as nn
#import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
import argparse
from peft import LoraConfig, get_peft_model, PeftModel
import transformers
from datasets import load_dataset

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float32)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def convert_hf_weight_name(name):
    return (
        name.replace(".", "_")
        .replace("self_attn", "attention")
        .replace("q_proj", "wq")
        .replace("k_proj", "wk")
        .replace("v_proj", "wv")
        .replace("o_proj", "wo")
        .replace("mlp", "feed_forward")
        .replace("gate_proj", "w1")
        .replace("down_proj", "w2")
        .replace("up_proj", "w3")
        .replace("input_layernorm", "attention_norm")
        .replace("post_attention_layernorm", "ffn_norm")
        .replace("embed_tokens", "tok_embeddings")
        .replace("lm_head", "output")
        .replace("model_", "")
        .replace("base_", "")
        .replace("default_", "")
    )

def peft_backward_hook(module, grad_input, grad_output):
    if len(grad_input) == 0 or len(grad_output) == 0:
        return
    assert(module.name is not None and module.bwd_step is not None)
    name = module.name.replace("base_model.model.model.", "")
    print(f"Backward Hook activated for module: {name}, bwd step: {module.bwd_step}")
    print("Backward GRAD Input:")
    for i,gi in enumerate(grad_input):
        if type(gi) == torch.Tensor:
            print(gi.shape)
            torch.save(gi, f"./hf_peft_tensors/bwd_step_{module.bwd_step}_{name}.gi_{i}")
        else:
            print(gi)
    print("Backward GRAD Output:")
    for i, go in enumerate(grad_output):
        if type(go) == torch.Tensor:
            print(go.shape)
            torch.save(go, f"./hf_peft_tensors/bwd_step_{module.bwd_step}_{name}.go_{i}")
        else:
            print(go)
    
    print("===")
    module.bwd_step += 1

def peft_forward_hook(module, input, output):
    if len(input) == 0 or len(output) == 0:
        return
    assert(module.name is not None and module.fwd_step is not None)
    name = module.name.replace("base_model.model.model.", "")
    print(f"Forward Hook activated for module: {name}, fwd step: {module.fwd_step}")
    print("Input:")
    for i,inp in enumerate(input):
        if type(inp) == torch.Tensor:
            print(inp.shape)
            torch.save(inp, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.input_{i}")
        else:
            print(inp)
    print("Output:")
    for i, out in enumerate(output):
        if type(out) == torch.Tensor:
            print(out.shape)
            torch.save(out, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.output_{i}")
        else:
            print(out)
    #print("Forward Input/Output: ", input[0].shape, output[0].shape)
    print("===")
    module.fwd_step += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-target-modules", type=str, default="down_proj", help="Comma-separated list of layers from the base model to target")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-full-precision", action="store_true", help="Use full precision")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--publish-peft-with-id", type=str, default="")
    parser.add_argument("--save-peft-tensors", action="store_true", help="Save PEFT hidden states and weights to file")
    args = parser.parse_args()
    model_name = args.model_name
    use_full_precision=args.use_full_precision
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_target_modules = args.lora_target_modules.split(",")
    lora_dropout = args.lora_dropout
    output_dir = args.output_dir
    publish_peft_with_id = args.publish_peft_with_id
    save_peft_tensors = args.save_peft_tensors
    # if len(output_dir) == 0 and len(publish_peft_with_id) == 0:
    #     raise ValueError("Please pass either a --output-dir or a --publish-peft-with-id to specify where to store the fine-tuned model")

    # Change working dir to folder storing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_8bit=True,
        torch_dtype = torch.float32 if use_full_precision else torch.float16,
        device_map='auto',
    )

    # Get Tokenizer
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hf_arch = getattr(hf_config, "architectures")[0]
    if hf_arch == "LLaMAForCausalLM" or hf_arch == "LlamaForCausalLM":
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=True, torch_dtype = torch.float32 if use_full_precision else torch.float16,)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype = torch.float32 if use_full_precision else torch.float16,)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
    
    peft_model_name = "goliaro/llama-2-7b-lora-full"
    model = PeftModel.from_pretrained(model, peft_model_name)
    
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    #model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        #target_modules=["q_proj", "v_proj"],
        #target_modules=["down_proj"],
        target_modules=lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    
    print(model)
    print(model.named_parameters())
    #model = get_peft_model(model, config)
    print_trainable_parameters(model)

    if save_peft_tensors:
        shutil.rmtree("./hf_peft_tensors", ignore_errors=True)
        # Check that the output folder exists
        os.makedirs("./hf_peft_tensors", exist_ok=True)
        # Save hidden states and gradients
        for name, layer in dict(model.named_modules()).items():
            layer.name = name
            layer.fwd_step = 0
            layer.bwd_step = 0
            print(f"Adding hooks to layer {layer.name}")
            layer.register_forward_hook(peft_forward_hook)
            layer.register_full_backward_hook(peft_backward_hook)
        # Save weights
        for name, params in model.named_parameters():
            if "lora" in name:
                torch.save(params, f"./hf_peft_tensors/{name}")
                # Overwrite FF cached weight
                dst_folder = f"/home/ubuntu/.cache/flexflow/weights/{peft_model_name}/full-precision"
                assert(os.path.exists(dst_folder))
                ff_w_name = convert_hf_weight_name(name)
                print(f"{dst_folder}/{ff_w_name}")
                params.detach().cpu().numpy().tofile(f"{dst_folder}/{ff_w_name}")
            if "lm_head" in name:
                torch.save(params, f"./hf_peft_tensors/{name}")

    data = load_dataset("/home/ubuntu/english_quotes")
    data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            max_steps=1,
            learning_rate=2e-4,
            fp16=True if not use_full_precision else False,
            logging_steps=1,
            output_dir=os.path.join(output_dir if len(output_dir) > 0 else "./", "lora_training_logs"),
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    for batch in trainer.get_train_dataloader():
        print("First batch: ")
        print(batch)
        break
    
    trainer.train()

    # if len(output_dir) > 0:
    #     print(f"Done fine-tuning! Saving the model to {output_dir}...")
    #     model.save_pretrained(output_dir)
    
    # if len(publish_peft_with_id) > 0:
    #     print(f"Done fine-tuning! Uploading the model to HF hub with id: {publish_peft_with_id}...")
    #     model.push_to_hub(publish_peft_with_id, use_auth_token=True)

if __name__ == "__main__":
    main()