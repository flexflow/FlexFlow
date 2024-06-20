import os, sys

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn

# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
import argparse
from peft import LoraConfig, get_peft_model
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="down_proj",
        help="Comma-separated list of layers from the base model to target",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--use-full-precision", action="store_true", help="Use full precision"
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--publish-peft-with-id", type=str, default="")
    args = parser.parse_args()
    model_name = args.model_name
    use_full_precision = args.use_full_precision
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_target_modules = args.lora_target_modules.split(",")
    lora_dropout = args.lora_dropout
    output_dir = args.output_dir
    publish_peft_with_id = args.publish_peft_with_id
    if len(output_dir) == 0 and len(publish_peft_with_id) == 0:
        raise ValueError(
            "Please pass either a --output-dir or a --publish-peft-with-id to specify where to store the trained model"
        )

    # Change working dir to folder storing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        torch_dtype=torch.float32 if use_full_precision else torch.float16,
        device_map="auto",
    )

    # Get Tokenizer
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hf_arch = getattr(hf_config, "architectures")[0]
    if hf_arch == "LLaMAForCausalLM" or hf_arch == "LlamaForCausalLM":
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            torch_dtype=torch.float32 if use_full_precision else torch.float16,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if use_full_precision else torch.float16,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    model.lm_head = CastOutputToFloat(model.lm_head)

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        # target_modules=["q_proj", "v_proj"],
        # target_modules=["down_proj"],
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(model)
    print(model.named_parameters())
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    data = load_dataset("Abirate/english_quotes")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=200,
            learning_rate=2e-4,
            fp16=True if not use_full_precision else False,
            logging_steps=1,
            output_dir=os.path.join(
                output_dir if len(output_dir) > 0 else "./", "lora_training_logs"
            ),
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = (
        False
    )  # silence the warnings. Please re-enable for inference!
    trainer.train()

    if len(output_dir) > 0:
        print(f"Done training! Saving the model to {output_dir}...")
        model.save_pretrained(output_dir)

    if len(publish_peft_with_id) > 0:
        print(
            f"Done training! Uploading the model to HF hub with id: {publish_peft_with_id}..."
        )
        model.push_to_hub(publish_peft_with_id, use_auth_token=True)


if __name__ == "__main__":
    main()
