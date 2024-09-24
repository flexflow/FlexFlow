import os, sys, shutil
import torch

# Reproducibility
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)

# import bitsandbytes as bnb
import argparse
import transformers

if transformers.__version__ < "4.31.0":
    raise RuntimeError(
        "Please update the transformers library version to 4.31.0 or above"
    )
from datasets import load_dataset


from hf_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft-model-id", type=str, default="goliaro/llama-160m-lora")
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=-1,
        help="The scaling coefficient for LoRA. Leave it set to -1 to use the original value from the HF config",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="The dropout rate for LoRA. Set it to -1 to use the original value from the HF config",
    )
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001)
    parser.add_argument("-n", "--max-steps", type=int, default=2)
    parser.add_argument(
        "--optimizer", type=str, choices=["sgs", "adam", "adamw"], default="sgd"
    )
    parser.add_argument(
        "--use-full-precision", action="store_true", help="Use full precision"
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--publish-peft-with-id", type=str, default="")
    parser.add_argument(
        "--save-peft-tensors",
        action="store_true",
        help="Save PEFT hidden states and weights to file",
    )
    args = parser.parse_args()

    # Change working dir to folder storing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Get PEFT config, model, tokenizer, and optimizer type
    peft_config = build_peft_config(args, finetuning=True)
    tokenizer = get_peft_tokenizer(args, peft_config)
    model = build_peft_model(args, peft_config)
    optim_type = get_optim_type(args)

    # Print model with PEFT
    print(model)
    for name, params in model.named_parameters():
        print(name)
    print_trainable_parameters(model)

    # Add hooks to save PEFT tensors, save any weights of interest before finetuning
    if args.save_peft_tensors:
        make_debug_dirs()
        register_peft_hooks(model)
        save_peft_weights(model, target_modules=["lora", "lm_head", "down_proj"])

    # Load fine-tuning dataset
    data = load_dataset("Abirate/english_quotes")
    # TODO: remove using of a single row
    key_to_filter = "quote"
    desired_value = "“Two things are infinite: the universe and human stupidity; and I'm not sure about the universe.”"
    data = filter_dataset_for_debugging(data, key_to_filter, desired_value)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    # Training loop
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_grad_norm=None,  # Disable gradient clipping
            warmup_steps=0,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=True if not args.use_full_precision else False,
            logging_steps=1,
            output_dir=os.path.join(
                args.output_dir if len(args.output_dir) > 0 else "./",
                "lora_training_logs",
            ),
            optim=optim_type,
            lr_scheduler_type=transformers.training_args.SchedulerType.CONSTANT,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
        callbacks=[HFTrainingCallBack] if args.save_peft_tensors else None,
    )
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False

    # for batch in trainer.get_train_dataloader():
    #     print("First batch: ")
    #     print(batch)
    #     break

    trainer.train()

    save_finetuned_model(model, args)


if __name__ == "__main__":
    main()
