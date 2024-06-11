import os, sys, shutil
import torch

# Reproducibility
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)
import torch.nn as nn

# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
import argparse
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import transformers

if transformers.__version__ < "4.31.0":
    raise RuntimeError(
        "Please update the transformers library version to 4.31.0 or above"
    )
from datasets import load_dataset, DatasetDict


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


def peft_backward_hook(module, grad_input, grad_output):
    assert(type(grad_input) == tuple and type(grad_output) == tuple)
    if len(grad_input) == 0 or len(grad_output) == 0:
        return
    assert module.name is not None and module.bwd_step is not None
    name = module.name.replace("base_model.model.model.", "")
    print(f"Backward Hook activated for module: {name}, bwd step: {module.bwd_step}")
    print("Backward GRAD Output:")
    for i, go in enumerate(grad_output):
        if type(go) == torch.Tensor:
            dst_filepath = f"./hf_peft_tensors/bwd_step_{module.bwd_step}_{name}.go_{i}"
            print("\t", go.shape)
            print(f"\t\tSaving to {dst_filepath}")
            torch.save(go, dst_filepath)
        else:
            print(go)
    print("Backward GRAD Input:")
    for i, gi in enumerate(grad_input):
        if type(gi) == torch.Tensor:
            dst_filepath = f"./hf_peft_tensors/bwd_step_{module.bwd_step}_{name}.gi_{i}"
            print("\t", gi.shape)
            print(f"\t\tSaving to {dst_filepath}")
            torch.save(gi, dst_filepath)
        else:
            print(gi)

    print("===")
    module.bwd_step += 1


def peft_forward_hook(module, input, output):
    if len(input) == 0 or len(output) == 0:
        return
    assert module.name is not None and module.fwd_step is not None
    name = module.name.replace("base_model.model.model.", "")
    print(f"Forward Hook activated for module: {name}, fwd step: {module.fwd_step}")
    print("Input:")
    if type(input) == torch.Tensor:
        print(input.shape)
        torch.save(
            input, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.input_0"
        )
    elif type(input) == tuple:
        for i, inp in enumerate(input):
            if type(inp) == torch.Tensor:
                print(inp.shape)
                torch.save(
                    inp, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.input_{i}"
                )
            else:
                print(inp)
    else:
        assert False
    print("Output:")
    if type(output) == torch.Tensor:
        print(output.shape)
        torch.save(
            output, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.output_0"
        )
        # if "layer_norm" in name:
        #     torch.save(
        #         output.grad_fn._saved_result1, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.saved_result_1"
        #     )
        #     torch.save(
        #         output.grad_fn._saved_result2, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.saved_result_2"
        #     )
    elif type(output) == tuple:
        for i, out in enumerate(output):
            if type(out) == torch.Tensor:
                print(out.shape)
                torch.save(
                    out, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.output_{i}"
                )
                # if "layer_norm" in name:
                #     torch.save(
                #         out.grad_fn._saved_result1, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.saved_result_1"
                #     )
                #     torch.save(
                #         out.grad_fn._saved_result2, f"./hf_peft_tensors/fwd_step_{module.fwd_step}_{name}.saved_result_2"
                #     )
            else:
                print(out)
    else:
        assert False
    # print("Forward Input/Output: ", input[0].shape, output[0].shape)
    print("===")
    module.fwd_step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--peft-model-id", type=str, default="goliaro/llama-160m-lora"
    )
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
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
    peft_model_id = args.peft_model_id
    use_full_precision = args.use_full_precision
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    output_dir = args.output_dir
    publish_peft_with_id = args.publish_peft_with_id
    save_peft_tensors = args.save_peft_tensors

    # Change working dir to folder storing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Get PEFT layer, edit any configs as needed
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    if peft_config.peft_type != "LORA":
        raise ValueError(f"PEFT type {peft_config.peft_type} not supported yet")
    peft_config.lora_alpha = lora_alpha
    peft_config.lora_dropout = lora_dropout
    peft_config.init_lora_weights = (
        False
    )  # prevent HF from re-inizialing the weights randomly
    model_name = peft_config.base_model_name_or_path
    # Load base model, and apply the PEFT layer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if use_full_precision else torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, peft_model_id, config=peft_config)

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

    # Freeze all layers except the LORA ones. Cast small layers to full precision for stability
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False  # freeze the model - train adapters later
        else:
            param.requires_grad = True
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    if not save_peft_tensors:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Print model with PEFT
    print(model)
    for name, params in model.named_parameters():
        print(name)
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
        # Save any weights of interest
        for name, params in model.named_parameters():
            simplified_name = name.replace("base_model.model.model.", "")
            if "lora" in name:
                torch.save(params, f"./hf_peft_tensors/{simplified_name}")
            if "lm_head" in name or "norm" in name:
                torch.save(params, f"./hf_peft_tensors/{simplified_name}")
            if "down_proj" in name or "self_attn" in name:
                torch.save(params, f"./hf_peft_tensors/{simplified_name}")

    # Load fine-tuning dataset
    data = load_dataset("Abirate/english_quotes")

    # TODO: remove using of a single row
    key_to_filter = "quote"
    desired_value = "“Two things are infinite: the universe and human stupidity; and I'm not sure about the universe.”"
    filtered_dataset_dict = DatasetDict()
    for split, dataset in data.items():
        filtered_dataset = dataset.filter(
            lambda example: example[key_to_filter] == desired_value
        )
        filtered_dataset_dict[split] = filtered_dataset
    data = filtered_dataset_dict
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            max_steps=1,
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

    # for batch in trainer.get_train_dataloader():
    #     print("First batch: ")
    #     print(batch)
    #     break

    trainer.train()

    if len(output_dir) > 0:
        print(f"Saving the model to {output_dir}...")
        model.save_pretrained(output_dir)

    if len(publish_peft_with_id) > 0:
        print(f"Uploading the model to HF hub with id: {publish_peft_with_id}...")
        model.push_to_hub(publish_peft_with_id, use_auth_token=True)


if __name__ == "__main__":
    main()
