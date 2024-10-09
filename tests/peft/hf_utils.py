import torch
import torch.nn as nn
import transformers
from transformers import (
    TrainerCallback,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)
import os, shutil
from peft import PeftConfig, PeftModel
from datasets import load_dataset, DatasetDict

debug_dir = None
debug_subdirs = ["fwd", "bwd", "optim", "weights"]
verbose = False


def make_debug_dirs():
    global debug_dir
    global debug_subdirs
    debug_dir = os.environ.get("FF_CACHE_PATH", os.path.expanduser("~/.cache/flexflow"))
    debug_dir = os.path.join(debug_dir, "debug", "huggingface")
    shutil.rmtree(debug_dir, ignore_errors=True)
    os.makedirs(debug_dir, exist_ok=True)
    assert debug_dir is not None
    assert os.path.isdir(debug_dir)
    for subdir in debug_subdirs:
        subdir_path = os.path.join(debug_dir, subdir)
        os.makedirs(subdir_path, exist_ok=False)


def get_dst_folder(subdir, step_idx=0):
    global debug_dir, debug_subdirs
    assert subdir in debug_subdirs
    dst_folder = os.path.join(debug_dir, subdir, f"step_{step_idx}")
    os.makedirs(dst_folder, exist_ok=True)
    return dst_folder


def simplify_name(name):
    return name.replace("base_model.model.model.", "").replace("base_model.model.", "").replace("model.layers.", "layers.").replace("model.", "").replace("decoder.", "")


def get_optim_type(args):
    if args.optimizer == "sgd":
        return transformers.training_args.OptimizerNames.SGD
    elif args.optimizer == "adam":
        return transformers.training_args.OptimizerNames.ADAM
    elif args.optimizer == "adamw":
        return transformers.training_args.OptimizerNames.ADAMW
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")


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
    assert type(grad_input) == tuple and type(grad_output) == tuple
    if len(grad_input) == 0 or len(grad_output) == 0:
        return
    assert module.name is not None and module.bwd_step is not None
    name = simplify_name(module.name)
    if verbose:
        print(
            f"Backward Hook activated for module: {name}, bwd step: {module.bwd_step}"
        )
        print("Backward GRAD Output:")
    for i, out_grad in enumerate(grad_output):
        if type(out_grad) == torch.Tensor:
            dst_folder = get_dst_folder("bwd", module.bwd_step)
            dst_filepath = os.path.join(dst_folder, f"{name}.output_gradient_{i}")
            if verbose:
                print("\t", out_grad.shape)
                print(f"\t\tSaving to {dst_filepath}")
            torch.save(out_grad, dst_filepath)
        else:
            if verbose:
                print(out_grad)
    if verbose:
        print("Backward GRAD Input:")
    for i, in_grad in enumerate(grad_input):
        if type(in_grad) == torch.Tensor:
            dst_folder = get_dst_folder("bwd", module.bwd_step)
            dst_filepath = os.path.join(dst_folder, f"{name}.input_gradient_{i}")
            if verbose:
                print("\t", in_grad.shape)
                print(f"\t\tSaving to {dst_filepath}")
            torch.save(in_grad, dst_filepath)
        else:
            if verbose:
                print(in_grad)
    if verbose:
        print("===")
    module.bwd_step += 1


def fwd_hook(module, input, output):
    if len(input) == 0 or len(output) == 0:
        return
    assert module.name is not None and module.fwd_step is not None
    name = simplify_name(module.name)
    if verbose:
        print(f"Forward Hook activated for module: {name}, fwd step: {module.fwd_step}")
        print("Input:")
    if type(input) == torch.Tensor:
        if verbose:
            print(input.shape)
        dst_folder = get_dst_folder("fwd", module.fwd_step)
        dst_filepath = os.path.join(dst_folder, f"{name}.input_0")
        torch.save(input, dst_filepath)
    elif type(input) == tuple:
        for i, inp in enumerate(input):
            if type(inp) == torch.Tensor:
                if verbose:
                    print(inp.shape)
                dst_folder = get_dst_folder("fwd", module.fwd_step)
                dst_filepath = os.path.join(dst_folder, f"{name}.input_{i}")
                torch.save(inp, dst_filepath)
            else:
                if verbose:
                    print(inp)
    else:
        assert False
    if verbose:
        print("Output:")
    if type(output) == torch.Tensor:
        if verbose:
            print(output.shape)
        dst_folder = get_dst_folder("fwd", module.fwd_step)
        dst_filepath = os.path.join(dst_folder, f"{name}.output_0")
        torch.save(output, dst_filepath)
    elif type(output) == tuple:
        for i, out in enumerate(output):
            if type(out) == torch.Tensor:
                if verbose:
                    print(out.shape)
                dst_folder = get_dst_folder("fwd", module.fwd_step)
                dst_filepath = os.path.join(dst_folder, f"{name}.output_{i}")
                torch.save(out, dst_filepath)
            else:
                if verbose:
                    print(out)
    else:
        assert False
    if verbose:
        print("===")
    module.fwd_step += 1


def peft_optimizer_hook(model_, callback_func_handle):
    def post_hook(optimizer, args, kwargs):
        if verbose:
            print("Optimizer Hook activated")
        bwd_step = callback_func_handle.step_count
        for name_, module in model_.named_modules():
            name = simplify_name(name_)
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    if verbose:
                        print(
                            f"Step #{bwd_step}: Saving weight gradient for {name} ({param.grad.shape})"
                        )
                    dst_folder = get_dst_folder("weights", bwd_step)
                    dst_filepath = os.path.join(dst_folder, f"{name}.gradient")
                    torch.save(param.grad, dst_filepath)

    return post_hook


class HFTrainingCallBack(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if verbose:
            print("Starting finetuning")
        model_ = kwargs.get("model", None)
        optim = kwargs.get("optimizer", None)
        assert model_ is not None
        assert optim is not None
        self.step_count = 0
        optim.optimizer.register_step_post_hook(peft_optimizer_hook(model_, self))

    def save_lora_weights(self, model, pre_finetuning=False):
        lora_weights_handles = [
            (simplify_name(name), params)
            for name, params in model.named_parameters()
            if "lora" in name
        ]
        for simplified_name, params in lora_weights_handles:
            dst_folder = get_dst_folder("weights", self.step_count)
            if pre_finetuning:
                dst_filepath = os.path.join(dst_folder, f"{simplified_name}_original")
                torch.save(params, dst_filepath)
                if verbose:
                    print(
                        f"Step #{self.step_count}: Saving ORIGINAL weight {simplified_name} ({params.shape})"
                    )
            else:
                dst_filepath = os.path.join(dst_folder, f"{simplified_name}_finetuned")
                torch.save(params, dst_filepath)
                if verbose:
                    print(
                        f"Step #{self.step_count}: Saving FINETUNED weight {simplified_name} ({params.shape})"
                    )
        if not pre_finetuning:
            self.step_count += 1

    def on_step_end(
        self, args, state, control, model, tokenizer, optimizer, lr_scheduler, **kwargs
    ):
        self.save_lora_weights(model, pre_finetuning=False)

    def on_step_begin(
        self, args, state, control, model, tokenizer, optimizer, lr_scheduler, **kwargs
    ):
        self.save_lora_weights(model, pre_finetuning=True)

    def on_train_end(self, args, state, control, **kwargs):
        if verbose:
            print(f"Finetuning ended after {self.step_count} steps")


def build_peft_config(args, finetuning=False):
    peft_config = PeftConfig.from_pretrained(args.peft_model_id)
    if peft_config.peft_type != "LORA":
        raise ValueError(f"PEFT type {peft_config.peft_type} not supported yet")
    if args.lora_alpha > 0.0:
        peft_config.lora_alpha = args.lora_alpha
    if peft_config.lora_dropout >= 0.0:
        peft_config.lora_dropout = args.lora_dropout
    # prevent HF from re-inizialing the weights randomly if finetuning
    if finetuning:
        peft_config.init_lora_weights = False
    return peft_config


def prepare_model_for_lora_finetuning(model, save_peft_tensors=False):
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
    return model


def build_peft_model(args, peft_config):
    # Load base model, and apply the PEFT layer
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float32 if args.use_full_precision else torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.peft_model_id, config=peft_config)
    model = prepare_model_for_lora_finetuning(model, args.save_peft_tensors)
    return model


def get_peft_tokenizer(args, peft_config):
    # Get Tokenizer
    hf_config = AutoConfig.from_pretrained(
        peft_config.base_model_name_or_path, trust_remote_code=True
    )
    hf_arch = getattr(hf_config, "architectures")[0]
    if hf_arch == "LLaMAForCausalLM" or hf_arch == "LlamaForCausalLM":
        tokenizer = LlamaTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            use_fast=True,
            torch_dtype=torch.float32 if args.use_full_precision else torch.float16,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float32 if args.use_full_precision else torch.float16,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
    return tokenizer


def register_peft_hooks(model):
    # Save hidden states and gradients
    for name, layer in dict(model.named_modules()).items():
        layer.name = name
        layer.fwd_step = 0
        layer.bwd_step = 0
        if verbose:
            print(f"Adding hooks to layer {layer.name}")
        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(peft_backward_hook)

def register_inference_hooks(model):
    for name, layer in dict(model.named_modules()).items():
        layer.name = name
        layer.fwd_step = 0
        if verbose:
            print(f"Adding hooks to layer {layer.name}")
        layer.register_forward_hook(fwd_hook)

def save_model_weights(model, target_modules=[]):
    # Save any weights of interest
    for name, params in model.named_parameters():
        simplified_name = simplify_name(name)
        for target_module in target_modules:
            if target_module in name:
                dst_folder = get_dst_folder("weights")
                dst_filepath = os.path.join(dst_folder, f"{simplified_name}")
                torch.save(params, dst_filepath)


def filter_dataset_for_debugging(data, key_to_filter, desired_value):
    filtered_dataset_dict = DatasetDict()
    for split, dataset in data.items():
        filtered_dataset = dataset.filter(
            lambda example: example[key_to_filter] == desired_value
        )
        filtered_dataset_dict[split] = filtered_dataset
    data = filtered_dataset_dict
    return data


def save_finetuned_model(model, args):
    if len(args.output_dir) > 0:
        if verbose:
            print(f"Saving the model to {args.output_dir}...")
        model.save_pretrained(args.output_dir)

    if len(args.publish_peft_with_id) > 0:
        if verbose:
            print(
                f"Uploading the model to HF hub with id: {args.publish_peft_with_id}..."
            )
        model.push_to_hub(args.publish_peft_with_id, use_auth_token=True)
