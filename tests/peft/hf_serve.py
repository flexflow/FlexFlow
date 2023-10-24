import argparse
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft-model-id", type=str, default="./finetuned-llama")
    parser.add_argument("--use-full-precision", action="store_true", help="Use full precision")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()
    peft_model_id = args.peft_model_id
    #peft_model_id = "goliaro/llama-7b-lora-half"
    use_full_precision=args.use_full_precision
    max_new_tokens = args.max_new_tokens

    # Change working dir to folder storing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, 
        return_dict=True, 
        #load_in_8bit=True, 
        torch_dtype = torch.float32 if use_full_precision else torch.float16,
        device_map='auto',
    )
    hf_config = AutoConfig.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    hf_arch = getattr(hf_config, "architectures")[0]
    if hf_arch == "LLaMAForCausalLM" or hf_arch == "LlamaForCausalLM":
        tokenizer = LlamaTokenizer.from_pretrained(
            config.base_model_name_or_path, use_fast=True, 
            torch_dtype = torch.float32 if use_full_precision else torch.float16,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path, 
            torch_dtype = torch.float32 if use_full_precision else torch.float16,
        )

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)
    batch = tokenizer("Two things are infinite: ", return_tensors='pt')
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=max_new_tokens)
    print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
