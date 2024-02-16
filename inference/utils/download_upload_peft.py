#!/usr/bin/env python
import argparse
from huggingface_hub import HfApi, HfFolder
import flexflow.serve as ff

def parse_args():
    parser = argparse.ArgumentParser(description="Download a PEFT model with FlexFlow, process it, and upload it to the Hugging Face Hub.")
    parser.add_argument("peft_model_id", type=str, help="Original Hugging Face PEFT model ID to download and process (e.g., 'username/peft-model').")
    parser.add_argument("--new-model-id", type=str, required=True, help="New Hugging Face Hub model ID for upload (e.g., 'your_username/new-peft-model-name').")
    parser.add_argument("--cache-folder", type=str, default="./peft_model_cache", help="Folder to use to store and process the PEFT model(s) assets in FlexFlow format.")
    parser.add_argument("--private", action="store_true", help="Whether to upload the processed PEFT model as a private model on Hugging Face Hub.")
    parser.add_argument("--refresh-cache", action="store_true", help="Use this flag to force the refresh of the PEFT model(s) weights/cache.")
    parser.add_argument("--full-precision", action="store_true", help="Download the full precision version of the weights for the PEFT model.")
    return parser.parse_args()

def download_and_process_peft_model(peft_model_id, cache_folder, refresh_cache, full_precision):
    data_type = ff.DataType.DT_FLOAT if full_precision else ff.DataType.DT_HALF
    print(f"Downloading and processing PEFT model: {peft_model_id}")
    peft = ff.PEFT(
        peft_model_id=peft_model_id,
        data_type=data_type,
        cache_path=cache_folder,
        refresh_cache=refresh_cache,
    )
    peft.download_hf_weights_if_needed()
    peft.download_hf_config()
    # any necessary conversion or processing by FlexFlow happens here
    return peft
    
    
def upload_peft_model_to_hub(peft, new_model_id, cache_folder, private):
    print(f"Uploading peft model to HuggingFace Hub: {new_model_id}")
    peft.upload_hf_model(new_model_id, cache_folder, private=private)
    print("Upload completed successfully.")
    

def main():
    args = parse_args()
    peft = download_and_process_peft_model(args.peft_model_id, args.cache_folder, args.refresh_cache, args.full_precision)
    upload_peft_model_to_hub(peft, args.new_model_id, args.cache_folder, args.private)

if __name__ == "__main__":
    main()
