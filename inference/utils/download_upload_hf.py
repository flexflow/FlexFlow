#!/usr/bin/env python
import argparse
from huggingface_hub import HfApi, HfFolder
import flexflow.serve as ff

def parse_args():
    parser = argparse.ArgumentParser(description="Download a model with FlexFlow, process it, and upload it to the Hugging Face Hub.")
    parser.add_argument("model_name", type=str, help="Original Hugging Face model ID to download and process (e.g., 'facebook/opt-125m').")
    parser.add_argument("--new-model-id", type=str, required=True, help="New Hugging Face Hub model ID for upload (e.g., 'your_username/new-model-name').")
    parser.add_argument("--cache-folder", type=str, default="./model_cache", help="Folder to use to store and process the model(s) assets in FlexFlow format.")
    parser.add_argument("--private", action="store_true", help="Whether to upload the processed model as a private model on Hugging Face Hub.")
    parser.add_argument("--refresh-cache", action="store_true", help="Use this flag to force the refresh of the model(s) weights/tokenizer cache.")
    parser.add_argument("--full-precision", action="store_true", help="Download the full precision version of the weights.")
    return parser.parse_args()

def download_and_process_model(model_name, cache_folder, refresh_cache, full_precision):
    data_type = ff.DataType.DT_FLOAT if full_precision else ff.DataType.DT_HALF
    print(f"Downloading and processing model: {model_name}")
    llm = ff.LLM(
        model_name=model_name,
        data_type=data_type,
        cache_path=cache_folder,
        refresh_cache=refresh_cache,
    )
    llm.download_hf_weights_if_needed()
    llm.download_hf_tokenizer_if_needed()
    llm.download_hf_config()
    # any necessary conversion or processing by FlexFlow happens here

def upload_processed_model_to_hub(new_model_id, cache_folder, private):
    print(f"Uploading processed model to Hugging Face Hub: {new_model_id}")
    api = HfApi()
    if not HfFolder.get_token():
        print("Hugging Face token not found. Please login using `huggingface-cli login`.")
        return
    api.create_repo(repo_id=new_model_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=cache_folder, repo_id=new_model_id)
    print("Upload completed successfully.")

def main():
    args = parse_args()
    download_and_process_model(args.model_name, args.cache_folder, args.refresh_cache, args.full_precision)
    upload_processed_model_to_hub(args.new_model_id, args.cache_folder, args.private)

if __name__ == "__main__":
    main()


# python download_upload_hf.py facebook/opt-125m --new-model-id username/modelname --cache-folder ./model_cache --private