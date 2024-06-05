
# this script is for testing downloading a model from huggingface and uploading it back to huggingface
# after the model is downloaded it will be transformed into flexflow format
# before uploading it back to huggingface, we need to convert it back to huggingface format
# which is done by calling llm.upload_hf_model()

#!/usr/bin/env python
import argparse, os
import flexflow.serve as ff
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a model with FlexFlow, process it, and upload it to the Hugging Face Hub."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Original Hugging Face model ID to download and process (e.g., 'facebook/opt-125m')."
    )
    parser.add_argument(
        "--new-model-id", 
        type=str, 
        required=True, 
        help="New Hugging Face Hub model ID for upload (e.g., 'your_username/new-model-name')."
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        help="Folder to use to store the model(s) assets in FlexFlow format",
        default=os.environ.get("FF_CACHE_PATH", ""),
    )
    parser.add_argument("--private", action="store_true", help="Whether to upload the processed model as a private model on Hugging Face Hub.")
    parser.add_argument("--full-precision", action="store_true", help="Download the full precision version of the weights.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_type = ff.DataType.DT_FLOAT if args.full_precision else ff.DataType.DT_HALF
    print(f"Downloading and processing model: {args.model_name}")
    llm = ff.LLM(
        model_name=args.model_name,
        data_type=data_type,
        cache_path=args.cache_folder,
        refresh_cache=False,
    )
    print(f"Uploading processed model to Hugging Face Hub: {args.new_model_id}")
    llm.upload_hf_model(args.new_model_id, private=args.private)
    print("Upload completed successfully.")

if __name__ == "__main__":
    main()