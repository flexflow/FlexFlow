
# this script is for testing downloading a model from huggingface and uploading it back to huggingface
# after the model is downloaded it will be transformed into flexflow format
# before uploading it back to huggingface, we need to convert it back to huggingface format
# which is done by calling llm.upload_hf_model()

#!/usr/bin/env python
import argparse
from huggingface_hub import HfApi, HfFolder
import flexflow.serve as ff
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Download a model with FlexFlow, process it, and upload it to the Hugging Face Hub.")
    parser.add_argument("model_name", type=str, help="Original Hugging Face model ID to download and process (e.g., 'facebook/opt-125m').")
    parser.add_argument("--new-model-id", type=str, required=True, help="New Hugging Face Hub model ID for upload (e.g., 'your_username/new-model-name').")
    parser.add_argument("--cache-folder", type=str, default="~/.cache/flexflow", help="Folder to use to store and process the model(s) assets in FlexFlow format.")
    parser.add_argument("--private", action="store_true", help="Whether to upload the processed model as a private model on Hugging Face Hub.")
    parser.add_argument("--refresh-cache", action="store_true", help="Use this flag to force the refresh of the model(s) weights/tokenizer cache.")
    parser.add_argument("--full-precision", action="store_true", help="Download the full precision version of the weights.")
    return parser.parse_args()


def main():
    model_name = "tiiuae/falcon-7b"
    new_model_id = "aprilyyt/falcon-upload-test-new"
    cache_folder = "~/.cache/flexflow"
    private = True
    refresh_cache = False
    full_precision = True

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

    print(f"Uploading processed model to Hugging Face Hub: {new_model_id}")
    llm.upload_hf_model(new_model_id, private=private)
    print("Upload completed successfully.")

if __name__ == "__main__":
    main()