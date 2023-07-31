import argparse
import json

def convert_json(input_file, output_file):
    # Load the input JSON data from the file
    with open(input_file, 'r') as file:
        input_data = json.load(file)

    # Extract the required fields and create the output JSON object
    output_data = {
        "n_layers": input_data["num_hidden_layers"],
        "vocab_size": input_data["vocab_size"],
        "n_heads": input_data["num_attention_heads"],
        "dim": input_data["hidden_size"],
        "multiple_of": 256,
        "norm_eps": input_data["rms_norm_eps"],
        "total_requests": 2560,
        "hidden_dim": input_data["intermediate_size"],
        "incremental_mode": input_data["use_cache"]
    }

    # Save the output JSON data to the file
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON file to a different format.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("output_file", help="Path to the output JSON file.")
    args = parser.parse_args()

    convert_json(args.input_file, args.output_file)
