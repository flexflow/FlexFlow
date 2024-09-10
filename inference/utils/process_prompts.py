import json
import argparse

def read_prompts_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

def write_prompts_to_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_prompts(input_file, output_file):
    prompts = read_prompts_from_json(input_file)
    processed_prompts = [{"prompt": prompt, "slo_ratio": 1.0} for prompt in prompts]
    write_prompts_to_json(output_file, processed_prompts)

def main():
    parser = argparse.ArgumentParser(description="Process prompts JSON file and generate slo_ratio for each prompt.")
    parser.add_argument('input_file', type=str, help="Input JSON file containing prompts.")
    parser.add_argument('output_file', type=str, help="Output JSON file to save the processed prompts.")
    
    args = parser.parse_args()

    process_prompts(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
