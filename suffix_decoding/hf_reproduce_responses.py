import sys, os, json
from dataclasses import asdict, dataclass, field
from typing import Optional
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
)
sys.path.append("/usr/suffix-tree-decoding/trace")
from convert_traces import TraceV2, TracePartition, TraceMetadata, save_trace_v2

@dataclass
class TraceEntryV2:
    prompt: str
    response: str
    prompt_length: int
    response_length: int
    hf_response: Optional[str] = None

def parse_json_to_tracev2(file_path: str) -> TraceV2:
    def parse_trace_metadata(metadata_data: dict) -> TraceMetadata:
        return TraceMetadata(
            avg_entries_per_partition=metadata_data['avg_entries_per_partition'],
            max_prompt_length=metadata_data['max_prompt_length'],
            min_prompt_length=metadata_data['min_prompt_length'],
            avg_prompt_length=metadata_data['avg_prompt_length'],
            max_response_length=metadata_data['max_response_length'],
            min_response_length=metadata_data['min_response_length'],
            avg_response_length=metadata_data['avg_response_length']
        )
    def parse_trace_partition(partition_data: dict) -> TracePartition:
        def parse_trace_entry(entry_data: dict) -> TraceEntryV2:
            return TraceEntryV2(
                prompt=entry_data['prompt'],
                response=entry_data['response'],
                prompt_length=entry_data['prompt_length'],
                response_length=entry_data['response_length']
            )
        return TracePartition(
            partition_name=partition_data['partition_name'],
            model_name=partition_data['model_name'],
            training_entries=[parse_trace_entry(entry) for entry in partition_data['training_entries']],
            eval_entries=[parse_trace_entry(entry) for entry in partition_data['eval_entries']]
        )
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return TraceV2(
        partitions=[parse_trace_partition(partition) for partition in data['partitions']],
        metadata=parse_trace_metadata(data['metadata'])
    )

def reproduce_reponses(trace: TraceV2, partition_name: str) -> TraceV2:
    for partition in trace.partitions:
        if partition.partition_name == partition_name:
            model = AutoModelForCausalLM.from_pretrained(partition.model_name, trust_remote_code=True, device_map="auto",)
            tokenizer = AutoTokenizer.from_pretrained(partition.model_name, trust_remote_code=True)
            generation_config = GenerationConfig.from_pretrained(partition.model_name)
            generation_config.do_sample = False
            generation_config.num_beams=1
            generation_config.temperature = None
            generation_config.top_p = None
            print(generation_config)
            for entry in tqdm(partition.eval_entries):
                batch = tokenizer(entry.prompt, return_tensors="pt", add_special_tokens=False)
                tokenized_length = batch["input_ids"].shape[1]
                assert(tokenized_length == entry.prompt_length)
                print(tokenized_length, entry.response_length)
                generated = model.generate(
                    batch["input_ids"],
                    max_new_tokens=entry.response_length,
                    generation_config=generation_config,
                )
                generated=generated[0][tokenized_length:]
                print(len(generated))
                # print(generated[0].shape)
                out = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print(out)
                # prompt_length = len(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
                # hf_response = out[prompt_length:]
                # print(hf_response)
                # prompt_length = len(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
                # print(prompt_length, entry.prompt_length)
                # all_text = out[prompt_length:]
                # print(all_text)
                # entry.hf_response = all_text
                # entry.hf_response = out
                # print(out)
                # print(entry.hf_response)
                # print(len(out)-tokenized_length, entry.response_length)
                break
    return trace

if __name__ == "__main__":
    # Change working directory to parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)

    trace_filepath = "/usr/suffix-tree-decoding/trace/cortex_v2.json"
    trace_v2 = parse_json_to_tracev2(trace_filepath)
    trace_v2 = reproduce_reponses(trace_v2, "SQL_FANOUT1")
    save_trace_v2(trace_v2, "/usr/suffix-tree-decoding/trace/cortex_v3.json")
