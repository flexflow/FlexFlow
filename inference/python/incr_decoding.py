# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flexflow.serve as ff
import argparse, json


def parse_args():
    parser = argparse.ArgumentParser()
    # LLM arguments
    parser.add_argument(
        "-llm-model",
        help="The name of the HuggingFace model to use. E.g. 'decapoda-research/llama-7b-hf'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-llm-weight",
        help="Path to the weights for the LLM. If omitted, FlexFlow will download (and cache) the weights from HuggingFace",
        type=str,
        default="",
    )
    parser.add_argument(
        "-llm-tokenizer",
        help="Path to the tokenizer file or folder for the LLM. If omitted, FlexFlow will download (and cache) the relevant tokenizer from HuggingFace",
        type=str,
        default="",
    )
    parser.add_argument(
        "-clean-model-cache",
        help="Use this flag to discard previous weights/tokenizer cache for this LLM.",
        action="store_true",
    )
    parser.add_argument(
        "-full-precision",
        help="Use this flag to require the use of full precision weights for the LLM.",
        action="store_true",
    )
    # Generation arguments
    parser.add_argument(
        "-prompt",
        help="Path to the prompt file. If omitted, a sample prompt will be used instead.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-output-file",
        help="Path to the output file. If omitted, the output will not be written to file.",
        type=str,
        default="",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize the FlexFlow runtime. ff.init() takes a dictionary or the path to a JSON file with the configs
    ff.init(
        {
            # required arguments
            "num_gpus": 4,
            "memory_per_gpu": 14000,
            "zero_copy_memory_per_gpu": 30000,
            # optional arguments
            "num_cpus": 4,
            "legion_utility_processors": 4,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 2,
            "pipeline_parallelism_degree": 2,
            "offload": False,
            "offload_reserve_space_size": 1024**2,
            "use_4bit_quantization": False,
            "use_8bit_quantization": False,
            "profiling": False,
            "fusion": True,
        }
    )

    # Create the FlexFlow LLM
    ff_data_type = ff.DataType.DT_FLOAT if args.full_precision else ff.DataType.DT_HALF
    llm = ff.LLM(
        args.llm_model,
        data_type=ff_data_type,
        tokenizer_path=args.llm_tokenizer,
        weights_path=args.llm_weight,
        clean_cache=args.clean_model_cache,
        output_file=args.output_file,
    )

    # Compile the LLM for inference and load the weights into memory
    sampling_config = ff.SamplingConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )
    llm.compile(
        ff.InferenceMode.INC_DECODING_MODE,
        sampling_config,
        max_batch_size=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
    )

    # Generation begins!
    if len(args.prompt) > 0:
        prompts = [s for s in json.load(open(args.prompt))]
        results = llm.generate(prompts)
    else:
        result = llm.generate("Here are some travel tips for Tokyo:\n")


if __name__ == "__main__":
    print("flexflow inference (incremental decoding)")
    main()
