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

from flexflow.serve import LLM, SamplingConfig
from flexflow.core import *

def get_prompts(json_filepath):
   json_obj = None
   return json_obj

def top_level_task():
    # Incremental decoding
    llama = LLM("decapoda-research/llama-30b-hf", data_type = "half")
    sampling_config = SamplingConfig(do_sample=False, temperature = 0.9, topp = 0.8, topk = 1)
    llama.compile(InferenceMode.INC_DECODING_MODE, sampling_config, use_full_precision=False, max_batch_size = 1, max_seq_length = 256, max_tokens_per_batch=64, tensor_parallel_degree = 4, pipeline_parallel_degree = 2)
    
    prompts = llama.generate(prompts, sampling=sampling_config)
    # result = llama.generate("What's the best xxx in yyy?", sampling = sampling_config)
    # print(result)

    # # Speculative inference
    # llama = LLM("decapoda-research/llama-30b-hf", data_type = "half")
    # ssm1 = LLM("Jackfram/llama-160m", data_type = "half")
    # ssm2 = LLM("facebook/opt-125m", data_type = "half")
    # sampling_config = SamplingConfig(temperature = 0.9, topp = 0.8, topk = 1)
    # llama.serve(max_batch_size = 1, max_seq_length = 256, max_tokens_per_batch=64, tensor_parallel_degree = 4, pipeline_parallel_degree = 2, ssms = {ssm1, ssm2})
    # result = llama.generate("What's the best xxx in yyy?", sampling = sampling_config)
    # print(result)

if __name__ == "__main__":
  print("flexflow inference")
  top_level_task()
