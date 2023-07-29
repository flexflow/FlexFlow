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

from flexflow.serve.models import FlexFlowLLAMA, FlexFlowOPT, FlexFlowFalcon
from flexflow.core import *
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import HfApi
import sys, torch


class SamplingConfig:
    def __init__(self, do_sample=False, temperature=0.9, topp=0.8, topk=1):
        self.do_sample = False
        self.temperature = 0.8
        self.topp = 0.6
        self.topk = 1


class LLM:
    def __init__(
        self,
        model_name,
        data_type=DataType.DT_HALF,
        tokenizer_path="",
        weights_path="",
        output_file="",
    ):
        self.supported_models = {
            "LlamaForCausalLM": (ModelType.LLAMA, FlexFlowLLAMA),
            "LLaMAForCausalLM": (ModelType.LLAMA, FlexFlowLLAMA),
            "OPTForCausalLM": (ModelType.OPT, FlexFlowOPT),
            "RWForCausalLM": (ModelType.FALCON, FlexFlowFalcon),
        }
        self.hf_config = AutoConfig.from_pretrained(model_name)
        self.model_type, self.model_class = self.__get_ff_model_type()
        self.data_type = data_type
        assert self.data_type == DataType.DT_HALF or self.data_type == DataType.DT_FLOAT
        self.tokenizer_path = tokenizer_path
        self.weights_path = weights_path
        self.output_file = output_file
        self.ffconfig = FFConfig()

    def __get_ff_model_type(self):
        architectures = getattr(self.hf_config, "architectures", [])
        ff_arch = None
        if next(iter(architectures), None) is not None:
            ff_arch = self.supported_models.get(architectures[0])
        if ff_arch is None:
            print(
                "Huggingface model of type {architectures} is not yet supported by FlexFlow"
            )
            sys.exit(1)
        return ff_arch

    def download_hf_weights(self):
        # Use local cache, or download new version
        self.weights_path = os.path.expanduser(
            f"~/.cache/flexflow/models/{self.hf_config._name_or_path}"
        )
        os.makedirs(self.weights_path, exist_ok=True)
        print(f"Creating directory {self.weights_path}...")

        # Get local revision SHA, check if it matches latest one on huggingface
        local_revision = None
        local_revision_file = os.path.join(self.weights_path, "rev_sha.txt")
        if os.path.exists(local_revision_file):
            local_revision = "".join(open(local_revision_file).read().split())
        hf_api = HfApi()
        latest_revision = hf_api.model_info(self.hf_config._name_or_path).sha
        print(local_revision, latest_revision)
        # Download if needed
        if local_revision != latest_revision:
            print(
                f"'{self.hf_config._name_or_path}' model weights not found in cache or outdated. Downloading from huggingface.co ..."
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.hf_config._name_or_path
            )
            print("Done downloading HF weights. Converting them now...")
            self.model_class.convert_hf_model(hf_model, self.weights_path)
            with open(local_revision_file, "w+") as f:
                f.write(latest_revision)
            print("Done converting the weights...")

    def load_hf_weights(self):
        print("Loading hf weights...")

        if self.data_type == DataType.DT_HALF:
            torch.set_default_tensor_type(torch.HalfTensor)

        if len(self.weights_path) > 0:
            print(f"Using weights from {self.weights_path.length}")
            # check that weights exist
            if not os.path.exists(self.weights_path) or not os.path.isdir(
                self.weights_path
            ):
                raise FileNotFoundError(
                    f"Path {self.weights_path} does not exist or is not a directory"
                )
            elif len(os.listdir(self.weights_path)) == 0:
                raise FileNotFoundError(f"Folder {self.weights_path} is empty")
        else:
            self.download_hf_weights()

        # TODO: load the weights

    def compile(
        self,
        mode=InferenceMode.INC_DECODING_MODE,
        sampling_config=SamplingConfig(),
        max_batch_size=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
        tensor_parallel_degree=4,
        pipeline_parallel_degree=2,
        ssms=[],
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.max_tokens_per_batch = max_tokens_per_batch
        self.tensor_parallel_degree = tensor_parallel_degree
        self.pipeline_parallel_degree = pipeline_parallel_degree
        self.ssms = ssms
        self.sampling_config = SamplingConfig()
        assert (
            mode == InferenceMode.INC_DECODING_MODE
            or mode == InferenceMode.BEAM_SEARCH_MODE
        ) == (len(ssms) == 0)

        # Create model
        self.model = self.model_class(
            mode,
            sampling_config,
            self.ffconfig,
            self.hf_config,
            self.data_type,
            max_batch_size,
            max_seq_length,
            max_tokens_per_batch,
        )

        # Download the weights from huggingface, if needed
        self.load_hf_weights()

        # Create request manager
        self.rm = RequestManager()
        self.rm.register_tokenizer(self.model_type, self.tokenizer_path)
        self.rm.register_output_filepath(self.output_file)

        # Create inference manager
        self.im = InferenceManager()
        # self.im.compile_model_and_allocate_buffer(self.model.ffmodel)
        # self.im.init_operators_inference(self.model.ffmodel)

        assert False and "Not implemented yet"

    def generate(self, prompt, sampling=None):
        self.sampling = sampling if sampling is not None else self.default_config
        assert False and "Not implemented yet"
