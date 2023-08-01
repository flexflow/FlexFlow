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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from huggingface_hub import HfApi
import sys, torch, shutil


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
        clean_cache=False,
        output_file="",
    ):
        self.supported_models = {
            "LlamaForCausalLM": (ModelType.LLAMA, FlexFlowLLAMA),
            "LLaMAForCausalLM": (ModelType.LLAMA, FlexFlowLLAMA),
            "OPTForCausalLM": (ModelType.OPT, FlexFlowOPT),
            "RWForCausalLM": (ModelType.FALCON, FlexFlowFalcon),
        }
        self.hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model_type, self.model_class = self.__get_ff_model_type()
        self.data_type = data_type
        assert self.data_type == DataType.DT_HALF or self.data_type == DataType.DT_FLOAT
        self.tokenizer_path = tokenizer_path
        self.weights_path = weights_path
        self.clean_cache = clean_cache
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
            f"~/.cache/flexflow/models/{self.hf_config._name_or_path}/{'full-precision' if self.data_type == DataType.DT_FLOAT else 'half-precision'}"
        )
        if self.clean_cache:
            print(
                f"Discarding cached weights (if they exist) for model {self.hf_config._name_or_path}..."
            )
            if os.path.exists(self.weights_path):
                shutil.rmtree(self.weights_path)
        os.makedirs(self.weights_path, exist_ok=True)
        print(f"Creating directory {self.weights_path} (if it doesn't exist)...")

        # Get local revision SHA, check if it matches latest one on huggingface
        local_revision = None
        local_revision_file = os.path.join(self.weights_path, "rev_sha.txt")
        if os.path.exists(local_revision_file):
            local_revision = "".join(open(local_revision_file).read().split())
        hf_api = HfApi()
        latest_revision = hf_api.model_info(self.hf_config._name_or_path).sha

        # Download if needed
        if local_revision != latest_revision:
            print(
                f"'{self.hf_config._name_or_path}' model weights not found in cache or outdated. Downloading from huggingface.co ..."
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.hf_config._name_or_path, trust_remote_code=True
            )
            print("Done downloading HF weights. Converting them now...")
            self.model_class.convert_hf_model(hf_model, self.weights_path)
            with open(local_revision_file, "w+") as f:
                f.write(latest_revision)
            print("Done converting the weights...")
        else:
            print(
                f"Loading '{self.hf_config._name_or_path}' model weights from the cache..."
            )

    def load_hf_tokenizer(self):
        print("Loading tokenizer...")
        if len(self.tokenizer_path) > 0:
            print(f"Using tokenizer from {self.tokenizer_path}")
            # check that tokenizer exist
            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(f"Path {self.tokenizer_path} does not exist")
            elif (
                os.path.isdir(self.tokenizer_path)
                and len(os.listdir(self.tokenizer_path)) == 0
            ):
                raise FileNotFoundError(f"Folder {self.tokenizer_path} is empty")
            return

        # Download tokenizer

        # Use local cache, or download new version
        self.tokenizer_path = os.path.expanduser(
            f"~/.cache/flexflow/tokenizers/{self.hf_config._name_or_path}/"
        )
        if self.clean_cache:
            print(
                f"Discarding cached tokenizer files (if they exist) for model {self.hf_config._name_or_path}..."
            )
            if os.path.exists(self.tokenizer_path):
                shutil.rmtree(self.tokenizer_path)
        if not os.path.exists(self.tokenizer_path):
            print(f"Creating directory {self.tokenizer_path} (if it doesn't exist)...")
        os.makedirs(self.tokenizer_path, exist_ok=True)

        # Get local revision SHA, check if it matches latest one on huggingface
        local_revision = None
        local_revision_file = os.path.join(self.tokenizer_path, "rev_sha.txt")
        if os.path.exists(local_revision_file):
            local_revision = "".join(open(local_revision_file).read().split())
        hf_api = HfApi()
        latest_revision = hf_api.model_info(self.hf_config._name_or_path).sha

        # Download if needed
        if local_revision != latest_revision:
            print(
                f"'{self.hf_config._name_or_path}' tokenizer not found in cache or outdated. Downloading from huggingface.co ..."
            )
            if self.model_type == ModelType.LLAMA:
                hf_tokenizer = LlamaTokenizer.from_pretrained(
                    self.hf_config._name_or_path, use_fast=True
                )
            else:
                hf_tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_config._name_or_path
                )
            hf_tokenizer.save_pretrained(self.tokenizer_path)
            print("Done downloading HF tokenizer.")
            with open(local_revision_file, "w+") as f:
                f.write(latest_revision)
            print("Loading the tokenizer...")
        else:
            print(
                f"Loading '{self.hf_config._name_or_path}' tokenizer from the cache..."
            )

    def load_hf_weights(self):
        print("Loading hf weights...")

        if self.data_type == DataType.DT_HALF:
            torch.set_default_tensor_type(torch.HalfTensor)

        if len(self.weights_path) > 0:
            print(f"Using weights from {self.weights_path}")
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

        # Create file data loader, load weights into tensors
        self.fileloader = FileDataLoader(
            self.weights_path,
            self.hf_config.num_attention_heads,
            self.hf_config.hidden_size,
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )

        model_layers_with_weights = self.model.get_layers_with_weights()
        self.fileloader.load_weights(
            self.model.ffmodel, model_layers_with_weights, self.data_type
        )

    def compile(
        self,
        mode=InferenceMode.INC_DECODING_MODE,
        sampling_config=SamplingConfig(),
        max_batch_size=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
        ssms=[],
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.max_tokens_per_batch = max_tokens_per_batch
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

        # Create inference manager
        self.im = InferenceManager()
        self.im.compile_model_and_allocate_buffer(self.model.ffmodel)

        # Download the weights and tokenizer from huggingface (if needed) and load them
        self.load_hf_weights()
        self.load_hf_tokenizer()

        # Create request manager
        self.rm = RequestManager()
        self.rm.register_tokenizer(self.model_type, self.tokenizer_path)
        self.rm.register_output_filepath(self.output_file)

        self.im.init_operators_inference(self.model.ffmodel)

        for ssm in self.ssms:
            self.rm.register_ssm_model(ssm.model.ffmodel)

    def generate(self, prompts):
        if type(prompts) == str:
            if len(prompts) == 0:
                return None
            return self.model.ffmodel.generate(prompts, 128)
        elif type(prompts) == list:
            if len(prompts) == 0:
                return []
            return [self.model.ffmodel.generate(prompt, 128) for prompt in prompts]
        else:
            assert False, "Please pass a non-empty string or list of strings"


class SSM(LLM):
    def __init__(
        self,
        model_name,
        data_type=DataType.DT_HALF,
        tokenizer_path="",
        weights_path="",
        clean_cache=False,
        output_file="",
    ):
        super().__init__(
            model_name,
            data_type,
            tokenizer_path,
            weights_path,
            clean_cache,
            output_file,
        )
        self.ffconfig.data_parallelism_degree = 1
        self.ffconfig.tensor_parallelism_degree = 1
        self.ffconfig.pipeline_parallelism_degree = 1
