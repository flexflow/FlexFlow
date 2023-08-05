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
from typing import Union, List


class SamplingConfig:
    """A class to store the sampling configs."""

    def __init__(
        self,
        do_sample: bool = False,
        temperature: float = 0.9,
        topp: float = 0.8,
        topk: int = 1,
    ):
        """Initialize the sampling configs

        :param do_sample: Whether to perform sampling, or use greedy decoding, defaults to False
        :type do_sample: bool, optional
        :param temperature: The temperature setting, defaults to 0.9
        :type temperature: float, optional
        :param topp: The top probabilities (top-p) setting, defaults to 0.8
        :type topp: float, optional
        :param topk: The top-k setting, defaults to 1
        :type topk: int, optional
        """
        self.do_sample = do_sample
        self.temperature = temperature
        self.topp = topp
        self.topk = topk


class LLM:
    """This class creates a LLM (Large-Language Model) object based on a model from HuggingFace"""

    def __init__(
        self,
        model_name: str,
        data_type: DataType = DataType.DT_HALF,
        cache_path: str = "",
        refresh_cache: bool = False,
        output_file: str = "",
    ):
        """Create the LLM object

        :param model_name: The name of the HuggingFace model to use. E.g. 'decapoda-research/llama-7b-hf'
        :type model_name: str
        :param data_type: The data type to use for the tensors (e.g. DataType.DT_FLOAT for full precision, or DataType.DT_HALF for half precision), defaults to DataType.DT_HALF
        :type data_type: DataType, optional
        :param cache_path: Path to the folder (which will be created if it does not yet exist) to use for the FlexFlow weights/tokenizers cache, defaults to "~/.cache/flexflow"
        :type tokenizer_path: str, optional
        :param refresh_cache: Use this flag to force the refresh of the model's weights/tokenizer cache, defaults to False
        :type refresh_cache: bool, optional
        :param output_file: Path to the output file. If left blank, the output will not be written to file, defaults to ""
        :type output_file: str, optional
        """
        self.supported_models = {
            "LlamaForCausalLM": (ModelType.LLAMA, FlexFlowLLAMA),
            "LLaMAForCausalLM": (ModelType.LLAMA, FlexFlowLLAMA),
            "OPTForCausalLM": (ModelType.OPT, FlexFlowOPT),
            "RWForCausalLM": (ModelType.FALCON, FlexFlowFalcon),
        }
        self.hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model_name = self.hf_config._name_or_path
        self.model_type, self.model_class = self.__get_ff_model_type()
        self.data_type = data_type
        assert self.data_type == DataType.DT_HALF or self.data_type == DataType.DT_FLOAT
        self.cache_path = cache_path if len(cache_path) > 0 else "~/.cache/flexflow"
        self.refresh_cache = refresh_cache
        self.output_file = output_file

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

    def download_hf_config(self):
        """Save the HuggingFace model configs to a json file. Useful mainly to run the C++ inference code."""
        self.config_dir = os.path.join(
            os.path.expanduser(self.cache_path), "configs", self.model_name.lower()
        )
        self.config_path = os.path.join(self.config_dir, "config.json")
        os.makedirs(self.config_dir, exist_ok=True)
        print(f"Creating directory {self.config_dir} (if it doesn't exist)...")
        print(f"Saving {self.model_name} configs to file {self.config_path}...")
        self.hf_config.to_json_file(self.config_path)

    def download_hf_weights_if_needed(self):
        """Check in the folder specified by the cache_path whether the LLM's model weights are available and up to date.
        If not, or if the refresh_cache parameter is set to True, download new weights.
        """
        # Use local cache, or download new version
        self.weights_path = os.path.join(
            os.path.expanduser(self.cache_path),
            "weights",
            self.model_name.lower(),
            "full-precision"
            if self.data_type == DataType.DT_FLOAT
            else "half-precision",
        )
        if self.refresh_cache:
            print(
                f"Refreshing weights in cache for model {self.model_name} at path {self.weights_path} ..."
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
        latest_revision = hf_api.model_info(self.model_name).sha

        # Download if needed
        if local_revision != latest_revision:
            print(
                f"'{self.model_name}' model weights not found in cache or outdated. Downloading from huggingface.co ..."
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            print("Done downloading HF weights. Converting them now...")
            self.model_class.convert_hf_model(hf_model, self.weights_path)
            with open(local_revision_file, "w+") as f:
                f.write(latest_revision)
            print("Done converting the weights...")
        else:
            print(f"Loading '{self.model_name}' model weights from the cache...")

    def download_hf_tokenizer_if_needed(self):
        """Check in the folder specified by the cache_path whether the LLM's tokenizer files are available and up to date.
        If not, or if the refresh_cache parameter is set to True, download new tokenizer files.
        """
        print("Loading tokenizer...")

        # Use local cache, or download new version
        self.tokenizer_path = os.path.join(
            os.path.expanduser(self.cache_path),
            "tokenizers",
            self.model_name.lower(),
        )
        if self.refresh_cache:
            print(
                f"Discarding cached tokenizer files (if they exist) for model {self.model_name}..."
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
        latest_revision = hf_api.model_info(self.model_name).sha

        # Download if needed
        if local_revision != latest_revision:
            print(
                f"'{self.model_name}' tokenizer not found in cache or outdated. Downloading from huggingface.co ..."
            )
            if self.model_type == ModelType.LLAMA:
                hf_tokenizer = LlamaTokenizer.from_pretrained(
                    self.model_name, use_fast=True
                )
            else:
                hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            hf_tokenizer.save_pretrained(self.tokenizer_path)
            print("Done downloading HF tokenizer.")
            with open(local_revision_file, "w+") as f:
                f.write(latest_revision)
            print("Loading the tokenizer...")
        else:
            print(f"Loading '{self.model_name}' tokenizer from the cache...")

    def __load_hf_weights(self):
        print("Loading hf weights...")

        if self.data_type == DataType.DT_HALF:
            torch.set_default_tensor_type(torch.HalfTensor)

        self.download_hf_weights_if_needed()

        # Create file data loader, load weights into tensors
        self.fileloader = FileDataLoader(
            self.weights_path,
            self.hf_config.num_attention_heads,
            self.hf_config.num_attention_heads,
            self.hf_config.hidden_size,
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
            self.ffconfig.tensor_parallelism_degree,
        )

        model_layers_with_weights = self.model.get_layers_with_weights()
        self.fileloader.load_weights(
            self.model.ffmodel, model_layers_with_weights, self.data_type
        )

    def compile(
        self,
        mode: InferenceMode = InferenceMode.INC_DECODING_MODE,
        sampling_config: SamplingConfig = SamplingConfig(),
        max_batch_size: int = 1,
        max_seq_length: int = 256,
        max_tokens_per_batch: int = 64,
        ssms: list = [],
    ):
        """Compile the LLM for inference and load the weights into memory

        :param mode: The LLM inference mode (InferenceMode.INC_DECODING_MODE for incremental decoding, InferenceMode.BEAM_SEARCH_MODE for beam search, or InferenceMode.TREE_VERIFY_MODE for token tree verification), defaults to InferenceMode.INC_DECODING_MODE
        :type mode: InferenceMode, optional
        :param sampling_config: The SamplingConfig object with the configurations to use for sampling, defaults to SamplingConfig()
        :type sampling_config: SamplingConfig, optional
        :param max_batch_size: The maximum batch size to allow, defaults to 1
        :type max_batch_size: int, optional
        :param max_seq_length: The maximum sequence length to allow per batch, defaults to 256
        :type max_seq_length: int, optional
        :param max_tokens_per_batch: The maximum number of tokens (across requests) to allow per batch, defaults to 64
        :type max_tokens_per_batch: int, optional
        :param ssms: The SSMs to use when operating in speculative inference mode, defaults to []
        :type ssms: list, optional
        """
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.max_tokens_per_batch = max_tokens_per_batch
        self.ssms = ssms
        self.sampling_config = SamplingConfig()
        self.ffconfig = FFConfig()
        assert (
            mode == InferenceMode.INC_DECODING_MODE
            or mode == InferenceMode.BEAM_SEARCH_MODE
        ) == (len(ssms) == 0)

        # Instantiate the relevant model
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
        self.__load_hf_weights()
        self.download_hf_tokenizer_if_needed()

        # Create request manager
        self.rm = RequestManager()
        self.rm.register_tokenizer(self.model_type, self.tokenizer_path)
        self.rm.register_output_filepath(self.output_file)

        self.im.init_operators_inference(self.model.ffmodel)

        for ssm in self.ssms:
            self.rm.register_ssm_model(ssm.model.ffmodel)

    def generate(self, prompts: Union[str, List[str]]):
        """Generate tokens based on the input prompt(s)

        :param prompts: The generation prompt(s) in the form of a string, or list of strings
        :type prompts: Union[str, List[str]]
        :return: the generation results
        :rtype: GenerationResult
        """
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
    """This class creates a SSM (Small-Speculative Model) object based on a model from HuggingFace"""

    def __init__(
        self,
        model_name: str,
        data_type: DataType = DataType.DT_HALF,
        cache_path: str = "~/.cache/flexflow",
        refresh_cache: bool = False,
        output_file: str = "",
    ):
        """Create the SSM object

        :param model_name: The name of the HuggingFace model to use. E.g. 'decapoda-research/llama-7b-hf'
        :type model_name: str
        :param data_type: The data type to use for the tensors (e.g. DataType.DT_FLOAT for full precision, or DataType.DT_HALF for half precision), defaults to DataType.DT_HALF
        :type data_type: DataType, optional
        :param cache_path: Path to the folder (which will be created if it does not yet exist) to use for the FlexFlow weights/tokenizers cache, defaults to "~/.cache/flexflow"
        :type tokenizer_path: str, optional
        :param refresh_cache: Use this flag to force the refresh of the model's weights/tokenizer cache, defaults to False
        :type refresh_cache: bool, optional
        :param output_file: Path to the output file. If left blank, the output will not be written to file, defaults to ""
        :type output_file: str, optional
        """
        super().__init__(
            model_name,
            data_type,
            cache_path,
            refresh_cache,
            output_file,
        )
