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

from flexflow.serve.models import (
    FlexFlowLLAMA,
    FlexFlowOPT,
    FlexFlowFalcon,
    FlexFlowSTARCODER,
    FlexFlowMPT,
)
from flexflow.serve.models import (
    LLAMAConfig,
    OPTConfig,
    FalconConfig,
    STARCODERConfig,
    MPTConfig,
)
from flexflow.core import *
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import PeftModel, PeftConfig
from huggingface_hub import HfApi
import torch, shutil, hashlib, json, gc
from typing import Union, List


class _SupportedModels:
    def __init__(
        self,
    ):
        self.supported_models = {
            "LlamaForCausalLM": (ModelType.LLAMA, FlexFlowLLAMA, LLAMAConfig),
            "LLaMAForCausalLM": (ModelType.LLAMA, FlexFlowLLAMA, LLAMAConfig),
            "OPTForCausalLM": (ModelType.OPT, FlexFlowOPT, OPTConfig),
            "RWForCausalLM": (ModelType.FALCON, FlexFlowFalcon, FalconConfig),
            "FalconForCausalLM": (ModelType.FALCON, FlexFlowFalcon, FalconConfig),
            "GPTBigCodeForCausalLM": (
                ModelType.STARCODER,
                FlexFlowSTARCODER,
                STARCODERConfig,
            ),
            "MPTForCausalLM": (ModelType.MPT, FlexFlowMPT, MPTConfig),
        }

    def get_ff_model_type(self, hf_config):
        architectures = getattr(hf_config, "architectures", [])
        ff_arch = None
        if next(iter(architectures), None) is not None:
            ff_arch = self.supported_models.get(architectures[0])
        if ff_arch is None:
            raise ValueError(
                f"Huggingface model of type {architectures} is not yet supported by FlexFlow"
            )
        return ff_arch


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

        :param model_name: The name of the HuggingFace model to use. E.g. 'meta-llama/Llama-2-7b-hf'
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
        self.supported_models = _SupportedModels()
        self.hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model_name = self.hf_config._name_or_path
        (
            self.model_type,
            self.model_class,
            self.config_class,
        ) = self.supported_models.get_ff_model_type(self.hf_config)
        self.data_type = data_type
        assert self.data_type == DataType.DT_HALF or self.data_type == DataType.DT_FLOAT
        self.cache_path = cache_path if len(cache_path) > 0 else "~/.cache/flexflow"
        self.refresh_cache = refresh_cache
        self.output_file = output_file
        self.rm = None
        self.pefts = []

    def __del__(self):
        # Stop the background server before deleting the object
        if type(self) == LLM and self.rm is not None:
            self.rm.stop_server()

    def add_peft(self, peft_model_id: str):
        """Add a previously created PEFT adapter to the LLM. The PEFT model should already exist locally or be available on HuggingFace"""
        peft_config = PeftConfig.from_pretrained(peft_model_id)
        peft_type = peft_config.peft_type
        if peft_type != "LORA":
            raise RuntimeError(f"PEFT type {peft_type} not yet supported in FlexFlow")
        if "base_model_name_or_path" not in peft_config.to_dict():
            raise ValueError(
                f"PEFT model {peft_model_id} does not have an associated base model"
            )
        if peft_config.base_model_name_or_path != self.model_name:
            raise RuntimeError(f"Attempting to add PEFT with base model name {peft_config.base_model_name_or_path} to LLM {self.model_name}")
        ff_peft_config = LoraLinearConfig(self.cache_path, peft_model_id)
        peft_dict = {
            "peft_config": peft_config,
            "peft_type": peft_type,
            "ff_peft_config": ff_peft_config,
        }
        self.pefts[peft_model_id] = peft_dict

    def download_hf_config(self):
        """Save the HuggingFace model configs to a json file. Useful mainly to run the C++ inference code."""
        config_dir = os.path.join(
            os.path.expanduser(self.cache_path), "configs", self.model_name.lower()
        )
        config_path = os.path.join(config_dir, "config.json")
        os.makedirs(config_dir, exist_ok=True)
        print(f"Creating directory {config_dir} (if it doesn't exist)...")
        print(f"Saving {self.model_name} configs to file {config_path}...")
        self.hf_config.to_json_file(config_path)
        
        # Save PEFT configs if the LLM has any registered PEFTs
        for peft_model_id, peft_dict in self.pefts.items():
            peft_config = peft_dict["hf_config"]
            peft_config_path = os.path.join(os.path.expanduser(self.cache_path), "configs", self.peft_model_id.lower())
            print(f"Saving {peft_model_id} configs to file {peft_config_path}...")
            with open(peft_config_path, "w") as json_file:
                class SetEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, set):
                            return list(obj)
                        return super().default(obj)
                json.dump(peft_config.to_dict(), json_file, indent=2, cls=SetEncoder)

    def __get_revision_hashes(self, model_name: str, folder: str):
        ff_revision = None
        ff_revision_file = os.path.join(folder, "rev_sha.txt")
            
        if os.path.exists(ff_revision_file):
            ff_revision = "".join(open(ff_revision_file).read().split())

        if os.path.exists(model_name) and os.path.isdir(model_name):
            # Local model
            files = os.listdir(model_name)
            state = files + [
                os.path.getmtime(os.path.join(model_name, f)) for f in files
            ]
            latest_revision = hashlib.md5(str(state).encode("utf-8")).hexdigest()
        else:
            # Remote HuggingFace model
            hf_api = HfApi()
            latest_revision = hf_api.model_info(self.model_name).sha
        return ff_revision, ff_revision_file, latest_revision

    def download_hf_weights_if_needed(self):
        """Check in the folder specified by the cache_path whether the LLM's model weights are available and up to date.
        If not, or if the refresh_cache parameter is set to True, download new weights.

        If any PEFT adapter is registered, perform the same operation for PEFT.
        """
        def get_weights_path(model_name):
            return os.path.join(os.path.expanduser(self.cache_path), "weights", model_name.lower(),
                (
                    "full-precision"
                    if self.data_type == DataType.DT_FLOAT
                    else "half-precision"
                ),
            )

        def refresh_cache_if_needed(model_name):
            weights_path = get_weights_path(model_name)
            if self.refresh_cache:
                print(
                    f"Refreshing weights in cache for model {model_name} at path {weights_path} ..."
                )
                if os.path.exists(weights_path):
                    shutil.rmtree(weights_path)
            os.makedirs(weights_path, exist_ok=True)
        
        def get_hf_llm(model_name):
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=(
                    torch.float32
                    if self.data_type == DataType.DT_FLOAT
                    else torch.float16
                ),
            )
        
        def download_llm_weights():
            weights_path = get_weights_path(self.model_name)
            refresh_cache_if_needed(self.model_name)
            ff_revision, ff_revision_file, latest_revision = self.__get_revision_hashes(self.model_name, weights_path)
            if ff_revision != latest_revision:
                print(f"'{self.model_name}' local model weights need updating! Downloading/converting new weights now...")
                hf_model = get_hf_llm(self.model_name)
                # Convert the model to FlexFlow format
                self.model_class.convert_hf_model(hf_model, weights_path)
                # Save new revision hash to file
                with open(ff_revision_file, "w+") as f:
                    f.write(latest_revision)
                print(f"Done converting the weights for model {self.model_name}")
                # Deallocate hf model
                del hf_model
                gc.collect()
                torch.cuda.empty_cache()
        
        def convert_peft_model(hf_peft_model, peft_type, weights_path):
            for name, params in hf_peft_model.named_parameters():
                if peft_type.lower() in name:
                    name = name.replace("base_model.model.model.", "").replace(
                        ".default", ""
                    )
                    name = self.model_class.convert_hf_weight_name(name)
                    params.detach().cpu().numpy().tofile(f"{weights_path}/{name}")
        
        def download_peft_weights():
            for peft_model_id, peft_dict in self.pefts.items():
                peft_config = peft_dict["peft_config"]
                peft_type = peft_config["peft_type"]
                
                weights_path = get_weights_path(peft_model_id)
                refresh_cache_if_needed(peft_model_id)
                ff_revision, ff_revision_file, latest_revision = self.__get_revision_hashes(peft_model_id, weights_path)
                
                if ff_revision != latest_revision:
                    print(f"'{peft_model_id}' local model weights need updating! Downloading/converting new weights now...")
                    hf_model = get_hf_llm(peft_model_id)
                    hf_peft_model = PeftModel.from_pretrained(hf_model, peft_model_id, config=peft_config)
                    # Convert the model to FlexFlow format
                    convert_peft_model(hf_peft_model, peft_type, weights_path)
                    # Save new revision hash to file
                    with open(ff_revision_file, "w+") as f:
                        f.write(latest_revision)
                    print(f"Done converting the weights for model {peft_model_id}")
                    # Deallocate hf model
                    del hf_peft_model
                    del hf_model
                    gc.collect()
                    torch.cuda.empty_cache()
        
        download_llm_weights()
        download_peft_weights()

    def download_hf_tokenizer_if_needed(self):
        """Check in the folder specified by the cache_path whether the LLM's tokenizer files are available and up to date.
        If not, or if the refresh_cache parameter is set to True, download new tokenizer files.
        """
        print("Loading tokenizer...")

        # Use local cache, or download new version
        tokenizer_path = os.path.join(
            os.path.expanduser(self.cache_path),
            "tokenizers",
            self.model_name.lower(),
        )
        if self.refresh_cache:
            print(f"Refreshing cached tokenizer for model {self.model_name} at path {tokenizer_path} ...")
            if os.path.exists(tokenizer_path):
                shutil.rmtree(tokenizer_path)
        if not os.path.exists(tokenizer_path):
            print(f"Creating directory {tokenizer_path} (if it doesn't exist)...")
            os.makedirs(tokenizer_path, exist_ok=True)

        # Get local revision SHA, check if it matches latest one on huggingface
        ff_revision, ff_revision_file, latest_revision = self.__get_revision_hashes(self.model_name, tokenizer_path)

        if ff_revision != latest_revision:
            print(f"'{self.model_name}' tokenizer needs updating! Downloading tokenizer now...")
            # Download tokenizer from HuggingFace, or load it from the local folder
            if self.model_type == ModelType.LLAMA:
                hf_tokenizer = LlamaTokenizer.from_pretrained(
                    self.model_name, use_fast=True
                )
            else:
                hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Save tokenizer
            hf_tokenizer.save_pretrained(tokenizer_path)
            print("Done updating HF tokenizer.")
            # Save new revision hash to file
            with open(ff_revision_file, "w+") as f:
                f.write(latest_revision)

    def compile(
        self,
        generation_config: GenerationConfig = GenerationConfig(),
        max_requests_per_batch: int = 1,
        max_seq_length: int = 256,
        max_tokens_per_batch: int = 64,
        model_specific_data_parallelism_degree: int = None,
        model_specific_tensor_parallelism_degree: int = None,
        model_specific_pipeline_parallelism_degree: int = None,
        ssms: list = [],
    ):
        """Compile the LLM for inference and load the weights into memory

        :param mode: The LLM inference mode (InferenceMode.INC_DECODING_MODE for incremental decoding, InferenceMode.BEAM_SEARCH_MODE for beam search, or InferenceMode.TREE_VERIFY_MODE for token tree verification), defaults to InferenceMode.INC_DECODING_MODE
        :type mode: InferenceMode, optional
        :param generation_config: The GenerationConfig object with the configurations to use for sampling, defaults to GenerationConfig()
        :type generation_config: GenerationConfig, optional
        :param max_requests_per_batch: The maximum batch size to allow, defaults to 1
        :type max_requests_per_batch: int, optional
        :param max_seq_length: The maximum sequence length to allow per batch, defaults to 256
        :type max_seq_length: int, optional
        :param max_tokens_per_batch: The maximum number of tokens (across requests) to allow per batch, defaults to 64
        :type max_tokens_per_batch: int, optional
        :param model_specific_data_parallelism_degree: Use this parameter if you want to give the LLM a different data parallelism degree than the one used to initialize the runtime, defaults to None
        :type model_specific_data_parallelism_degree: int, optional
        :param model_specific_tensor_parallelism_degree: Use this parameter if you want to give the LLM a different tensor parallelism degree than the one used to initialize the runtime, defaults to None
        :type model_specific_tensor_parallelism_degree: int, optional
        :param model_specific_pipeline_parallelism_degree: Use this parameter if you want to give the LLM a different pipeline parallelism degree than the one used to initialize the runtime, defaults to None
        :type model_specific_pipeline_parallelism_degree: int, optional
        :param ssms: The SSMs to use when operating in speculative inference mode, defaults to []
        :type ssms: list, optional
        """
        # self.max_requests_per_batch = max_requests_per_batch
        # self.max_seq_length = max_seq_length
        # self.max_tokens_per_batch = max_tokens_per_batch
        self.ssms = ssms
        self.generation_config = GenerationConfig()
        self.ffconfig = FFConfig()
        if len(ssms) > 0:
            assert type(self) == LLM
            mode = InferenceMode.TREE_VERIFY_MODE
        elif type(self) == SSM:
            mode = InferenceMode.BEAM_SEARCH_MODE
        else:
            assert type(self) == LLM
            mode = InferenceMode.INC_DECODING_MODE

        # Apply model-specific parallelism degrees, if needed
        if model_specific_data_parallelism_degree:
            self.ffconfig.data_parallelism_degree = (
                model_specific_data_parallelism_degree
            )
        if model_specific_tensor_parallelism_degree:
            self.ffconfig.tensor_parallelism_degree = (
                model_specific_tensor_parallelism_degree
            )
        if model_specific_pipeline_parallelism_degree:
            self.ffconfig.pipeline_parallelism_degree = (
                model_specific_pipeline_parallelism_degree
            )

        # Create request manager and set serving configuration
        self.rm = RequestManager()
        self.rm.set_max_requests_per_batch(max_requests_per_batch)
        self.rm.set_max_tokens_per_batch(max_tokens_per_batch)
        self.rm.set_max_sequence_length(max_seq_length)

        # Instantiate the relevant model
        self.model = self.model_class(
            mode,
            generation_config,
            self.ffconfig,
            self.hf_config,
            self.data_type,
            max_tokens_per_batch,
        )

        # Add PEFT layer if registered
        for _, peft_dict in self.pefts.items():
            ff_peft_config = peft_dict["ff_peft_config"]
            ff_peft_model_id = self.model.add_lora_layer(ff_peft_config)
            peft_dict["ff_peft_model_id"] = ff_peft_model_id

        # Download the weights from huggingface (if needed)
        self.download_hf_weights_if_needed()

        # Create file data loader, load weights into tensors
        model_configs = self.config_class(self.hf_config)

        self.fileloader = FileDataLoader(
            self.weights_path,
            model_configs.num_attention_heads,
            model_configs.num_key_value_heads,
            model_configs.hidden_size,
            model_configs.hidden_size // model_configs.num_attention_heads,
            self.ffconfig.tensor_parallelism_degree,
            self.data_type == DataType.DT_FLOAT,
        )

        # Register weights file loader
        self.im = InferenceManager()
        self.im.register_model_weights_loader(self.model.ffmodel, self.fileloader)

        # Download the tokenizer from huggingface (if needed) and load them
        self.download_hf_tokenizer_if_needed()

        # Create tokenizer (this must be done after we have downloaded the tokenizer
        bos_token_id = (
            -1 if self.hf_config.bos_token_id is None else self.hf_config.bos_token_id
        )
        eos_token_id = (
            -1 if self.hf_config.eos_token_id is None else self.hf_config.eos_token_id
        )
        self.rm.register_tokenizer(
            self.model_type, bos_token_id, eos_token_id, self.tokenizer_path
        )
        self.rm.register_output_filepath(self.output_file)

        for ssm in self.ssms:
            self.rm.register_ssm_model(ssm.model.ffmodel)

        # start background server
        if (mode == InferenceMode.TREE_VERIFY_MODE) or (
            mode == InferenceMode.INC_DECODING_MODE
        ):
            import atexit

            atexit.register(self.rm.stop_server)

    def generate(self, prompts: Union[str, List[str], Request, List[Request]], max_length: int = 128):
        """Generate tokens based on the input prompt(s)

        :param prompts: The generation prompt(s) in the form of a string, a list of strings, a Request, or list of Requests
        :type prompts: Union[str, List[str], Request, List[Request]]
        :return: the generation results
        :rtype: GenerationResult
        """
        if type(prompts) == str:
            if len(prompts) == 0:
                return None
            return self.model.ffmodel.generate_inf_only([prompts], max_length)
        elif type(prompts) == list:
            if len(prompts) == 0:
                return []
            return self.model.ffmodel.generate_inf_only(prompts, max_length)
        else:
            assert False, "Please pass a non-empty string or list of strings"

    def start_server(self):
        self.rm.start_server(self.model.ffmodel)
        print("Background server started.")

    def stop_server(self):
        self.rm.stop_server()
        print("Background server stopped.")


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

        :param model_name: The name of the HuggingFace model to use. E.g. 'meta-llama/Llama-2-7b-hf'
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

    def compile(
        self,
        generation_config: GenerationConfig = GenerationConfig(),
        max_requests_per_batch: int = 16,
        max_seq_length: int = 256,
        max_tokens_per_batch: int = 128,
        model_specific_data_parallelism_degree: int = 1,
        model_specific_tensor_parallelism_degree: int = 1,
        model_specific_pipeline_parallelism_degree: int = 1,
        ssms: list = [],
    ):
        """Compile the SSM for inference and load the weights into memory

        :param mode: The SSM inference mode (InferenceMode.INC_DECODING_MODE for incremental decoding, InferenceMode.BEAM_SEARCH_MODE for beam search, or InferenceMode.TREE_VERIFY_MODE for token tree verification), defaults to InferenceMode.INC_DECODING_MODE
        :type mode: InferenceMode, optional
        :param generation_config: The GenerationConfig object with the configurations to use for sampling, defaults to GenerationConfig()
        :type generation_config: GenerationConfig, optional
        :param max_requests_per_batch: The maximum batch size to allow, defaults to 16
        :type max_requests_per_batch: int, optional
        :param max_seq_length: The maximum sequence length to allow per batch, defaults to 256
        :type max_seq_length: int, optional
        :param max_tokens_per_batch: The maximum number of tokens (across requests) to allow per batch, defaults to 128
        :type max_tokens_per_batch: int, optional
        :param model_specific_data_parallelism_degree: Use this parameter if you want to give the SSM a different data parallelism degree than the default one, defaults to 1
        :type model_specific_data_parallelism_degree: int, optional
        :param model_specific_tensor_parallelism_degree: Use this parameter if you want to give the SSM a different tensor parallelism degree than the default one, defaults to 1
        :type model_specific_tensor_parallelism_degree: int, optional
        :param model_specific_pipeline_parallelism_degree: Use this parameter if you want to give the SSM a different pipeline parallelism degree than the default one, defaults to 1
        :type model_specific_pipeline_parallelism_degree: int, optional
        :param ssms: The SSMs to use when operating in speculative inference mode, defaults to []
        :type ssms: list, optional
        """
        super().compile(
            generation_config,
            max_requests_per_batch,
            max_seq_length,
            max_tokens_per_batch,
            model_specific_data_parallelism_degree,
            model_specific_tensor_parallelism_degree,
            model_specific_pipeline_parallelism_degree,
            ssms,
        )
