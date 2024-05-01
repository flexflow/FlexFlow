# FlexFlow Serve: Low-Latency, High-Performance LLM Serving
![build](https://github.com/flexflow/flexflow/workflows/build/badge.svg?branch=inference) ![gpu tests](https://github.com/flexflow/flexflow/workflows/gpu-ci/badge.svg?branch=inference) ![multinode gpu tests](https://github.com/flexflow/flexflow/workflows/multinode-test/badge.svg?branch=master) ![docker](https://github.com/flexflow/flexflow/workflows/docker-build/badge.svg?branch=inference) ![pip](https://github.com/flexflow/flexflow/workflows/pip-install/badge.svg?branch=inference) ![shell-check](https://github.com/flexflow/flexflow/workflows/Shell%20Check/badge.svg?branch=inference) ![clang-format](https://github.com/flexflow/flexflow/workflows/clang-format%20Check/badge.svg?branch=inference) [![Documentation Status](https://readthedocs.org/projects/flexflow/badge/?version=latest)](https://flexflow.readthedocs.io/en/latest/?badge=latest)


---

## News🔥:

* [09/02/2023] Adding AMD GPU support, released Docker images for ROCM 5.3->5.6
* [08/16/2023] Adding Starcoder model support
* [08/14/2023] Released Docker images for different CUDA versions

## What is FlexFlow Serve
  
The high computational and memory requirements of generative large language
models (LLMs) make it challenging to serve them quickly and cheaply. 
FlexFlow Serve is an open-source compiler and distributed system for 
__low latency__, __high performance__ LLM serving. FlexFlow Serve outperforms 
existing systems by 1.3-2.0x for single-node, multi-GPU inference and by 
1.4-2.4x for multi-node, multi-GPU inference.

<p align="center">
<img src="https://github.com/flexflow/FlexFlow/blob/inference/img/performance.png?raw=true" alt="Performance comparison" height="320"/>
</p>


## Install FlexFlow Serve


### Requirements
* OS: Linux
* GPU backend: Hip-ROCm or CUDA
	* CUDA version: 10.2 – 12.0
	* NVIDIA compute capability: 6.0 or higher
* Python: 3.6 or higher
* Package dependencies: [see here](https://github.com/flexflow/FlexFlow/blob/inference/requirements.txt)

### Install with pip
You can install FlexFlow Serve using pip:

```bash
pip install flexflow
```

### Try it in Docker
If you run into any issue during the install, or if you would like to use the C++ API without needing to install from source, you can also use our pre-built Docker package for different CUDA versions (NVIDIA backend) and multiple ROCM versions (AMD backend). To download and run our pre-built Docker container:

```bash
docker run --gpus all -it --rm --shm-size=8g ghcr.io/flexflow/flexflow-cuda-12.0:latest
```

To download a Docker container for a backend other than CUDA v12.0, you can replace the `cuda-12.0` suffix with any of the following backends: `cuda-11.1`, `cuda-11.2`, `cuda-11.3`, `cuda-11.4`, `cuda-11.5`, `cuda-11.6`, `cuda-11.7`, `cuda-11.8`, and `hip_rocm-5.3`, `hip_rocm-5.4`, `hip_rocm-5.5`, `hip_rocm-5.6`). More info on the Docker images, with instructions to build a new image from source, or run with additional configurations, can be found [here](../docker/README.md).

### Build from source

You can install FlexFlow Serve from source code by building the inference branch of FlexFlow. Please follow these [instructions](https://flexflow.readthedocs.io/en/latest/installation.html).

## Quickstart
The following example shows how to deploy an LLM using FlexFlow Serve and accelerate its serving using [speculative inference](#speculative-inference). First, we import `flexflow.serve` and initialize the FlexFlow Serve runtime. Note that `memory_per_gpu` and `zero_copy_memory_per_node` specify the size of device memory on each GPU (in MB) and zero-copy memory on each node (in MB), respectively. 
We need to make sure the aggregated GPU memory and zero-copy memory are **both** sufficient to store LLM parameters in non-offloading serving. FlexFlow Serve combines tensor and pipeline model parallelism for LLM serving.
```python
import flexflow.serve as ff

ff.init(
        num_gpus=4,
        memory_per_gpu=14000,
        zero_copy_memory_per_node=30000,
        tensor_parallelism_degree=4,
        pipeline_parallelism_degree=1
    )
```
Second, we specify the LLM to serve and the SSM(s) used to accelerate LLM serving. The list of supported LLMs and SSMs is available at [supported models](#supported-llms-and-ssms).
```python
# Specify the LLM
llm = ff.LLM("meta-llama/Llama-2-7b-hf")

# Specify a list of SSMs (just one in this case)
ssms=[]
ssm = ff.SSM("JackFram/llama-68m")
ssms.append(ssm)
```
Next, we declare the generation configuration and compile both the LLM and SSMs. Note that all SSMs should run in the **beam search** mode, and the LLM should run in the **tree verification** mode to verify the speculated tokens from SSMs. You can also use the following arguments to specify serving configuration when compiling LLMs and SSMs:

* max\_requests\_per\_batch: the maximum number of requests to serve in a batch (default: 16)
* max\_seq\_length: the maximum number of tokens in a request (default: 256)
* max\_tokens\_per\_batch: the maximum number of tokens to process in a batch (default: 128)

```python
# Create the sampling configs
generation_config = ff.GenerationConfig(
    do_sample=False, temperature=0.9, topp=0.8, topk=1
)

# Compile the SSMs for inference and load the weights into memory
for ssm in ssms:
    ssm.compile(generation_config)

# Compile the LLM for inference and load the weights into memory
llm.compile(generation_config,
            max_requests_per_batch = 16,
            max_seq_length = 256,
            max_tokens_per_batch = 128,
            ssms=ssms)
```
Next, we call `llm.start_server()` to start an LLM server running on a seperate background thread, which allows users to perform computations in parallel with LLM serving. Finally, we call `llm.generate` to generate the output, which is organized as a list of `GenerationResult`, which include the output tokens and text. After all serving requests are processed, you can either call `llm.stop_server()` to terminate the background thread or directly exit the python program, which will automatically terminate the background server thread.
```python
llm.start_server()
result = llm.generate("Here are some travel tips for Tokyo:\n")
llm.stop_server() # This invocation is optional
```

### Incremental decoding
<details>
<summary>Expand here</summary>
<br>

```python
import flexflow.serve as ff

# Initialize the FlexFlow runtime. ff.init() takes a dictionary or the path to a JSON file with the configs
ff.init(
        num_gpus=4,
        memory_per_gpu=14000,
        zero_copy_memory_per_node=30000,
        tensor_parallelism_degree=4,
        pipeline_parallelism_degree=1
    )

# Create the FlexFlow LLM
llm = ff.LLM("meta-llama/Llama-2-7b-hf")

# Create the sampling configs
generation_config = ff.GenerationConfig(
    do_sample=True, temperature=0.9, topp=0.8, topk=1
)

# Compile the LLM for inference and load the weights into memory
llm.compile(generation_config,
            max_requests_per_batch = 16,
            max_seq_length = 256,
            max_tokens_per_batch = 128)

# Generation begins!
llm.start_server()
result = llm.generate("Here are some travel tips for Tokyo:\n")
llm.stop_server() # This invocation is optional
```

</details>

### C++ interface
If you'd like to use the C++ interface (mostly used for development and benchmarking purposes), you should install from source, and follow the instructions below. 

<details>
<summary>Expand here</summary>
<br>

#### Downloading models
Before running FlexFlow Serve, you should manually download the LLM and SSM(s) model of interest using the [inference/utils/download_hf_model.py](https://github.com/flexflow/FlexFlow/blob/inference/inference/utils/download_hf_model.py) script (see example below). By default, the script will download all of a model's assets (weights, configs, tokenizer files, etc...) into the cache folder `~/.cache/flexflow`. If you would like to use a different folder, you can request that via the parameter `--cache-folder`.

```bash
python3 ./inference/utils/download_hf_model.py <HF model 1> <HF model 2> ...
```

#### Running the C++ examples
A C++ example is available at [this folder](../inference/spec_infer/). After building FlexFlow Serve, the executable will be available at `/build_dir/inference/spec_infer/spec_infer`. You can use the following command-line arguments to run FlexFlow Serve:

* `-ll:gpu`: number of GPU processors to use on each node for serving an LLM (default: 0)
* `-ll:fsize`: size of device memory on each GPU in MB
* `-ll:zsize`: size of zero-copy memory (pinned DRAM with direct GPU access) in MB. FlexFlow Serve keeps a replica of the LLM parameters on zero-copy memory, and therefore requires that the zero-copy memory is sufficient for storing the LLM parameters.
* `-llm-model`: the LLM model ID from HuggingFace (e.g. "meta-llama/Llama-2-7b-hf")
* `-ssm-model`: the SSM model ID from HuggingFace (e.g. "JackFram/llama-160m"). You can use multiple `-ssm-model`s in the command line to launch multiple SSMs.
* `-cache-folder`: the folder
* `-data-parallelism-degree`, `-tensor-parallelism-degree` and `-pipeline-parallelism-degree`: parallelization degrees in the data, tensor, and pipeline dimensions. Their product must equal the number of GPUs available on the machine. When any of the three parallelism degree arguments is omitted, a default value of 1 will be used. 
* `-prompt`: (optional) path to the prompt file. FlexFlow Serve expects a json format file for prompts. In addition, users can also use the following API for registering requests:
* `-output-file`: (optional) filepath to use to save the output of the model, together with the generation latency

For example, you can use the following command line to serve a LLaMA-7B or LLaMA-13B model on 4 GPUs and use two collectively boost-tuned LLaMA-68M models for speculative inference.

```bash
./inference/spec_infer/spec_infer -ll:gpu 4 -ll:cpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-68m -prompt /path/to/prompt.json -tensor-parallelism-degree 4 --fusion
```
</details>

## Speculative Inference
A key technique that enables FlexFlow Serve to accelerate LLM serving is speculative
inference, which combines various collectively boost-tuned small speculative
models (SSMs) to jointly predict the LLM’s outputs; the predictions are organized as a
token tree, whose nodes each represent a candidate token sequence. The correctness
of all candidate token sequences represented by a token tree is verified against the
LLM’s output in parallel using a novel tree-based parallel decoding mechanism.
FlexFlow Serve uses an LLM as a token tree verifier instead of an incremental decoder,
which largely reduces the end-to-end inference latency and computational requirement
for serving generative LLMs while provably preserving model quality.

<p align="center">
<img src="https://github.com/flexflow/FlexFlow/blob/inference/img/spec_infer_demo.gif?raw=true" alt="A Speculative Inference Demo" width="630"/>
</p>

### Supported LLMs and SSMs

FlexFlow Serve currently supports all HuggingFace models with the following architectures:
* `LlamaForCausalLM` / `LLaMAForCausalLM` (e.g. LLaMA/LLaMA-2, Guanaco, Vicuna, Alpaca, ...)
* `OPTForCausalLM` (models from the OPT family)
* `RWForCausalLM` (models from the Falcon family)
* `GPTBigCodeForCausalLM` (models from the Starcoder family)

Below is a list of models that we have explicitly tested and for which a SSM may be available:

| Model | Model id on HuggingFace | Boost-tuned SSMs |
| :---- | :---- | :---- |
| LLaMA-7B | meta-llama/Llama-2-7b-hf | [LLaMA-68M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-13B | decapoda-research/llama-13b-hf | [LLaMA-68M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-30B | decapoda-research/llama-30b-hf | [LLaMA-68M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-65B | decapoda-research/llama-65b-hf | [LLaMA-68M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-2-7B | meta-llama/Llama-2-7b-hf | [LLaMA-68M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-2-13B | meta-llama/Llama-2-13b-hf | [LLaMA-68M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-2-70B | meta-llama/Llama-2-70b-hf | [LLaMA-68M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| OPT-6.7B | facebook/opt-6.7b | [OPT-125M](https://huggingface.co/facebook/opt-125m) |
| OPT-13B | facebook/opt-13b | [OPT-125M](https://huggingface.co/facebook/opt-125m) |
| OPT-30B | facebook/opt-30b | [OPT-125M](https://huggingface.co/facebook/opt-125m) |
| OPT-66B | facebook/opt-66b | [OPT-125M](https://huggingface.co/facebook/opt-125m) |
| Falcon-7B | tiiuae/falcon-7b | |
| Falcon-40B | tiiuae/falcon-40b | |
| StarCoder-7B | bigcode/starcoderbase-7b | |
| StarCoder-15.5B | bigcode/starcoder | |

### CPU Offloading
FlexFlow Serve also offers offloading-based inference for running large models (e.g., llama-7B) on a single GPU. CPU offloading is a choice to save tensors in CPU memory, and only copy the tensor to GPU when doing calculation. Notice that now we selectively offload the largest weight tensors (weights tensor in Linear, Attention). Besides, since the small model occupies considerably less space, it it does not pose a bottleneck for GPU memory, the offloading will bring more runtime space and computational cost, so we only do the offloading for the large model. [TODO: update instructions] You can run the offloading example by enabling the `-offload` and `-offload-reserve-space-size` flags.

### Quantization
FlexFlow Serve supports int4 and int8 quantization. The compressed tensors are stored on the CPU side. Once copied to the GPU, these tensors undergo decompression and conversion back to their original precision. Please find the compressed weight files in our s3 bucket, or use [this script](../inference/utils/compress_llama_weights.py) from [FlexGen](https://github.com/FMInference/FlexGen) project to do the compression manually.

### Prompt Datasets
We provide five prompt datasets for evaluating FlexFlow Serve: [Chatbot instruction prompts](https://specinfer.s3.us-east-2.amazonaws.com/prompts/chatbot.json), [ChatGPT Prompts](https://specinfer.s3.us-east-2.amazonaws.com/prompts/chatgpt.json), [WebQA](https://specinfer.s3.us-east-2.amazonaws.com/prompts/webqa.json), [Alpaca](https://specinfer.s3.us-east-2.amazonaws.com/prompts/alpaca.json), and [PIQA](https://specinfer.s3.us-east-2.amazonaws.com/prompts/piqa.json).

## TODOs

FlexFlow Serve is under active development. We currently focus on the following tasks and strongly welcome all contributions from bug fixes to new features and extensions.

* AMD benchmarking. We are actively working on benchmarking FlexFlow Serve on AMD GPUs and comparing it with the performance on NVIDIA GPUs.
* Chatbot prompt templates and Multi-round conversations
* Support for FastAPI server
* Integration with LangChain for document question answering

## Acknowledgements
This project is initiated by members from CMU, Stanford, and UCSD. We will be continuing developing and supporting FlexFlow Serve. Please cite FlexFlow Serve as:

``` bibtex
@misc{miao2023specinfer,
      title={SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification}, 
      author={Xupeng Miao and Gabriele Oliaro and Zhihao Zhang and Xinhao Cheng and Zeyu Wang and Rae Ying Yee Wong and Alan Zhu and Lijie Yang and Xiaoxiang Shi and Chunan Shi and Zhuoming Chen and Daiyaan Arfeen and Reyna Abhyankar and Zhihao Jia},
      year={2023},
      eprint={2305.09781},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
FlexFlow uses Apache License 2.0.
