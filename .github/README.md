# FlexFlow Serve: Low-Latency, High-Performance LLM Serving
![build](https://github.com/flexflow/flexflow/workflows/build/badge.svg?branch=master) ![gpu tests](https://github.com/flexflow/flexflow/workflows/gpu-ci/badge.svg?branch=master) ![multinode gpu tests](https://github.com/flexflow/flexflow/workflows/multinode-test/badge.svg?branch=master) ![docker](https://github.com/flexflow/flexflow/workflows/docker-build/badge.svg?branch=master) ![pip](https://github.com/flexflow/flexflow/workflows/pip-install/badge.svg?branch=master) ![shell-check](https://github.com/flexflow/flexflow/workflows/Shell%20Check/badge.svg?branch=master) ![clang-format](https://github.com/flexflow/flexflow/workflows/clang-format%20Check/badge.svg?branch=master) [![Documentation Status](https://readthedocs.org/projects/flexflow/badge/?version=latest)](https://flexflow.readthedocs.io/en/latest/?badge=latest)


---

## News:

* [08/14/2023] Released Dockerfile for different CUDA versions

## What is FlexFlow Serve
  
The high computational and memory requirements of generative large language
models (LLMs) make it challenging to serve them quickly and cheaply. 
FlexFlow Serve is an open-source compiler and distributed system for 
__low latency__, __high performance__ LLM serving. FlexFlow Serve outperforms 
existing systems by 1.3-2.0x for single-node, multi-GPU inference and by 
1.4-2.4x for multi-node, multi-GPU inference.

that accelerates generative LLM
inference with __speculative inference__ and __token tree verification__. A key insight
behind SpecInfer is to combine various collectively boost-tuned small speculative
models (SSMs) to jointly predict the LLM’s outputs; the predictions are organized as a
token tree, whose nodes each represent a candidate token sequence. The correctness
of all candidate token sequences represented by a token tree is verified against the
LLM’s output in parallel using a novel tree-based parallel decoding mechanism.
SpecInfer uses an LLM as a token tree verifier instead of an incremental decoder,
which largely reduces the end-to-end inference latency and computational requirement
for serving generative LLMs while provably preserving model quality.

<p align="center">
<img src="../img/performance.png" alt="Performance comparison" height="320"/>
</p>

## Install FlexFlow Serve

The quickest way to try FlexFlow Serve is to use our pre-built Docker package for different CUDA versions and the `hip_rocm` backend. More info on the Docker images can be found [here](../docker/README.md). To download and run our pre-built Docker container:
```bash
./docker/pull.sh flexflow
./docker/run.sh flexflow
```

You can also install FlexFlow Serve using pip or [from source code](https://flexflow.readthedocs.io/en/latest/installation.html):
```bash
pip install flexflow
```

SpecInfer is built on top of FlexFlow. You can build/install SpecInfer by building the inference branch of FlexFlow. Please read the [instructions](../INSTALL.md) for building/installing FlexFlow from source code. If you would like to quickly try SpecInfer, we also provide pre-built Docker packages ([specinfer-cuda](https://github.com/flexflow/FlexFlow/pkgs/container/specinfer-cuda) with a CUDA backend, [specinfer-hip_rocm](https://github.com/flexflow/FlexFlow/pkgs/container/specinfer-hip_rocm) with a HIP-ROCM backend) with all dependencies pre-installed (N.B.: currently, the CUDA pre-built containers are only fully compatible with host machines that have CUDA 11.7 installed), together with [Dockerfiles](./docker) if you wish to build the containers manually. 

## Run FlexFlow Serve
[TODO: update instructions to run FlexFlow Serve.]
The source code of the SpecInfer pipeline is available at [this folder](../inference/spec_infer/). The SpecInfer executable will be available at `/build_dir/inference/spec_infer/spec_infer` at compilation. You can use the following command-line arguments to run SpecInfer:

* `-ll:gpu`: number of GPU processors to use on each node for serving an LLM (default: 0)
* `-ll:fsize`: size of device memory on each GPU in MB
* `-ll:zsize`: size of zero-copy memory (pinned DRAM with direct GPU access) in MB. SpecInfer keeps a replica of the LLM parameters on zero-copy memory, and therefore requires that the zero-copy memory is sufficient for storing the LLM parameters.
* `-llm-model`: the LLM model type as a case-insensitive string (e.g. "opt" or "llama")
* `-llm-weight`: path to the folder that stores the LLM weights
* `-llm-config`: path to the json file that stores the LLM model configs
* `-ssm-model`: the LLM model type as a case-insensitive string (e.g. "opt" or "llama"). You can use multiple `-ssm-model`s in the command line to launch multiple SSMs.
* `-ssm-weight`: path to the folder that stores the small speculative models' weights. The number of `-ssm-weight`s must match the number of `-ssm-model`s and `-ssm-config`s.
* `-ssm-config`: path to the json file that stores the SSM model configs. The number of `-ssm-config`s must match the number of `-ssm-model`s and `-ssm-weight`s.
* `-tokenizer`: path to the tokenizer file (see [Tokenizers](#tokenizers) for preparing a tokenizer for SpecInfer).
* `-data-parallelism-degree`, `-tensor-parallelism-degree` and `-pipeline-parallelism-degree`: parallelization degrees in the data, tensor, and pipeline dimensions. Their product must equal the number of GPUs available on the machine. When any of the three parallelism degree arguments is omitted, a default value of 1 will be used. 
* `-prompt`: (optional) path to the prompt file. SpecInfer expects a json format file for prompts, all of which will be served by SpecInfer. In addition, users can also use the following API for registering requests:
* `-output-file`: (optional) filepath to use to save the output of the model, together with the generation latency


```c++
class RequestManager {
  RequestGuid register_new_request(std::string const &prompt, int max_sequence_length);
}
```
For example, you can use the following command line to serve a LLaMA-7B or LLaMA-13B model on 4 GPUs and use two collectively boost-tuned LLaMA-190M models for speculative inference.

```bash
./inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight /path/to/llm/weights -llm-config /path/to/llm/config.json -ssm-model llama -ssm-weight /path/to/ssm1/weights -ssm-config /path/to/ssm/config.json -ssm-model llama -smm-weight /path/to/ssm2/weights -ssm-config /path/to/ssm2/config.json -tokenizer /path/to/tokenizer.model -prompt /path/to/prompt.json --use-full-precision -tensor-parallelism-degree 2 -pipeline-parallelism-degree 2
```

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

### Supported LLMs and SSMs

FlexFlow Serve supports a variety of HuggingFace models:

| Model | Model id on HuggingFace | Boost-tuned SSMs |
| :---- | :---- | :---- |
| LLaMA-7B | decapoda-research/llama-7b-hf | [LLaMA-65M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-13B | decapoda-research/llama-13b-hf | [LLaMA-65M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-30B | decapoda-research/llama-30b-hf | [LLaMA-65M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-65B | decapoda-research/llama-65b-hf | [LLaMA-65M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-2-7B | meta-llama/Llama-2-7b-hf | [LLaMA-65M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-2-13B | meta-llama/Llama-2-13b-hf | [LLaMA-65M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| LLaMA-2-70B | meta-llama/Llama-2-70b-hf | [LLaMA-65M](https://huggingface.co/JackFram/llama-68m) , [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) |
| OPT-6.7B | facebook/opt-6.7b | [OPT-125M](https://huggingface.co/facebook/opt-125m) |
| OPT-13B | facebook/opt-13b | [OPT-125M](https://huggingface.co/facebook/opt-125m) |
| OPT-30B | facebook/opt-30b | [OPT-125M](https://huggingface.co/facebook/opt-125m) |
| OPT-66B | facebook/opt-66b | [OPT-125M](https://huggingface.co/facebook/opt-125m) |
| Falcon-7B | tiiuae/falcon-7b | |
| Falcon-40B | tiiuae/falcon-40b | |
| StarCoder-15.5B | bigcode/starcoder | |


### CPU Offloading
FlexFlow Serve also offers offloading-based inference for running large models (e.g., llama-7B) on a single GPU. CPU offloading is a choice to save tensors in CPU memory, and only copy the tensor to GPU when doing calculation. Notice that now we selectively offload the largest weight tensors (weights tensor in Linear, Attention). Besides, since the small model occupies considerably less space, it it does not pose a bottleneck for GPU memory, the offloading will bring more runtime space and computational cost, so we only do the offloading for the large model. [TODO: update instructions] You can run the offloading example by enabling the `-offload` and `-offload-reserve-space-size` flags.

### Quantization
FlexFlow Serve supports int4 and int8 quantization. The compressed tensors are stored on the CPU side. Once copied to the GPU, these tensors undergo decompression and conversion back to their original precision. Please find the compressed weight files in our s3 bucket, or use [this script](../inference/utils/compress_llama_weights.py) from [FlexGen](https://github.com/FMInference/FlexGen) project to do the compression manually. [TODO: update instructions for quantization].

### Prompt Datasets
We have evaluated SpecInfer on the following prompts datasets: [Chatbot instruction prompts](https://specinfer.s3.us-east-2.amazonaws.com/prompts/chatbot.json), [ChatGPT Prompts](https://specinfer.s3.us-east-2.amazonaws.com/prompts/chatgpt.json), [WebQA](https://specinfer.s3.us-east-2.amazonaws.com/prompts/webqa.json), [Alpaca](https://specinfer.s3.us-east-2.amazonaws.com/prompts/alpaca.json), and [PIQA](https://specinfer.s3.us-east-2.amazonaws.com/prompts/piqa.json).

## TODOs

SpecInfer is under active development. We currently focus on the following tasks and strongly welcome all contributions to SpecInfer from bug fixes to new features and extensions.

* Low-precision and mixed-precision support. The current version uses single-precision floating points for computing tree attention. We are actively working on support half-precision floating points, and int4 and int8 quantizations.
* Offloading-based generative LLM inference. Another promising avenue for future work is using speculative inference and token tree verification to reduce the end-to-end inference for offloading-based generative LLM inference. A potential application of this technique is enabling a single commodity GPU to serve LLMs for latency critical tasks. 

## Acknowledgements
This project is initiated by members from CMU, Stanford, and UCSD. We will be continuing developing and supporting FlexFlow Serve. The following paper describes design, implementation, and key optimizations of SpecInfer.

* Xupeng Miao*, Gabriele Oliaro*, Zhihao Zhang*, Xinhao Cheng*, Zeyu Wang, Rae Ying Yee Wong, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. [SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification](https://arxiv.org/abs/2305.09781).

\* Denotes equal contribution

### Citation
Please cite as:

``` bibtex
@misc{miao2023specinfer,
      title={SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification}, 
      author={Xupeng Miao and Gabriele Oliaro and Zhihao Zhang and Xinhao Cheng and Zeyu Wang and Rae Ying Yee Wong and Zhuoming Chen and Daiyaan Arfeen and Reyna Abhyankar and Zhihao Jia},
      year={2023},
      eprint={2305.09781},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
FlexFlow uses Apache License 2.0.
