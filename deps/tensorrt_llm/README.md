## Custom AllReduce Implementation

This is an adapted version of the custom AllReduce plugin from NVIDIA's [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) repository.

To replace the NCCL AllReduce call, we should also add a CUDA IPC support to the custom AllReduce usage. Our IPC&AllReduce implementation is referenced from [mlc-ai/relax](https://github.com/mlc-ai/relax).
