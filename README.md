# FlexFlow: Low-Latency, High-Performance Training and Serving
![build](https://github.com/flexflow/flexflow/workflows/build/badge.svg?branch=inference) ![gpu tests](https://github.com/flexflow/flexflow/workflows/gpu-ci/badge.svg?branch=inference) ![multinode gpu tests](https://github.com/flexflow/flexflow/workflows/multinode-test/badge.svg?branch=master) ![docker](https://github.com/flexflow/flexflow/workflows/docker-build/badge.svg?branch=inference) ![pip](https://github.com/flexflow/flexflow/workflows/pip-install/badge.svg?branch=inference) ![shell-check](https://github.com/flexflow/flexflow/workflows/Shell%20Check/badge.svg?branch=inference) ![clang-format](https://github.com/flexflow/flexflow/workflows/clang-format%20Check/badge.svg?branch=inference) [![Documentation Status](https://readthedocs.org/projects/flexflow/badge/?version=latest)](https://flexflow.readthedocs.io/en/latest/?badge=latest)


---

## News 🔥:

* [09/02/2023] Adding AMD GPU support, released Docker images for ROCM 5.3->5.6
* [08/16/2023] Adding Starcoder model support
* [08/14/2023] Released Docker image for different CUDA versions

## Install FlexFlow


### Requirements
* OS: Linux
* GPU backend: Hip-ROCm or CUDA
  * CUDA version: 10.2 – 12.0
  * NVIDIA compute capability: 6.0 or higher
* Python: 3.6 or higher
* Package dependencies: [see here](https://github.com/flexflow/FlexFlow/blob/inference/requirements.txt)

### Install with pip
You can install FlexFlow using pip:

```bash
pip install flexflow
```

### Try it in Docker
If you run into any issue during the install, or if you would like to use the C++ API without needing to install from source, you can also use our pre-built Docker package for different CUDA versions and the `hip_rocm` backend. To download and run our pre-built Docker container:

```bash
docker run --gpus all -it --rm --shm-size=8g ghcr.io/flexflow/flexflow-cuda-12.0:latest
```

To download a Docker container for a backend other than CUDA v12.0, you can replace the `cuda-12.0` suffix with any of the following backends: `cuda-11.1`, `cuda-11.6`, `cuda-11.7`, `cuda-11.8`, `cuda-12.0`, `cuda-12.1`, `cuda-12.1`, and `hip_rocm-5.3`, `hip_rocm-5.4`, `hip_rocm-5.5`, `hip_rocm-5.6`. More info on the Docker images, with instructions to build a new image from source, or run with additional configurations, can be found [here](./docker/README.md).

### Build from source

You can install FlexFlow Serve from source code by building the inference branch of FlexFlow. Please follow these [instructions](https://flexflow.readthedocs.io/en/latest/installation.html).


## Get Started!

To get started, check out the quickstart guides below for the FlexFlow training and serving libraries.

* [FlexFlow Train](./TRAIN.md)
* [FlexFlow Serve](./SERVE.md)


## Contributing

Please let us know if you encounter any bugs or have any suggestions by [submitting an issue](https://github.com/flexflow/flexflow/issues).

We welcome all contributions to FlexFlow from bug fixes to new features and extensions.

## Citations

**FlexFlow Serve:**

* Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, Zhihao Jia. [SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification](https://arxiv.org/abs/2305.09781). In ArXiV, May 2023.


**FlexFlow Train:**

* Colin Unger, Zhihao Jia, Wei Wu, Sina Lin, Mandeep Baines, Carlos Efrain Quintero Narvaez, Vinay Ramakrishnaiah, Nirmal Prajapati, Pat McCormick, Jamaludin Mohd-Yusof, Xi Luo, Dheevatsa Mudigere, Jongsoo Park, Misha Smelyanskiy, and Alex Aiken. [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization](https://www.usenix.org/conference/osdi22/presentation/unger). In Proceedings of the Symposium on Operating Systems Design and Implementation (OSDI), July 2022. 

* Zhihao Jia, Matei Zaharia, and Alex Aiken. [Beyond Data and Model Parallelism for Deep Neural Networks](https://cs.stanford.edu/~zhihao/papers/sysml19a.pdf). In Proceedings of the 2nd Conference on Machine Learning and Systems (MLSys), April 2019.

* Zhihao Jia, Sina Lin, Charles R. Qi, and Alex Aiken. [Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks](http://proceedings.mlr.press/v80/jia18a/jia18a.pdf). In Proceedings of the International Conference on Machine Learning (ICML), July 2018.

## The Team
FlexFlow is developed and maintained by teams at CMU, Facebook, Los Alamos National Lab, MIT, and Stanford (alphabetically).

## License
FlexFlow uses Apache License 2.0.

