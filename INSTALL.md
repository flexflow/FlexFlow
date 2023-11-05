# Building from source
To build and install FlexFlow, follow the instructions below.

## 1. Download the source code
Clone the FlexFlow source code, and the third-party dependencies from GitHub.
```
git clone --recursive https://github.com/flexflow/FlexFlow.git
```

## 2. Install system dependencies
FlexFlow has system dependencies on cuda and/or rocm depending on which gpu backend you target. The gpu backend is configured by the cmake variable `FF_GPU_BACKEND`. By default, FlexFlow targets CUDA. `docker/base/Dockerfile` installs system dependencies in a standard ubuntu system.

### Targeting CUDA - `FF_GPU_BACKEND=cuda`
If you are targeting CUDA, FlexFlow requires CUDA and CUDNN to be installed. You can follow the standard nvidia installation instructions [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CUDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

Disclaimer: CUDA architectures < 60 (Maxwell and older) are no longer supported.

### Targeting ROCM - `FF_GPU_BACKEND=hip_rocm`
If you are targeting ROCM, FlexFlow requires a ROCM and HIP installation with a few additional packages. Note that this can be done on a system with or without an AMD GPU. You can follow the standard installation instructions [ROCM](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.3/page/Introduction_to_ROCm_Installation_Guide_for_Linux.html) and [HIP](https://docs.amd.com/bundle/HIP-Installation-Guide-v5.3/page/Introduction_to_HIP_Installation_Guide.html). When running `amdgpu-install`, install the use cases hip and rocm. You can avoid installing the kernel drivers (not necessary on systems without an AMD graphics card) with `--no-dkms` I.e. `amdgpu-install --usecase=hip,rocm --no-dkms`. Additionally, install the packages `hip-dev`, `hipblas`, `miopen-hip`, and `rocm-hip-sdk`.

See `./docker/base/Dockerfile` for an example ROCM install.

### Targeting CUDA through HIP - `FF_GPU_BACKEND=hip_cuda`
This is not currently supported.

## 3. Install the Python dependencies
If you are planning to build the Python interface, you will need to install several additional Python libraries, please check [this](https://github.com/flexflow/FlexFlow/blob/master/requirements.txt) for details. If you are only looking to use the C++ interface, you can skip to the next section.

**We recommend that you create your own `conda` environment and then install the Python dependencies, to avoid any version mismatching with your system pre-installed libraries.** 

The `conda` environment can be created and activated as:
```
conda env create -f conda/flexflow.yml
conda activate flexflow
```

## 4. Configuring the FlexFlow build
You can configure a FlexFlow build by running the `config/config.linux` file in the build folder. If you do not want to build with the default options, you can set your configurations by passing (or exporting) the relevant environment variables. We recommend that you spend some time familiarizing with the available options by scanning the `config/config.linux` file. In particular, the main parameters are:

1. `CUDA_DIR` is used to specify the directory of CUDA. It is only required when CMake can not automatically detect the installation directory of CUDA.
2. `CUDNN_DIR` is used to specify the directory of CUDNN. It is only required when CUDNN is not installed in the CUDA directory.
3. `FF_CUDA_ARCH` is used to set the architecture of targeted GPUs, for example, the value can be 60 if the GPU architecture is Pascal. To build for more than one architecture, pass a list of comma separated values (e.g. `FF_CUDA_ARCH=70,75`). To compile FlexFlow for all GPU architectures that are detected on the machine, pass `FF_CUDA_ARCH=autodetect` (this is the default value, so you can also leave `FF_CUDA_ARCH` unset. If you want to build for all GPU architectures compatible with FlexFlow, pass `FF_CUDA_ARCH=all`. **If your machine does not have any GPU, you have to set FF_CUDA_ARCH to at least one valid architecture code (or `all`)**, since the compiler won't be able to detect the architecture(s) automatically.
4. `FF_USE_PYTHON` controls whether to build the FlexFlow Python interface.
5. `FF_USE_NCCL` controls whether to build FlexFlow with NCCL support. By default, it is set to ON.
6. `FF_LEGION_NETWORKS` is used to enable distributed run of FlexFlow. If you want to run FlexFlow on multiple nodes, follow instructions in the [Multinode tutorial](https://flexflow.readthedocs.io/en/latest/multinode.html) and set the corresponding parameters as follows:
* To build FlexFlow with GASNet, set `FF_LEGION_NETWORKS=gasnet` and `FF_GASNET_CONDUIT` as a specific conduit (e.g. `ibv`, `mpi`, `udp`, `ucx`) in `config/config.linux` when configuring the FlexFlow build. Set `FF_UCX_URL` when you want to customize the URL to download UCX.
* To build FlexFlow with native UCX, set `FF_LEGION_NETWORKS=ucx` in `config/config.linux` when configuring the FlexFlow build. Set `FF_UCX_URL` when you want to customize the URL to download UCX.
8. `FF_BUILD_EXAMPLES` controls whether to build all C++ example programs.
9. `FF_MAX_DIM` is used to set the maximum dimension of tensors, by default it is set to 4. 
10. `FF_USE_{NCCL,LEGION,ALL}_PRECOMPILED_LIBRARY`, controls whether to build FlexFlow using a pre-compiled version of the Legion, NCCL (if `FF_USE_NCCL` is `ON`), or both libraries . By default, `FF_USE_NCCL_PRECOMPILED_LIBRARY` and `FF_USE_LEGION_PRECOMPILED_LIBRARY` are both set to `ON`, allowing you to build FlexFlow faster. If you want to build Legion and NCCL from source, set them to `OFF`.

More options are available in cmake, please run `ccmake` and search for options starting with FF. 

## 5. Build FlexFlow
You can build FlexFlow in three ways: with CMake, with Make, and with `pip`. We recommend that you use the CMake building system as it will automatically build all C++ dependencies inlcuding NCCL and Legion. 

### Building FlexFlow with CMake
To build FlexFlow with CMake, go to the FlexFlow home directory, and run
```
mkdir build
cd build
../config/config.linux
make -j N
```
where N is the desired number of threads to use for the build.

### Building FlexFlow with pip
To build Flexflow with `pip`, run `pip install .` from the FlexFlow home directory. This command will build FlexFlow, and also install the Python interface as a Python module.

### Building FlexFlow with Make
The Makefile we provide is mainly for development purposes, and may not be fully up to date. To use it, run:
```
cd python
make -j N
```

## 6. Test FlexFlow
After building FlexFlow, you can test it to ensure that the build completed without issue, and that your system is ready to run FlexFlow.

### Set the `FF_HOME` environment variable before running FlexFlow. To make it permanent, you can add the following line in ~/.bashrc.
```
export FF_HOME=/path/to/FlexFlow
```

### Run FlexFlow Python examples
The Python examples are in the [examples/python](https://github.com/flexflow/FlexFlow/tree/master/examples/python). The native, Keras integration and PyTorch integration examples are listed in `native`, `keras` and `pytorch` respectively.

To run the Python examples, you have two options: you can use the `flexflow_python` interpreter, available in the `build` folder, or you can use the native Python interpreter. If you choose to use the native Python interpreter, you should either install FlexFlow, or, if you prefer to build without installing, export the required environment flags by running the following command (edit the path if your build folder is not named `build`):

```
source ./build/set_python_envs.sh
```

**We recommend that you run the** `mnist_mlp` **test under** `native` **using the following cmd to check if FlexFlow has been installed correctly:**

```
cd "$FF_HOME"
./python/flexflow_python examples/python/native/mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize <size of gpu buffer> -ll:zsize <size of zero buffer>
```
A script to run all the Python examples is available at `tests/training_tests.sh`

### Run FlexFlow C++ examples

The C++ examples are in the [examples/cpp](https://github.com/flexflow/FlexFlow/tree/master/examples/cpp). 
For example, the AlexNet can be run as:
```
./alexnet -ll:gpu 1 -ll:fsize <size of gpu buffer> -ll:zsize <size of zero buffer>
``` 

Size of buffers is in MBs, e.g. for an 8GB gpu `-ll:fsize 8000`

## 7. Install FlexFlow
If you built/installed FlexFlow using `pip`, this step is not required. If you built using Make or CMake, install FlexFlow with:
```
cd build
make install
```
