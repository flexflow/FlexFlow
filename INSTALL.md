# Installing FlexFlow
To build and install FlexFlow, follow the instructions below.

## 1. Download the source code
Clone the FlexFlow source code, and the third-party dependencies from GitHub.
```
git clone --recursive https://github.com/flexflow/FlexFlow.git
```

## 2. Install the Python dependencies
If you are planning to build the Python interface, you will need to install several additional Python libraries, please check [this](https://github.com/flexflow/FlexFlow/blob/master/requirements.txt) for details. If you are only looking to use the C++ interface, you can skip to the next section.

**We recommend that you create your own `conda` environment and then install the Python dependencies, to avoid any version mismatching with your system pre-installed libraries.** 

## 3. Configuring the FlexFlow build
Before building FlexFlow, you should configure the build by editing the `config/config.linux` file. Leave it unchanged if you want to build with the default options. We recommend that you spend some time familiarizing with the available options. In particular, the main parameters are:
* `CUDA_DIR` is used to specify the directory of CUDA. It is only required when CMake can not automatically detect the installation directory of CUDA.
* `CUDNN_DIR` is used to specify the directory of CUDNN. It is only required when CUDNN is not installed in the CUDA directory.
* `FF_CUDA_ARCH` is used to set the architecture of targeted GPUs, for example, the value can be 60 if the GPU architecture is Pascal. If it is not sepecified, FlexFlow is compiled for all architectures that are detecte on the machine. **If your machine does not have any GPU, you have to set FF_CUDA_ARCH to at least one valid architecture code**, since the compiler won't be able to detect the architecture(s) automatically.
* `FF_USE_PYTHON` controls whether to build the FlexFlow Python interface.
* `FF_USE_NCCL` controls whether to build FlexFlow with NCCL support. By default, it is set to ON.
* `FF_USE_GASNET` is used to enable distributed run of FlexFlow.
* `FF_BUILD_EXAMPLES` controls whether to build all C++ example programs.
* `FF_MAX_DIM` is used to set the maximum dimension of tensors, by default it is set to 4. 

More options are available in cmake, please run `ccmake` and search for options starting with FF. 

## 4. Build FlexFlow
You can build FlexFlow in three ways: with CMake, with Make, and with `pip`. We recommend that you use the CMake building system as it will automatically build all C++ dependencies inlcuding NCCL and Legion. 

### Building FlexFlow with CMake
To build FlexFlow with CMake, go to the FlexFlow home directory, and run
```
mkdir build
cd build
../config/configure.sh
make -j N
```
where N is the desired number of threads to use for the build.

### Building FlexFlow with pip
To build Flexflow with `pip`, run `pip install .` from the FlexFlow home directory. This command will build FlexFlow, and also install the Python interface as a Python module.

### Building FlexFlow with Make
The Makefile we provide is mainly for development purpose, and may not be fully up to date. 


## 5. Test FlexFlow
After building FlexFlow, you can test it to ensure that the build completed without issue, and that your system is ready to run FlexFlow.

### Set the `FF_HOME` environment variable before running FlexFlow. To make it permanent, you can add the following line in ~/.bashrc.
```
export FF_HOME=/path/to/FlexFlow
```

### Run FlexFlow Python examples
The Python examples are in the [examples/python](https://github.com/flexflow/FlexFlow/tree/master/examples/python). 
The native, Keras integration and PyTorch integration examples are listed in `native`, `keras` and `pytorch` respectively.

**We recommend that you run the `mnist_mlp` test under `native` using the following cmd to check if FlexFlow has been installed correctly.**

**Please use our python interpreter `flexflow_python` instead of the native one**
```
cd python
./flexflow_python examples/python/native/mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize <size of gpu buffer> -ll:zsize <size of zero buffer>
``` 
The script of running all the Python examples is `python/test.sh`

### Run FlexFlow C++ examples

The C++ examples are in the [examples/cpp](https://github.com/flexflow/FlexFlow/tree/master/examples/cpp). 
For example, the AlexNet can be run as:
```
./alexnet -ll:gpu 1 -ll:fsize <size of gpu buffer> -ll:zsize <size of zero buffer>
``` 

Size of buffers is in MBs, e.g. for an 8GB gpu `-ll:fsize 8000`

## 6. Install FlexFlow
If you built/installed FlexFlow using `pip`, this step is not required. If you built using Make or CMake, install FlexFlow with:
```
cd build
make install
```
