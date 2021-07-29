# FlexFlow Installation
FlexFlow can be built from source code using the following instructions.

# 1. Download the source code
* Clone the FlexFlow source code from the github.
```
git clone --recursive https://github.com/flexflow/FlexFlow.git
```

## FlexFlow Python dependencies
* The FlexFlow Python support requires several additional Python libraries, please check [this](https://github.com/flexflow/FlexFlow/blob/master/python/requirements.txt) for details. 
We recommend to use `pip` or `conda` to install the dependencies. 

Note: all Python dependencies will be automatically installed if install the FlexFlow Python Interface using the PyPi repository (see the Installation below).

# 2. Build the FlexFlow

It is prefer to use the CMake building system as it will automatically build all dependencies inlcuding NCCL and Legion. 

### Build the FlexFlow (including C++ and Python)

The `config/config.linux` is an example of how to set the varibles required for CMake build. The followings are configurable parameters:

* `CUDA_DIR` is used to specify the directory of CUDA. It is only required when CMake can not automatically detect the installation directory of CUDA.
* `CUDNN_DIR` is only required when CUDNN is not installed in the CUDA directory.
* `FF_CUDA_ARCH` is used to set the architecture of targeted GPUs, for example, the value can be 60 if the GPU architecture is Pascal. If it is not sepecified, FlexFlow is compiled for all architectures.
* `FF_USE_PYTHON` is used to enable the Python support for the FlexFlow.
* `FF_USE_NCCL` is used to enable the NCCL support for the FlexFlow, by default it is set to ON.
* `FF_USE_GASNET` is used to enable distributed run of the FlexFlow.
* `FF_BUILD_EXAMPLES` is used to enable all C++ examples.
* `FF_MAX_DIM` is used to set the maximum dimension of tensors, by default it is set to 4. 

More options are available in cmake, please run ccmake and search for options starting with FF. 

Once the variables in the `config.linux` is set correctly, go to the home directory of FlexFlow, and run
```
mkdir build
cd build
../config/config.linux
make
```

# 3. Test the FlexFlow
## 1. Set the `FF_HOME` environment variable before running the FlexFlow. You can add the following line in ~/.bashrc.
```
export FF_HOME=/path/to/FlexFlow
```

## 2. Run FlexFlow Python examples
The C++ examples are in the [examples/python](https://github.com/flexflow/FlexFlow/tree/master/examples/python). 
For example, the AlexNet can be run as:
```
cd python
./flexflow_python examples/python/native/alexnet.py -ll:py 1 -ll:gpu 1 -ll:fsize <size of gpu buffer> -ll:zsize <size of zero buffer>
``` 
The script of running all the Python examples is `python/test.sh`

## 3. Run FlexFlow C++ examples

The C++ examples are in the [examples/cpp](https://github.com/flexflow/FlexFlow/tree/master/examples/cpp). 
For example, the AlexNet can be run as:
```
./alexnet -ll:gpu 1 -ll:fsize <size of gpu buffer> -ll:zsize <size of zero buffer>
``` 

Size of buffers is in MBs, e.g. for an 8GB gpu `-ll:fsize 8000`

# 4. Install the FlexFlow

## 1. Install the FlexFlow binary, header file and library if using CMake. 
```
cd build
make install
```

## 2. Install the FlexFlow Python interface using pip
If install from local:
```
pip install .
```
All Python depencies will be automatically installed. 
The default installation is configurated with `FF_USE_NCCL=OFF` `FF_USE_GASNET=OFF`and `FF_MAX_DIM=4`.
In order to change the configuration, please use the `--global-option` of the setuptools
```
pip install --global-option="build_ext" --global-option="CMake options" . 
```
