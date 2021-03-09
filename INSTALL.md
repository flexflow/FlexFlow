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
## 2.1 Makefile
### Build dependent libraries

* Build the NCCL library. (If using NCCL for parameter synchronization. )
```
cd nccl
make -j src.build NVCC_GENCODE="-gencode=arch=compute_XX,code=sm_XX"
```
Replace XX with the compatibility of your GPU devices (e.g., 70 for Volta GPUs and 60 for Pascal GPUs).

But you could also install it via `apt` if your system supports it:
```
sudo apt install libnccl-dev
```

### Build FlexFlow runtime with C++ interface
The `FF_HOME` environment variable is used for building and running FlexFlow. You can add the following line in `~/.bashrc`.
```
export FF_HOME=/path/to/FlexFlow
```
The path should point to where you cloned this repository.

Use the following command line to build a DNN model (e.g., InceptionV3). See the [examples](examples) folders for more existing FlexFlow applications.
```
./ffcompile.sh examples/InceptionV3
```

### Build FlexFlow Runtime with Python Interface (C++ interface is also enabled)

1. Set the following environment variables. For `CUDNN_HOME`, you should be able to find `cudnn.h` under `CUDNN_HOME/include` and `libcudnn.so` under `CUDNN_HOME/lib` or `CUDNN_HOME/lib64`.
```
export FF_HOME=/path/to/FlexFlow
export CUDNN_HOME=/path/to/cudnn
```
To expedite the compilation, you can also set the `GPU_ARCH` environment variable to be the compatibility of your GPU devices (e.g., 70 for Volta GPUs and 60 for Pascal GPUs).
```
export GPU_ARCH=your_gpu_arch
```
If you have different cards, pass them all via comma, e.g.:

```
export GPU_ARCH=70,86
```

If Legion can not automatically detect your Python installation, you need to tell Legion manually by setting the `PYTHON_EXE`, `PYTHON_LIB` and `PYTHON_VERSION_MAJOR`, please refer to the `python/Makefile` for details

2. Build the flexflow python executable using the following command line
```
cd python
make 
```

## 2.2 CMake

### Build the FlexFlow (including C++ and Python)
```
cd FlexFlow
cd config
```

The `config.linux` is an example of how to set the varibles required for CMake build. Please modify `FF_CUDA_ARCH`, `CUDNN_DIR`, `CUDA_DIR` according to your environment. `CUDA_DIR` is only required when CMake can not automatically detect the installation directory of CUDA. `CUDNN_DIR` is only required when CUDNN is not installed in the CUDA directory.

* `FF_CUDA_ARCH` is used to set the architecture of targeted GPUs, for example, the value can be 60 if the GPU architecture is Pascal. 
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
1. Set the `FF_HOME` environment variable before running the FlexFlow. You can add the following line in ~/.bashrc.
```
export FF_HOME=/path/to/FlexFlow
```

2. Run FlexFlow Python examples
The C++ examples are in the [examples/python](https://github.com/flexflow/FlexFlow/tree/master/examples/python). 
For example, the AlexNet can be run as:
```
cd python
./flexflow_python examples/python/native/alexnet.py -ll:py 1 -ll:gpu 1 -ll:fsize <size of gpu buffer> -ll:zsize <size of zero buffer>
``` 
The script of running all the Python examples is `python/test.sh`

3. Run FlexFlow C++ examples

The C++ examples are in the [examples/cpp](https://github.com/flexflow/FlexFlow/tree/master/examples/cpp). 
For example, the AlexNet can be run as:
```
./alexnet -ll:gpu 1 -ll:fsize <size of gpu buffer> -ll:zsize <size of zero buffer>
``` 

Size of buffers is in MBs, e.g. for an 8GB gpu `-ll:fsize 8000`

# 4. Install the FlexFlow

1. Install the FlexFlow binary, header file and library if using CMake. 
```
cd build
make install
```

2. Install the FlexFlow Python interface using pip
If install from local:
```
cd python
pip install .
```

If installing from the PyPI repository
```
pip install flexflow
```
All Python depencies will be automatically installed. 
