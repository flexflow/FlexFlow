# FlexFlow Installation
FlexFlow can be built from source code using the following instructions.

## Build FlexFlow Runtime

* To get started, clone the FlexFlow source code from github.
```
git clone --recursive https://github.com/flexflow/FlexFlow.git
cd FlexFlow
git submodule init
git submodule update -r
```
The `FF_HOME` environment variable is used for building and running FlexFlow. You can add the following line in `~/.bashrc`.
```
export FF_HOME=/path/to/FlexFlow
```

* Build the Protocol Buffer library.
Skip this step if the Protocol Buffer library is already installed.
```
cd protobuf
./autogen.sh
./configure
make
```
* Build the NCCL library. (If using NCCL for parameter synchornization.)
```
cd nccl
make -j src.build NVCC_GENCODE="-gencode=arch=compute_XX,code=sm_XX"
```
Replace XX with the compatability of your GPU devices (e.g., 70 for Volta GPUs and 60 for Pascal GPUs).

* Build a DNN model in FlexFlow
Use the following command line to build a DNN model (e.g., InceptionV3). See the [examples](examples) folders for more existing FlexFlow applications.
```
./ffcompile.sh examples/InceptionV3
```

## Build FlexFlow Python Interface

* Get the FlexFlow source code using the same instruction as the C++ interface

* Set the following enviroment variables
```
export FF_HOME=/path/to/FlexFlow
export CUDNN_DIR=/path/to/cudnn
export CUDA_DIR=/path/to/cuda
export PROTOBUF_DIR=/path/to/protobuf
export LG_RT_DIR=/path/to/Legion
```
To expedite the compilation, you can also set the `GPU_ARCH` enviroment variable
```
export GPU_ARCH=your_gpu_arch
``` 
If Legion can not automatically detect your Python installation, you need to tell Legion manually by setting the `PYTHON_EXE`, `PYTHON_LIB` and `PYTHON_VERSION_MAJOR`, please refer to the `python/Makefile` for details

* Build the flexflow python executable using the following command line
```
cd python
make 
```

* To run a DNN model, use the following command line
```
./flexflow_python example/xxx.py -ll:py 1 -ll:gpu 1 -ll:fsize size of gpu buffer -ll:zsize size of zero buffer
``` 