# FlexFlow Installation
FlexFlow can be built from source code using the following instructions.

# Download the FlexFlow source code and its dependencies
* FlexFlow requires Legion, Protocol Buffer, NCCL (optionally) and GASNet (optinally), which are in the FlexFlow third party libraries repo.
To get started, clone them from the github.
```
git clone --recursive https://github.com/flexflow/flexflow-third-party.git
```

* Clone the FlexFlow source code from the github.
```
git clone https://github.com/flexflow/FlexFlow.git
```

# Build the FlexFlow
## Makefile
### Build FlexFlow Runtime

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

The `FF_HOME` environment variable is used for building and running FlexFlow. You can add the following line in `~/.bashrc`.
```
export FF_HOME=/path/to/FlexFlow
```
Use the following command line to build a DNN model (e.g., InceptionV3). See the [examples](examples) folders for more existing FlexFlow applications.
```
./ffcompile.sh examples/InceptionV3
```

### Build FlexFlow Python Interface

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

## CMake

* Build the FlexFlow third party libraries.
```
cd flexflow-third-party
mkdir build
cd build
cmake ../ -DCUDA_ARCH=x0 -DPYTHON_VERSION=3.x (replace the x with the corrected number)
make
make install
```
Note: CMake sometimes can not automatically detect the correct `CUDA_ARCH`, so please set `CUDA_ARCH` if CMake can not detect it. 

* Build the FlexFlow
```
cd FlexFlow
cd config
```

The `config.linux` is an example of how to set the varibles required for CMake build. Please modify `CUDA_ARCH`, `CUDNN_DIR`, `LEGION_DIR` and `PROTOBUF_DIR` according to your environment.  `LEGION_DIR` and `PROTOBUF_DIR` are the installation directories of Legion and Protocol Buffer, not the source code directories.
Then go to the home directory of FlexFlow, and run
```
mkdir build
cd build
../config/config.linux
make
```

After the compilation is done, the flexflow python can be run using the following command lines:
```
cd python
./flexflow_python example/xxx.py -ll:py 1 -ll:gpu 1 -ll:fsize size of gpu buffer -ll:zsize size of zero buffer
``` 

Make sure The FF_HOME environment variable is set before running FlexFlow. You can add the following line in ~/.bashrc.
```
export FF_HOME=/path/to/FlexFlow
```

* Install the FlexFlow

Install the FlexFlow binary, header file and library. 
```
cd build
make install
```

Install the FlexFlow python code using pip
```
cd python
pip install .
```
