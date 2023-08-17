# Legion Triton Backend

This directory contains an incomplete prototype for a new 
[backend for Triton](https://github.com/triton-inference-server/backend) built on top of the 
[Legion runtime](https://legion.stanford.edu) for handling multi-node multi-GPU inference
requests. While Legion is the primary runtime carrying out multi-node inference jobs, users
do not need to understand Legion at all to use this backend.  

## Build instructions

### CMake

A simple CMake is provided to build Legion backend and to resolve its dependencies.
Note that the build will install protobuf with customized settting, please make sure
that the system doesn't have protobuf installed to avoid conflict.

```
$ mkdir build
$ cd build
$ cmake  ..
$ make
```

After build, the backend shared library can be found at `/PATH/TO/BUILDDIR/triton-legion/backends/legion`

By default, the unit tests and test data are installed at `/PATH/TO/BUILDDIR/triton-legion/test`,
which can be run after switching the current directory to the installed location.

### Make

Protobuf is required for the backend and it must be installed from source with the following command
to build the static protobuf library that can be linked with the backend shared library

```
git clone https://github.com/protocolbuffers/protobuf.git
git checkout v3.17.1
cd protobuf/cmake
cmake -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON -Dprotobuf_BUILD_TESTS:BOOL=OFF -Dprotobuf_WITH_ZLIB:BOOL=OFF -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF -DCMAKE_BUILD_TYPE:STRING=RELEASE -DBUILD_SHARED_LIBS:STRING=no .
make install
```

Set the `LG_RT_DIR` environment variable to point to the `legion/runtime` directory in a Legion repo

Set the `TRITON_DIR` to point to an installation of the Triton server

Go into the `src` directory and type `make`

Copy the `libtriton_flexflow.so` shared object to a triton model repository
