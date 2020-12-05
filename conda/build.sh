# build flexflow
export CXXFLAGS=$(echo $CXXFLAGS | sed 's/-O2//g')
export CXXFLAGS=$(echo $CXXFLAGS | sed 's/-std=c++17//g')
export CXXFLAGS=$(echo $CXXFLAGS | sed 's/-DNDEBUG//g')
export CXXFLAGS=$(echo $CXXFLAGS | sed 's/-D_FORTIFY_SOURCE=2//g')
export CPPFLAGS=$(echo $CPPFLAGS | sed 's/-O2//g')
export CPPFLAGS=$(echo $CPPFLAGS | sed 's/std=c++17//g')
export CPPFLAGS=$(echo $CXXFLAGS | sed 's/-DNDEBUG//g')
export CPPFLAGS=$(echo $CXXFLAGS | sed 's/-D_FORTIFY_SOURCE=2//g')

#export CUDNN_HOME=/projects/opt/centos7/cuda/10.1
#export CUDA_HOME=/projects/opt/centos7/cuda/10.1
export PROTOBUF_DIR=$BUILD_PREFIX
export FF_HOME=$SRC_DIR
export LG_RT_DIR=$SRC_DIR/legion/runtime
#export FF_ENABLE_DEBUG=1
#export DEBUG=0

cd python
make
