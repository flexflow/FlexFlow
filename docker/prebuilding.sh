#! /usr/bin/env bash
set -euo pipefail

# Parse input params
python_version=${1:-empty}
cuda_version=${2:-empty}
gpu_backend=${3:-empty}

# Install Conda and FlexFlow Dependencies

# Build Legion
export PATH=/opt/conda/bin:$PATH
export CUDNN_DIR=/usr/local/cuda
export CUDA_DIR=/usr/local/cuda
export FF_CUDA_ARCH=all
export BUILD_LEGION_ONLY=ON

# create and export the installation path
export INSTALL_DIR="export/legion"
mkdir -p export/legion

cores_available=$(nproc --all)
n_build_cores=$(( cores_available -1 ))

mkdir build
cd build
../config/config.linux
make -j $n_build_cores
../config/config.linux
make install

# prepare library files extract LEGION tarball file 
export LEGION_TARBALL="legion_ubuntu-20.04_${gpu_backend}.tar.gz"
echo "Creating archive $LEGION_TARBALL"
cd export
touch "$LEGION_TARBALL"
tar --exclude="$LEGION_TARBALL" -zcvf "$LEGION_TARBALL" .
echo "Checking the size of the Legion tarball..."
du -h "$LEGION_TARBALL"