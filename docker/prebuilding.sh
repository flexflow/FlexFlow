#! /usr/bin/env bash
set -euo pipefail

# Parse input params
python_version=${1:-empty}
cuda_version=${2:-empty}
gpu_backend=${3:-empty}

echo "$python_version"
echo "$cuda_version"
echo "$gpu_backend"

# Install Conda and FlexFlow Dependencies
apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git zlib1g-dev lsb-release nano libhdf5-dev && \
    rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential apt-utils \
    ca-certificates libssl-dev curl unzip htop && DEBIAN_FRONTEND=noninteractive \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:ubuntu-toolchain-r/test && \
        apt-get update -y && \
        apt-get upgrade -y libstdc++6

wget -c -q https://repo.continuum.io/miniconda/Miniconda3-"$python_version"-Linux-x86_64.sh && \
    mv Miniconda3-"$python_version"-Linux-x86_64.sh ~/Miniconda3-"$python_version"-Linux-x86_64.sh && \
    chmod +x ~/Miniconda3-"$python_version"-Linux-x86_64.sh && \
    bash ~/Miniconda3-"$python_version"-Linux-x86_64.sh -b -p /opt/conda && \
    rm ~/Miniconda3-"$python_version"-Linux-x86_64.sh && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya

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