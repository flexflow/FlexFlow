#!/bin/bash
set -euo pipefail
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# General dependencies
echo "Installing apt dependencies..."
sudo apt-get update && sudo apt-get install -y --no-install-recommends wget binutils git zlib1g-dev libhdf5-dev jq && \
    sudo rm -rf /var/lib/apt/lists/*

FF_GPU_BACKEND=${FF_GPU_BACKEND:-"cuda"}
hip_version=${hip_version:-"5.6"}
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid."
  exit 1
fi
# Install CUDNN if needed
if [[ "$FF_GPU_BACKEND" == "cuda" || "$FF_GPU_BACKEND" = "hip_cuda" ]]; then
    # Install CUDNN
    ./install_cudnn.sh
    # Install NCCL
    ./install_nccl.sh
fi
# Install HIP dependencies if needed
if [[ "$FF_GPU_BACKEND" == "hip_cuda" || "$FF_GPU_BACKEND" = "hip_rocm" ]]; then
    echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"
    # Check that hip_version is one of 5.3,5.4,5.5,5.6
    if [[ "$hip_version" != "5.3" && "$hip_version" != "5.4" && "$hip_version" != "5.5" && "$hip_version" != "5.6" ]]; then
        echo "hip_version '${hip_version}' is not supported, please choose among {5.3, 5.4, 5.5, 5.6}"
        exit 1
    fi
    # Compute script name and url given the version
    AMD_GPU_SCRIPT_NAME=amdgpu-install_5.6.50600-1_all.deb
    if [ "$hip_version" = "5.3" ]; then
        AMD_GPU_SCRIPT_NAME=amdgpu-install_5.3.50300-1_all.deb
    elif [ "$hip_version" = "5.4" ]; then
        AMD_GPU_SCRIPT_NAME=amdgpu-install_5.4.50400-1_all.deb
    elif [ "$hip_version" = "5.5" ]; then
        AMD_GPU_SCRIPT_NAME=amdgpu-install_5.5.50500-1_all.deb
    fi
    AMD_GPU_SCRIPT_URL="https://repo.radeon.com/amdgpu-install/${hip_version}/ubuntu/focal/${AMD_GPU_SCRIPT_NAME}"
    # Download and install AMD GPU software with ROCM and HIP support
    wget "$AMD_GPU_SCRIPT_URL"
    sudo apt-get install -y ./${AMD_GPU_SCRIPT_NAME}
    sudo rm ./${AMD_GPU_SCRIPT_NAME}
    sudo amdgpu-install -y --usecase=hip,rocm --no-dkms
    sudo apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk rocm-device-libs

    # Install protobuf v3.20.x manually
    sudo apt-get update -y && sudo apt-get install -y pkg-config zip g++ zlib1g-dev unzip python autoconf automake libtool curl make
    git clone -b 3.20.x https://github.com/protocolbuffers/protobuf.git
    cd protobuf/
    git submodule update --init --recursive
    ./autogen.sh
    ./configure
    cores_available=$(nproc --all)
    n_build_cores=$(( cores_available -1 ))
    if (( n_build_cores < 1 )) ; then n_build_cores=1 ; fi
    make -j $n_build_cores
    sudo make install
    sudo ldconfig
    cd ..
else
    echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"
fi
sudo rm -rf /var/lib/apt/lists/*
