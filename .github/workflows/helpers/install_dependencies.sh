#!/bin/bash
set -euo pipefail
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# General dependencies
echo "Installing apt dependencies..."
sudo apt-get update && sudo apt-get install -y --no-install-recommends wget binutils git zlib1g-dev libhdf5-dev && \
    sudo rm -rf /var/lib/apt/lists/*

# Install CUDNN
./install_cudnn.sh

# Install HIP dependencies if needed
FF_GPU_BACKEND=${FF_GPU_BACKEND:-"cuda"}
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid."
  exit 1
elif [[ "$FF_GPU_BACKEND" == "hip_cuda" || "$FF_GPU_BACKEND" = "hip_rocm" ]]; then
    echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"
    wget https://repo.radeon.com/amdgpu-install/22.20.5/ubuntu/focal/amdgpu-install_22.20.50205-1_all.deb
    sudo apt-get install -y ./amdgpu-install_22.20.50205-1_all.deb
    rm ./amdgpu-install_22.20.50205-1_all.deb
    sudo amdgpu-install -y --usecase=hip,rocm --no-dkms
    sudo apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk
else
    echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"
fi
sudo rm -rf /var/lib/apt/lists/*
