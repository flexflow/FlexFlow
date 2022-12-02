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

# Install Miniconda
echo "Installing Miniconda..."
wget -c -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ./Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya

# Install HIP dependencies if needed
FF_GPU_BACKEND=${FF_GPU_BACKEND:-"cuda"}
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid."
  exit 1
elif [[ "$FF_GPU_BACKEND" == "hip_cuda" || "$FF_GPU_BACKEND" = "hip_rocm" ]]; then
    echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"
    # Get latest version of amdgpu-install script. We have to do this because new versions are rolled out
    # relatively often, and previous versions are removed, so only the latest version is available.
    # Similar approach as: https://stackoverflow.com/questions/22510705/get-the-latest-download-link-programmatically
    latest_version=$(curl 'http://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/' | \
    grep -oP 'href="amdgpu-install_\K[0-9]+\.[0-9]+\.[0-9]+\-[0-9]+' | \
    sort -t. -rn -k1,1 -k2,2 -k3,3 | head -1)
    script_name="amdgpu-install_${latest_version}_all.deb"
    script_url="https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/${script_name}"
    eval wget "$script_url"
    sudo apt-get install -y "./${script_name}"
    rm "./${script_name}"
    sudo amdgpu-install -y --usecase=hip,rocm --no-dkms
    sudo apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk
else
    echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"
fi
sudo rm -rf /var/lib/apt/lists/*

# Install conda packages
echo "Installing conda packages..."
/opt/conda/bin/conda install cmake make pillow
/opt/conda/bin/conda install -c conda-forge numpy keras-preprocessing pybind11 cmake-build-extension
