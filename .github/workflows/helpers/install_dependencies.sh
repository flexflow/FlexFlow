#!/bin/bash

# General dependencies
echo "Installing apt dependencies..."
sudo apt-get update && sudo apt-get install -y --no-install-recommends wget binutils git zlib1g-dev && \
    sudo rm -rf /var/lib/apt/lists/*

# Install CUDNN
echo "Installing CUDNN..."
wget -c -q http://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.1-linux-x64-v8.0.5.39.tgz && \
    sudo tar -xzf cudnn-11.1-linux-x64-v8.0.5.39.tgz -C /usr/local && \
    rm cudnn-11.1-linux-x64-v8.0.5.39.tgz && \
    sudo ldconfig

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
if [ "$FF_GPU_BACKEND" = "hip_cuda" ] || [ "$FF_GPU_BACKEND" = "hip_rocm" ]; then \
    echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"; \
    wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/bionic/amdgpu-install_22.20.50200-1_all.deb; \
    sudo apt-get install -y ./amdgpu-install_22.20.50200-1_all.deb; \
    rm ./amdgpu-install_22.20.50200-1_all.deb; \
    sudo amdgpu-install -y --usecase=hip,rocm --no-dkms; \
    sudo apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk; \
else \
    echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"; \
fi
sudo rm -rf /var/lib/apt/lists/*

# Install conda packages
echo "Installing conda packages..."
/opt/conda/bin/conda install cmake make pillow
/opt/conda/bin/conda install -c conda-forge numpy keras-preprocessing pybind11 cmake-build-extension

# Examples
sudo apt-get update -y && sudo apt-get install -y --no-install-recommends libhdf5-dev
