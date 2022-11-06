#!/bin/bash
set -euo pipefail
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# General dependencies
echo "Installing apt dependencies..."
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    wget binutils git nano zlib1g-dev libhdf5-dev software-properties-common \
    build-essential apt-utils ca-certificates libssl-dev curl unzip htop && \
    sudo rm -rf /var/lib/apt/lists/*

# Install CUDNN
./install_cudnn.sh

# Install Miniconda
echo "Installing Miniconda..."
wget -c -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ./Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya

# Install conda packages
echo "Installing conda packages..."
export PATH=/opt/conda/bin:$PATH
conda install -c conda-forge cmake make cmake-build-extension pybind11 numpy pandas keras-preprocessing 
conda install pytorch==1.9.0 torchvision==0.10.1 torchaudio==0.9.1 cpuonly -c pytorch
conda install -c conda-forge pillow transformers sentencepiece onnx tensorflow
    
