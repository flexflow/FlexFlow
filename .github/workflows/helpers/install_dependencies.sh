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

# Install conda packages
echo "Installing conda packages..."
/opt/conda/bin/conda install cmake make pillow
/opt/conda/bin/conda install -c conda-forge numpy keras-preprocessing pybind11 cmake-build-extension

DEBIAN_FRONTEND=noninteractive

# Install basic packages
rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends \
        software-properties-common && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        ca-certificates \
        libssl-dev \
        curl \
        unzip \
        htop \
        nano && \
    DEBIAN_FRONTEND=noninteractive \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:ubuntu-toolchain-r/test && \
        apt-get update -y && \
        apt-get upgrade -y libstdc++6

# Install CPU-only Pytorch
/opt/conda/bin/conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch
# Install Hugging-face/MT5 related packages
/opt/conda/bin/conda install -c conda-forge pandas numpy transformers=4.16.2 sentencepiece
# Install packages required by other example applications
pip install onnx tensorflow keras2onnx

