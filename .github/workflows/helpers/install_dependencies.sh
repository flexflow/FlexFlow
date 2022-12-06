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

# Install conda packages
echo "Installing conda packages..."
/opt/conda/bin/conda install cmake make pillow
/opt/conda/bin/conda install -c conda-forge numpy keras-preprocessing pybind11 cmake-build-extension
