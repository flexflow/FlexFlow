#!/bin/bash
set -euo pipefail
set -x

echo "Installing PyTorch dependencies"
rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends software-properties-common build-essential apt-utils ca-certificates libssl-dev curl \
    unzip htop nano
sudo add-apt-repository ppa:ubuntu-toolchain-r/test && sudo apt-get update -y && sudo apt-get upgrade -y libstdc++6

# Install CPU-only Pytorch and related packages
echo "Installing PyTorch and related packages"
export PATH=/opt/conda/bin:$PATH
# Install CPU-only Pytorch
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch
# Install Hugging-face/MT5 related packages
conda install -c conda-forge pandas numpy transformers=4.16.2 sentencepiece
# Install packages required by other example applications
pip install onnx tensorflow keras2onnx
