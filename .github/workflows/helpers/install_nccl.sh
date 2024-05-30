#!/bin/bash
set -euo pipefail
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Add NCCL key ring
ubuntu_version=$(lsb_release -rs)
ubuntu_version=${ubuntu_version//./}
wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu_version}/x86_64/cuda-keyring_1.0-1_all.deb"
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update -y
rm -f cuda-keyring_1.0-1_all.deb

# Install NCCL
cuda_version=${1:-11.8.0}
cuda_version=$(echo "${cuda_version}" | cut -f1,2 -d'.')
echo "Installing NCCL for CUDA version: ${cuda_version} ..."

# We need to run a different install command based on the CUDA version, otherwise running `sudo apt install libnccl2 libnccl-dev`
# will automatically upgrade CUDA to the latest version.

if [[ "$cuda_version" == "11.0" ]]; then
    sudo apt install libnccl2=2.15.5-1+cuda11.0 libnccl-dev=2.15.5-1+cuda11.0
elif [[ "$cuda_version" == "11.1" ]]; then
    sudo apt install libnccl2=2.8.4-1+cuda11.1 libnccl-dev=2.8.4-1+cuda11.1
elif [[ "$cuda_version" == "11.2" ]]; then
    sudo apt install libnccl2=2.8.4-1+cuda11.2 libnccl-dev=2.8.4-1+cuda11.2
elif [[ "$cuda_version" == "11.3" ]]; then
    sudo apt install libnccl2=2.9.9-1+cuda11.3 libnccl-dev=2.9.9-1+cuda11.3
elif [[ "$cuda_version" == "11.4" ]]; then
    sudo apt install libnccl2=2.11.4-1+cuda11.4 libnccl-dev=2.11.4-1+cuda11.4
elif [[ "$cuda_version" == "11.5" ]]; then
    sudo apt install libnccl2=2.11.4-1+cuda11.5 libnccl-dev=2.11.4-1+cuda11.5
elif [[ "$cuda_version" == "11.6" ]]; then
    sudo apt install libnccl2=2.12.12-1+cuda11.6 libnccl-dev=2.12.12-1+cuda11.6
elif [[ "$cuda_version" == "11.7" ]]; then
    sudo apt install libnccl2=2.14.3-1+cuda11.7 libnccl-dev=2.14.3-1+cuda11.7
elif [[ "$cuda_version" == "11.8" ]]; then
    sudo apt install libnccl2=2.16.5-1+cuda11.8 libnccl-dev=2.16.5-1+cuda11.8
elif [[ "$cuda_version" == "12.0" ]]; then
    sudo apt install libnccl2=2.18.3-1+cuda12.0 libnccl-dev=2.18.3-1+cuda12.0
elif [[ "$cuda_version" == "12.1" ]]; then
    sudo apt install libnccl2=2.18.3-1+cuda12.1 libnccl-dev=2.18.3-1+cuda12.1
elif [[ "$cuda_version" == "12.2" ]]; then
    sudo apt install libnccl2=2.18.3-1+cuda12.2 libnccl-dev=2.18.3-1+cuda12.2
else
    echo "Installing NCCL for CUDA version ${cuda_version} is not supported"
    exit 1
fi
