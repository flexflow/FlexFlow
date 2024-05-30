#!/bin/bash
set -euo pipefail
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Install CUDNN
cuda_version=${1:-11.8.0}
cuda_version=$(echo "${cuda_version}" | cut -f1,2 -d'.')
echo "Installing CUDNN for CUDA version: ${cuda_version} ..."
CUDNN_LINK=http://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.1-linux-x64-v8.0.5.39.tgz
CUDNN_TARBALL_NAME=cudnn-11.1-linux-x64-v8.0.5.39.tgz
if [[ "$cuda_version" == "10.1" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-10.1-linux-x64-v8.0.5.39.tgz
    CUDNN_TARBALL_NAME=cudnn-10.1-linux-x64-v8.0.5.39.tgz
elif [[ "$cuda_version" == "10.2" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-10.2-linux-x64-v8.0.5.39.tgz
    CUDNN_TARBALL_NAME=cudnn-10.2-linux-x64-v8.0.5.39.tgz
elif [[ "$cuda_version" == "11.0" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.0-linux-x64-v8.0.5.39.tgz
    CUDNN_TARBALL_NAME=cudnn-11.0-linux-x64-v8.0.5.39.tgz
elif [[ "$cuda_version" == "11.1" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.1-linux-x64-v8.0.5.39.tgz
    CUDNN_TARBALL_NAME=cudnn-11.1-linux-x64-v8.0.5.39.tgz
elif [[ "$cuda_version" == "11.2" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.1/cudnn-11.2-linux-x64-v8.1.1.33.tgz
    CUDNN_TARBALL_NAME=cudnn-11.2-linux-x64-v8.1.1.33.tgz
elif [[ "$cuda_version" == "11.3" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.2.1/cudnn-11.3-linux-x64-v8.2.1.32.tgz
    CUDNN_TARBALL_NAME=cudnn-11.3-linux-x64-v8.2.1.32.tgz
elif [[ "$cuda_version" == "11.4" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.2.4/cudnn-11.4-linux-x64-v8.2.4.15.tgz
    CUDNN_TARBALL_NAME=cudnn-11.4-linux-x64-v8.2.4.15.tgz
elif [[ "$cuda_version" == "11.5" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.0/cudnn-11.5-linux-x64-v8.3.0.98.tgz
    CUDNN_TARBALL_NAME=cudnn-11.5-linux-x64-v8.3.0.98.tgz
elif [[ "$cuda_version" == "11.6" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.4.0/local_installers/11.6/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz
    CUDNN_TARBALL_NAME=cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz
elif [[ "$cuda_version" == "11.7" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
    CUDNN_TARBALL_NAME=cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
elif [[ "$cuda_version" == "11.8" ]]; then
    CUDNN_LINK=https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
    CUDNN_TARBALL_NAME=cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
elif [[ "$cuda_version" == "12.0" ]]; then
    echo "CUDNN support for CUDA version 12.0 not yet added"
    exit 1
fi
wget -c -q $CUDNN_LINK
if [[ "$cuda_version" == "11.6" || "$cuda_version" == "11.7" || "$cuda_version" == "11.8" ]]; then
    tar -xf $CUDNN_TARBALL_NAME -C ./
    CUDNN_EXTRACTED_TARBALL_NAME="${CUDNN_TARBALL_NAME::-7}"
    sudo cp -r "$CUDNN_EXTRACTED_TARBALL_NAME"/include/* /usr/local/include
    sudo cp -r "$CUDNN_EXTRACTED_TARBALL_NAME"/lib/* /usr/local/lib
    rm -rf "$CUDNN_EXTRACTED_TARBALL_NAME"
else
    sudo tar -xzf $CUDNN_TARBALL_NAME -C /usr/local
fi
rm $CUDNN_TARBALL_NAME 
sudo ldconfig
