#! /usr/bin/env bash
set -euo pipefail

# Usage: ./pull.sh <docker_image_name>
# Optional environment variables: FF_GPU_BACKEND, cuda_version

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Parse input params
image=${1:-flexflow}
FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
cuda_version=${cuda_version:-"empty"}

# Check docker image name
if [[ "${image}" != @(flexflow-environment|flexflow) ]]; then
  echo "Error, docker image name '${image}' is invalid. Choose between 'flexflow-environment' and 'flexflow'."
  exit 1
fi

# Check GPU backend
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
elif [[ "${FF_GPU_BACKEND}" != "cuda" ]]; then
  echo "Downloading $image docker image with gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Downloading $image docker image with default GPU backend: cuda"
fi

if [[ "${FF_GPU_BACKEND}" == "cuda" || "${FF_GPU_BACKEND}" == "hip_cuda" ]]; then
  # Autodetect cuda version if not specified
  if [[ $cuda_version == "empty" ]]; then
    cuda_version=$(command -v nvcc >/dev/null 2>&1 && nvcc --version | grep "release" | awk '{print $NF}')
    # Change cuda_version eg. V11.7.99 to 11.7
    cuda_version=${cuda_version:1:4}
  fi
  # Check that CUDA version is supported
  if [[ "$cuda_version" != @(11.1|11.3|11.7|11.2|11.5|11.6|11.8) ]]; then
    echo "cuda_version is not supported, please choose among {11.1|11.2|11.3|11.5|11.6|11.7|11.8}"
    exit 1
  fi
  # Set cuda version suffix to docker image name
  echo "Downloading $image docker image with CUDA $cuda_version"
  cuda_version="-${cuda_version}"
else
  # Empty cuda version suffix for non-CUDA images
  cuda_version=""
fi

# Download image
docker pull ghcr.io/flexflow/"$image-${FF_GPU_BACKEND}${cuda_version}"

# Tag downloaded image
docker tag ghcr.io/flexflow/"$image-${FF_GPU_BACKEND}${cuda_version}":latest "$image-${FF_GPU_BACKEND}${cuda_version}":latest 

# Check that image exists
docker image inspect "${image}-${FF_GPU_BACKEND}${cuda_version}":latest > /dev/null
