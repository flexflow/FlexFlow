#! /usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh <docker_image_name>
# Optional environment variables: FF_GPU_BACKEND, cuda_version, ATTACH_GPUS, SHM_SIZE

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Parse input params
image=${1:-flexflow}
FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
cuda_version=${cuda_version:-"empty"}

# Parameter controlling whether to attach GPUs to the Docker container
ATTACH_GPUS=${ATTACH_GPUS:-true}
gpu_arg=""
if $ATTACH_GPUS ; then gpu_arg="--gpus all" ; fi

# Amount of shared memory to give the Docker container access to
# If you get a Bus Error, increase this value. If you don't have enough memory
# on your machine, decrease this value.
SHM_SIZE=${SHM_SIZE:-8192m}

# Check docker image name
if [[ "$image" != @(flexflow-environment|flexflow) ]]; then
  echo "Error, image name ${image} is invalid. Choose between 'flexflow-environment' and 'flexflow'."
  exit 1
fi

# Check GPU backend
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
elif [[ "${FF_GPU_BACKEND}" != "cuda" ]]; then
  echo "Running $image docker image with gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Running $image docker image with default GPU backend: cuda"
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
  echo "Running $image docker image with CUDA $cuda_version"
  cuda_version="-${cuda_version}"
else
  # Empty cuda version suffix for non-CUDA images
  cuda_version=""
fi
  
if [[ "$image" == "flexflow-environment" ]]; then
    eval docker run -it "$gpu_arg" "--shm-size=${SHM_SIZE}" "flexflow-environment-${FF_GPU_BACKEND}${cuda_version}:latest"
elif [[ "$image" == "flexflow" ]]; then
    eval docker run -it "$gpu_arg" "--shm-size=${SHM_SIZE}" "flexflow-${FF_GPU_BACKEND}${cuda_version}:latest"
elif [[ "$image" == "mt5" ]]; then
    # Backward compatibility
    eval docker run -it "$gpu_arg" "--shm-size=${SHM_SIZE}" \
    -v "$(pwd)"/../examples/python/pytorch/mt5/data:/usr/FlexFlow/examples/python/pytorch/mt5/data \
    -v "$(pwd)"/../examples/python/pytorch/mt5/eng-sin.tar:/usr/FlexFlow/examples/python/pytorch/mt5/eng-sin.tar \
    "flexflow-${FF_GPU_BACKEND}${cuda_version}:latest"
else
    echo "Docker image name not valid"
fi
