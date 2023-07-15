#! /usr/bin/env bash
set -euo pipefail

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Parameter controlling whether to attach GPUs to the Docker container
ATTACH_GPUS=true

# Amount of shared memory to give the Docker container access to
# If you get a Bus Error, increase this value. If you don't have enough memory
# on your machine, decrease this value.
SHM_SIZE=8192m

gpu_arg=""
if $ATTACH_GPUS ; then gpu_arg="--gpus all" ; fi
image=${1:-flexflow}

FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
elif [[ "${FF_GPU_BACKEND}" != "cuda" ]]; then
  echo "Running FlexFlow with GPU backend: ${FF_GPU_BACKEND}"
else
  echo "Running FlexFlow with default GPU backend: cuda"
fi


if [[ "$image" == "flexflow-environment" ]]; then
    eval docker run -it "$gpu_arg" "--shm-size=${SHM_SIZE}" "flexflow-environment-${FF_GPU_BACKEND}-11.6.2:latest"
elif [[ "$image" == "flexflow" ]]; then
    eval docker run -it "$gpu_arg" "--shm-size=${SHM_SIZE}" "flexflow-${FF_GPU_BACKEND}:latest"
elif [[ "$image" == "mt5" ]]; then
    # Backward compatibility
    eval docker run -it "$gpu_arg" "--shm-size=${SHM_SIZE}" \
    -v "$(pwd)"/../examples/python/pytorch/mt5/data:/usr/FlexFlow/examples/python/pytorch/mt5/data \
    -v "$(pwd)"/../examples/python/pytorch/mt5/eng-sin.tar:/usr/FlexFlow/examples/python/pytorch/mt5/eng-sin.tar \
    "flexflow-${FF_GPU_BACKEND}:latest"
else
    echo "Docker image name not valid"
fi
