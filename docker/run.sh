#! /usr/bin/env bash
set -euo pipefail

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Parameter controlling whether to attach GPUs to the Docker container
ATTACH_GPUS=true

gpu_arg=""
if $ATTACH_GPUS ; then gpu_arg="--gpus all" ; fi
image=${1:-flexflow}

FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
elif [[ "${FF_GPU_BACKEND}" != "cuda" ]]; then
  echo "Configuring FlexFlow to build for gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Letting FlexFlow build for a default GPU backend: cuda"
fi


if [[ "$image" == "flexflow-environment" ]]; then
    eval docker run -it "$gpu_arg" "flexflow-environment-${FF_GPU_BACKEND}:latest"
elif [[ "$image" == "flexflow" ]]; then
    eval docker run -it "$gpu_arg" "flexflow-${FF_GPU_BACKEND}:latest"
elif [[ "$image" == "mt5" ]]; then
    # Backward compatibility
    eval docker run -it "$gpu_arg" \
    -v "$(pwd)"/../examples/python/pytorch/mt5/data:/usr/FlexFlow/examples/python/pytorch/mt5/data \
    -v "$(pwd)"/../examples/python/pytorch/mt5/eng-sin.tar:/usr/FlexFlow/examples/python/pytorch/mt5/eng-sin.tar \
    "flexflow-${FF_GPU_BACKEND}:latest"
else
    echo "Docker image name not valid"
fi
