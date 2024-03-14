#! /usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh <docker_image_name>
# Optional environment variables: FF_GPU_BACKEND, cuda_version, hip_version, ATTACH_GPUS, SHM_SIZE

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Parse input params
image=${1:-flexflow}
FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
cuda_version=${cuda_version:-"empty"}
hip_version=${hip_version:-"empty"}

# Parameter controlling whether to attach GPUs to the Docker container
ATTACH_GPUS=${ATTACH_GPUS:-true}
gpu_arg=""
if $ATTACH_GPUS ; then gpu_arg="--gpus all" ; fi

# Whether to attach inference weights / files (make sure to download the weights first)
ATTACH_INFERENCE_FILES=${ATTACH_INFERENCE_FILES:-false}

# Amount of shared memory to give the Docker container access to
# If you get a Bus Error, increase this value. If you don't have enough memory
# on your machine, decrease this value.
SHM_SIZE=${SHM_SIZE:-8192m}

# Check docker image name
if [[ "$image" != @(flexflow-environment|flexflow) ]]; then
  echo "Error, image name ${image} is invalid. Choose between 'flexflow-environment', 'flexflow'."
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

# gpu backend version suffix for the docker image.
gpu_backend_version=""

if [[ "${FF_GPU_BACKEND}" == "cuda" || "${FF_GPU_BACKEND}" == "hip_cuda" ]]; then
  # Autodetect cuda version if not specified
  if [[ $cuda_version == "empty" ]]; then
    # shellcheck disable=SC2015
    cuda_version=$(command -v nvcc >/dev/null 2>&1 && nvcc --version | grep "release" | awk '{print $NF}' || true)
    # Change cuda_version eg. V11.7.99 to 11.7
    cuda_version=${cuda_version:1:4}
    if [[ -z "$cuda_version" ]]; then
      echo "Could not detect CUDA version. Please specify one manually by setting the 'cuda_version' env."
      exit 1
    fi
  fi
  # Check that CUDA version is supported
  if [[ "$cuda_version" != @(11.1|11.2|11.3|11.4|11.5|11.6|11.7|11.8|12.0|12.1|12.2) ]]; then
    echo "cuda_version is not supported, please choose among {11.1|11.2|11.3|11.4|11.5|11.6|11.7|11.8|12.0|12.1|12.2}"
    exit 1
  fi
  # Use CUDA 12.2 for all versions greater or equal to 12.2 for now
  if [[ "$cuda_version" == @(12.3|12.4|12.5|12.6|12.7|12.8|12.9) ]]; then
    cuda_version=12.2
  fi
  # Set cuda version suffix to docker image name
  echo "Running $image docker image with CUDA $cuda_version"
  gpu_backend_version="-${cuda_version}"
fi

if [[ "${FF_GPU_BACKEND}" == "hip_rocm" || "${FF_GPU_BACKEND}" == "hip_cuda" ]]; then
  # Autodetect HIP version if not specified
  if [[ $hip_version == "empty" ]]; then
    # shellcheck disable=SC2015
    hip_version=$(command -v hipcc >/dev/null 2>&1 && hipcc --version | grep "HIP version:" | awk '{print $NF}' || true)
    # Change hip_version eg. 5.6.31061-8c743ae5d to 5.6
    hip_version=${hip_version:0:3}
    if [[ -z "$hip_version" ]]; then
      echo "Could not detect HIP version. Please specify one manually by setting the 'hip_version' env."
      exit 1
    fi
  fi
  # Check that HIP version is supported
  if [[ "$hip_version" != @(5.3|5.4|5.5|5.6) ]]; then
    echo "hip_version is not supported, please choose among {5.3, 5.4, 5.5, 5.6}"
    exit 1
  fi
  echo "Running $image docker image with HIP $hip_version"
  if [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
    gpu_backend_version="-${hip_version}"
  fi
fi

# Check that image exists, if fails, print the default error message.
if [[ "$(docker images -q "${image}-${FF_GPU_BACKEND}${gpu_backend_version}":latest 2> /dev/null)" == "" ]]; then
  echo "Error, ${image}-${FF_GPU_BACKEND}${gpu_backend_version}:latest does not exist!"
  if [[ "${FF_GPU_BACKEND}" == "cuda" ]]; then
    echo ""
    echo "To download the docker image, run:"
    echo "    FF_GPU_BACKEND=${FF_GPU_BACKEND} cuda_version=${cuda_version} $(pwd)/pull.sh $image"
    echo "To build the docker image from source, run:"
    echo "    FF_GPU_BACKEND=${FF_GPU_BACKEND} cuda_version=${cuda_version} $(pwd)/build.sh $image"
    echo ""
  elif [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
    echo ""
    echo "To download the docker image, run:"
    echo "    FF_GPU_BACKEND=${FF_GPU_BACKEND} hip_version=${hip_version} $(pwd)/pull.sh $image"
    echo "To build the docker image from source, run:"
    echo "    FF_GPU_BACKEND=${FF_GPU_BACKEND} hip_version=${hip_version} $(pwd)/build.sh $image"
    echo ""
  fi
  exit 1
fi

inference_volumes=""
if $ATTACH_INFERENCE_FILES ; then 
  inference_volumes="-v ~/.cache/flexflow:/usr/FlexFlow/inference";
fi

eval docker run -it "$gpu_arg" "--shm-size=${SHM_SIZE}" "${inference_volumes}" "${image}-${FF_GPU_BACKEND}${gpu_backend_version}:latest"
