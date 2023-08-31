#! /usr/bin/env bash
set -euo pipefail

# Usage: ./publish.sh <docker_image_name>
# Optional environment variables: FF_GPU_BACKEND, cuda_version, hip_version

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Parse input params
image=${1:-flexflow}
FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
cuda_version=${cuda_version:-"empty"}
hip_version=${hip_version:-"empty"}

# Check docker image name
if [[ "${image}" != @(flexflow-environment|flexflow) ]]; then
  echo "Error, docker image name '${image}' is invalid. Choose between 'flexflow-environment' and 'flexflow'."
  exit 1
fi

# gpu backend version suffix for the docker image.
gpu_backend_version=""

# Check GPU backend
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
elif [[ "${FF_GPU_BACKEND}" != "cuda" ]]; then
  echo "Publishing $image docker image with gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Publishing $image docker image with default GPU backend: cuda"
fi

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
  # Set cuda version suffix to docker image name
  echo "Publishing $image docker image with CUDA $cuda_version"
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
  echo "Pubilishing $image docker image with HIP $hip_version"
  if [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
    gpu_backend_version="-${hip_version}"
  fi
fi

# Check that image exists
docker image inspect "${image}-${FF_GPU_BACKEND}${gpu_backend_version}":latest > /dev/null

# Log into container registry
FLEXFLOW_CONTAINER_TOKEN=${FLEXFLOW_CONTAINER_TOKEN:-}
if [ -z "$FLEXFLOW_CONTAINER_TOKEN" ]; then echo "FLEXFLOW_CONTAINER_TOKEN secret is not available, cannot publish the docker image to ghrc.io"; exit; fi
echo "$FLEXFLOW_CONTAINER_TOKEN" | docker login ghcr.io -u flexflow --password-stdin

# Tag image to be uploaded
git_sha=${GITHUB_SHA:-$(git rev-parse HEAD)}
if [ -z "$git_sha" ]; then echo "Commit hash cannot be detected, cannot publish the docker image to ghrc.io"; exit; fi
docker tag "${image}-${FF_GPU_BACKEND}${gpu_backend_version}":latest ghcr.io/flexflow/"${image}-${FF_GPU_BACKEND}${gpu_backend_version}":latest

# Upload image
docker push ghcr.io/flexflow/"${image}-${FF_GPU_BACKEND}${gpu_backend_version}":latest

