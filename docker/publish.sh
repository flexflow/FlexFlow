#! /usr/bin/env bash
set -euo pipefail

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

cuda_version=${cuda_version:-11.1}
FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
image=${1:-flexflow}

echo "Publishing $image docker image for CUDA $cuda_version"

# modify cuda version to available versions
if [[ "$cuda_version" == @(11.1|11.3|11.7) ]]; then
  cuda_version=${cuda_version}.1
elif [[ "$cuda_version" == @(11.2|11.5|11.6) ]]; then 
  cuda_version=${cuda_version}.2
elif [[ "$cuda_version" == @(11.8) ]]; then 
  cuda_version=${cuda_version}.0
else
  echo "cuda_version is not supported, please choose among {11.1|11.2|11.3|11.5|11.6|11.7|11.8}"
  exit 1
fi

if [[ "${image}" != @(flexflow-environment-cuda|flexflow-environment-hip_cuda|flexflow-environment-hip_rocm|flexflow-environment-intel|flexflow-cuda|flexflow-hip_cuda|flexflow-hip_rocm|flexflow-intel) ]]; then
  echo "Error, image name ${image} is invalid. Choose between 'flexflow-environment-{cuda,hip_cuda,hip_rocm,intel}' and 'flexflow-{cuda,hip_cuda,hip_rocm,intel}'."
  exit 1
fi

# Check that image exists
if [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
  docker image inspect "${image}":latest > /dev/null
else
  docker image inspect "${image}-${cuda_version}":latest > /dev/null
fi

# Log into container registry
FLEXFLOW_CONTAINER_TOKEN=${FLEXFLOW_CONTAINER_TOKEN:-}
if [ -z "$FLEXFLOW_CONTAINER_TOKEN" ]; then echo "FLEXFLOW_CONTAINER_TOKEN secret is not available, cannot publish the docker image to ghrc.io"; exit; fi
echo "$FLEXFLOW_CONTAINER_TOKEN" | docker login ghcr.io -u flexflow --password-stdin

# Tag image to be uploaded
git_sha=${GITHUB_SHA:-$(git rev-parse HEAD)}
if [ -z "$git_sha" ]; then echo "Commit hash cannot be detected, cannot publish the docker image to ghrc.io"; exit; fi


if [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
  docker tag "$image":latest ghcr.io/flexflow/"$image":latest
else
  docker tag "${image}-${cuda_version}":latest ghcr.io/flexflow/"$image-$cuda_version":latest
fi

# Upload image
if [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
  docker push ghcr.io/flexflow/"$image":latest
else
  docker push ghcr.io/flexflow/"$image-$cuda_version":latest
fi
