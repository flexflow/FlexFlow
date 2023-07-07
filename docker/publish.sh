#! /usr/bin/env bash
set -euo pipefail

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

cuda_version="empty"
image="flexflow-cuda"

# Parse command-line options
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --cuda_version)
      cuda_version="$2"
      shift 2
      ;;
    --image_name)
      image="$2"
      shift 2
      ;;
    *)
      echo "Invalid option: $key"
      exit 1
      ;;
  esac
done

if [[ $cuda_version == "empty" ]]; then
  cuda_version=$(command -v nvcc >/dev/null 2>&1 && nvcc --version | grep "release" | awk '{print $NF}')
  # Change cuda_version eg. V11.7.99 to 11.7
  cuda_version=${cuda_version:1:4}
fi

if [[ "$cuda_version" != @(11.1|11.3|11.5|11.6|11.7|11.8) ]]; then
  # validate the verison of CUDA against a list of supported ones
  # 11.1, 11.3, 11.5, 11.6, 11.7, 11.8
  echo "cuda_version is not supported, please choose among {11.1,11.3,11.5,11.6,11.7,11.8}"
  exit 1
fi

# modify cuda version to available versions
if [[ "$cuda_version" != @(11.1|11.3|11.7) ]]; then
  cuda_version=${cuda_version}.1
elif [[ "$cuda_version" != @(11.2|11.5|11.6) ]]; then 
  cuda_version=${cuda_version}.2
elif [[ "$cuda_version" != @(11.8) ]]; then 
  cuda_version=${cuda_version}.0
fi


# Check publish specinfer environment image
if [[ "${image}" == @(specinfer-environment-cuda|specinfer-environment-hip_cuda|specinfer-environment-hip_rocm|specinfer-environment-intel) ]]; then
  echo "specinfer does not publish environment images"
  exit 1
fi

# Check valid image name
if [[ "${image}" != @(flexflow-environment-cuda|flexflow-environment-hip_cuda|flexflow-environment-hip_rocm|flexflow-environment-intel|flexflow-cuda|flexflow-hip_cuda|flexflow-hip_rocm|flexflow-intel|specinfer-cuda|specinfer-hip_cuda|specinfer-hip_rocm|specinfer-intel) ]]; then
  echo "Error, image name ${image} is invalid. Choose between 'flexflow-environment-{cuda,hip_cuda,hip_rocm,intel}' and 'flexflow-{cuda,hip_cuda,hip_rocm,intel}'."
  exit 1
fi

# Check that image exists
if [[ "${image}" == @(specinfer-cuda|specinfer-hip_cuda|specinfer-hip_rocm|specinfer-intel) ]]; then 
  SUBSTR="${image:10}"
  docker image inspect "flexflow-${SUBSTR}-${cuda_version}":latest > /dev/null
else
  docker image inspect "${image}-${cuda_version}":latest > /dev/null
fi

# Log into container registry
FLEXFLOW_CONTAINER_TOKEN=${FLEXFLOW_CONTAINER_TOKEN:-}
if [ -z "$FLEXFLOW_CONTAINER_TOKEN" ]; then echo "FLEXFLOW_CONTAINER_TOKsecret is not available, cannot publish the docker image to ghrc.io"; exit; fi
echo "$FLEXFLOW_CONTAINER_TOKEN" | docker login ghcr.io -u flexflow --password-stdin

# Tag image to be uploaded
git_sha=${GITHUB_SHA:-$(git rev-parse HEAD)}
if [ -z "$git_sha" ]; then echo "Commit hash cannot be detected, cannot publish the docker image to ghrc.io"; exit; fi

# If in "inference" branch, which tries to publish "specinfer" images,
# tags the all images as "specinfer-{cuda, hip_cuda, hip_rocm, intel}"; if in others, do as orginal
if [[ "${image}" == @(specinfer-cuda|specinfer-hip_cuda|specinfer-hip_rocm|specinfer-intel) ]]; then 
  # tag with the cuda version
  SUBSTR="${image:10}-${cuda_version}"
  docker tag flexflow-"$SUBSTR":latest ghcr.io/flexflow/specinfer-"$SUBSTR":latest
else
  docker tag "$image":latest ghcr.io/flexflow/"$image":latest
fi

# Upload image with the cuda version
docker push ghcr.io/flexflow/"$image-$cuda_version":latest
