#! /usr/bin/env bash
set -euo pipefail

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

image=${1:-"flexflow-cuda"}
if [[ "${image}" != @(flexflow-environment-cuda|flexflow-environment-hip_cuda|flexflow-environment-hip_rocm|flexflow-environment-intel|flexflow-cuda|flexflow-hip_cuda|flexflow-hip_rocm|flexflow-intel) ]]; then
  echo "Error, image name ${image} is invalid. Choose between 'flexflow-environment-{cuda,hip_cuda,hip_rocm,intel}' and 'flexflow-{cuda,hip_cuda,hip_rocm,intel}'."
  exit 1
fi

# Download image
docker pull ghcr.io/flexflow/"$image"

# Tag downloaded image
docker tag ghcr.io/flexflow/"$image":latest "$image":latest 

# Check that image exists
docker image inspect "${image}":latest > /dev/null
