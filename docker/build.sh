#! /usr/bin/env bash
set -euo pipefail

# 1. pass the desired version of CUDA to build.sh 
# (if not, build with the version on current machine, produce an error if no CUDA installed)
# 2. pass the CUDA version to docker file and use CUDA images based on input version
# 3. Add the CUDA version as a suffix to the image name (both for the environment and flexflow/specinfer images).
# 4. Edit the publish.sh / workflow file to publish one image per version, using for example 11.1, 11.3, 11.5, 11.6, 11.7, 11.8 (you can start by just building only two versions, e.g. 11.1 and 11.8 to check if it works)

# Usage: ./build.sh <docker_image_name>

# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Cd into $FF_HOME. Assumes this script is in $FF_HOME/docker
cd "$SCRIPT_DIR/.."

#!/bin/bash

# Use default value on current machine, if cuda is not installed, cuda_version will be empty,
# which will cause an error later 

cuda_version="empty"
image="flexflow"

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
  # Available versions: 11.1.1 | 11.2.2 | 11.3.1 | 11.5.2 | 11.6.2 | 11.7.1 | 11.8.0
  echo "cuda_version is not supported, please choose among {11.1,11.3,11.5,11.6,11.7,11.8}"
  exit 1
fi

# modify cuda version to available versions
if [[ "$cuda_version" == @(11.1|11.3|11.7) ]]; then
  cuda_version=${cuda_version}.1
elif [[ "$cuda_version" == @(11.2|11.5|11.6) ]]; then 
  cuda_version=${cuda_version}.2
elif [[ "$cuda_version" == @(11.8) ]]; then 
  cuda_version=${cuda_version}.0
fi



# Get name of desired Docker image as input
if [[ "$image" != @(flexflow-environment|flexflow) ]]; then
  echo "Error, image name ${image} is invalid. Choose between 'flexflow-environment' and 'flexflow'."
  exit 1
fi

# Set up GPU backend
FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
elif [[ "${FF_GPU_BACKEND}" != "cuda" ]]; then
  echo "Configuring FlexFlow to build for gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Letting FlexFlow build for a default GPU backend: cuda"
fi

# Build the FlexFlow Enviroment docker image with input cuda version
docker build --build-arg "FF_GPU_BACKEND=${FF_GPU_BACKEND}" --build-arg "cuda_version=${cuda_version}" -t "flexflow-environment-${FF_GPU_BACKEND}-${cuda_version}" -f docker/flexflow-environment/Dockerfile .

# If the user only wants to build the environment image, we are done
if [[ "$image" == "flexflow-environment" ]]; then
  exit 0
fi

# Gather arguments needed to build the FlexFlow image
# Get number of cores available on the machine. Build with all cores but one, to prevent RAM choking
cores_available=$(nproc --all)
n_build_cores=$(( cores_available -1 ))

# If FF_CUDA_ARCH is set to autodetect, we need to perform the autodetection here because the Docker
# image will not have access to GPUs during the build phase (due to a Docker restriction). In all other
# cases, we pass the value of FF_CUDA_ARCH directly to Cmake.
if [[ "${FF_CUDA_ARCH:-autodetect}" == "autodetect" ]]; then
  # Get CUDA architecture(s), if GPUs are available
  cat << EOF > ./get_gpu_arch.cu
#include <stdio.h>
int main() {
  int count = 0;
  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;
  if (count == 0) return -1;
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp prop;
    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))
      printf("%d ", prop.major*10+prop.minor);
  }
  return 0;
}
EOF
  gpu_arch_codes=""
  if command -v nvcc &> /dev/null
  then
    nvcc ./get_gpu_arch.cu -o ./get_gpu_arch
    gpu_arch_codes="$(./get_gpu_arch)"
  fi
  gpu_arch_codes="$(echo "$gpu_arch_codes" | xargs -n1 | sort -u | xargs)"
  gpu_arch_codes="${gpu_arch_codes// /,}"
  rm -f ./get_gpu_arch.cu ./get_gpu_arch

  if [[ -n "$gpu_arch_codes" ]]; then
  echo "Host machine has GPUs with architecture codes: $gpu_arch_codes"
  echo "Configuring FlexFlow to build for the $gpu_arch_codes code(s)."
  FF_CUDA_ARCH="${gpu_arch_codes}"
  export FF_CUDA_ARCH
  else
    echo "FF_CUDA_ARCH is set to 'autodetect', but the host machine does not have any compatible GPUs."
    exit 1
  fi
fi

# Build FlexFlow Docker image with input cuda version
# shellcheck source=/dev/null
. config/config.linux get-docker-configs
# Set value of BUILD_CONFIGS
get_build_configs

docker build --build-arg "N_BUILD_CORES=${n_build_cores}" --build-arg "FF_GPU_BACKEND=${FF_GPU_BACKEND}" --build-arg "BUILD_CONFIGS=${BUILD_CONFIGS}" --build-arg "cuda_version=${cuda_version}" -t "flexflow-${FF_GPU_BACKEND}-${cuda_version}" -f docker/flexflow/Dockerfile .
