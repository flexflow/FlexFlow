#! /usr/bin/env bash
set -euo pipefail

# Usage: ./build.sh <docker_image_name>
# Optional environment variables: FF_GPU_BACKEND, cuda_version

# Cd into $FF_HOME. Assumes this script is in $FF_HOME/docker
cd "${BASH_SOURCE[0]%/*}/.."

# Parse input params
image=${1:-flexflow}
FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
cuda_version=${cuda_version:-"empty"}
python_version=${python_version:-latest}

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
  echo "Building $image docker image with gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Building $image docker image with default GPU backend: cuda"
fi

if [[ "${FF_GPU_BACKEND}" == "cuda" || "${FF_GPU_BACKEND}" == "hip_cuda" ]]; then
  # Autodetect cuda version if not specified
  if [[ $cuda_version == "empty" ]]; then
    cuda_version=$(command -v nvcc >/dev/null 2>&1 && nvcc --version | grep "release" | awk '{print $NF}')
    # Change cuda_version eg. V11.7.99 to 11.7
    cuda_version=${cuda_version:1:4}
  fi
  # Check that CUDA version is supported, and modify cuda version to include default subsubversion
  if [[ "$cuda_version" == @(11.1|11.3|11.7) ]]; then
    cuda_version_input=${cuda_version}.1
  elif [[ "$cuda_version" == @(11.2|11.5|11.6) ]]; then 
    cuda_version_input=${cuda_version}.2
  elif [[ "$cuda_version" == @(11.8) ]]; then 
    cuda_version_input=${cuda_version}.0
  else
    echo "cuda_version is not supported, please choose among {11.1|11.2|11.3|11.5|11.6|11.7|11.8}"
    exit 1
  fi
  # Set cuda version suffix to docker image name
  echo "Building $image docker image with CUDA $cuda_version"
  cuda_version="-${cuda_version}"
else
  # Empty cuda version suffix for non-CUDA images
  cuda_version=""
  # Pick a default CUDA version for the base docker image from NVIDIA
  cuda_version_input="11.8.0"
fi

# check python_version
if [[ "$python_version" != @(3.8|3.9|3.10|3.11|latest) ]]; then
  echo "python_version not supported!"
  exit 0
fi

docker build --build-arg "FF_GPU_BACKEND=${FF_GPU_BACKEND}" --build-arg "cuda_version=${cuda_version_input}" --build-arg "python_version=${python_version}" -t "flexflow-environment-${FF_GPU_BACKEND}${cuda_version}" -f docker/flexflow-environment/Dockerfile .

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

# Build FlexFlow Docker image
# shellcheck source=/dev/null
. config/config.linux get-docker-configs
# Set value of BUILD_CONFIGS
get_build_configs

docker build --build-arg "N_BUILD_CORES=${n_build_cores}" --build-arg "FF_GPU_BACKEND=${FF_GPU_BACKEND}" --build-arg "BUILD_CONFIGS=${BUILD_CONFIGS}" --build-arg "cuda_version=${cuda_version}" -t "flexflow-${FF_GPU_BACKEND}${cuda_version}" -f docker/flexflow/Dockerfile .
