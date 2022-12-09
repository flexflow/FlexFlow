#! /usr/bin/env bash
set -euo pipefail

# Usage: ./build.sh [-b] <docker_image_name>
# Pass the -b flag to build the flexflow-environment image from scratch 
# (as opposed to downloading the latest version) from ghrc.io

# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Cd into $FF_HOME. Assumes this script is in $FF_HOME/docker
cd "$SCRIPT_DIR/.."

# Get name of desired Docker image as input
image=${1:-flexflow}
if [[ "${image}" != @(flexflow-environment|flexflow) ]]; then
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

# Build the FlexFlow Enviroment docker image
docker build --build-arg "FF_GPU_BACKEND=${FF_GPU_BACKEND}" -t "flexflow-environment-${FF_GPU_BACKEND}" -f docker/flexflow-environment/Dockerfile .

# If the user only wants to build the environment image, we are done
if [[ "$image" == "flexflow-environment" ]]; then
  exit 0
fi

# Gather arguments needed to build the FlexFlow image
# Get number of cores available on the machine. Build with all cores but one, to prevent RAM choking
cores_available=$(nproc --all)
n_build_cores=$(( cores_available -1 ))

if [[ "${FF_AUTODETECT_CUDA_ARCH:-ON}" == "OFF" ]]; then
  echo "FF_AUTODETECT_CUDA_ARCH is turned off. Building for all compatible architectures."
else
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
  else
    echo "Could not detect any GPU on the host machine."
    echo "Letting FlexFlow build for a default GPU architecture: code=70"
    FF_CUDA_ARCH=70
  fi
  export FF_CUDA_ARCH
fi

# Build FlexFlow Docker image
# shellcheck source=/dev/null
. config/config.linux get-docker-configs
# Set value of BUILD_CONFIGS
get_build_configs

docker build --build-arg "N_BUILD_CORES=${n_build_cores}" --build-arg "FF_GPU_BACKEND=${FF_GPU_BACKEND}" --build-arg "BUILD_CONFIGS=${BUILD_CONFIGS}" -t "flexflow-${FF_GPU_BACKEND}" -f docker/flexflow/Dockerfile .
