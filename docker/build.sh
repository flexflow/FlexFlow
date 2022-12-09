#! /usr/bin/env bash
set -euo pipefail

# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Cd into $FF_HOME. Assumes this script is in $FF_HOME/docker
cd "$SCRIPT_DIR/.."

# Get number of cores available on the machine. Build with all cores but one, to prevent RAM choking
cores_available=$(nproc --all)
n_build_cores=$(( cores_available -1 ))

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

FF_GPU_BACKEND=${FF_GPU_BACKEND:-""}

if [[ -n "$FF_GPU_BACKEND" ]]; then
  echo "Configuring FlexFlow to build for gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Letting FlexFlow build for a default GPU backend: cuda"
  FF_GPU_BACKEND="cuda"
fi

# Build base Docker image
docker build --build-arg N_BUILD_CORES=$n_build_cores --build-arg "FF_CUDA_ARCH=${FF_CUDA_ARCH}" --build-arg "FF_GPU_BACKEND=${FF_GPU_BACKEND}" -t flexflow -f docker/base/Dockerfile .

# Build mt5 docker image if required
image=${1:-base}

if [[ "$image" == "mt5" ]]; then
  docker build -t flexflow-mt5 -f docker/mt5/Dockerfile .
fi
