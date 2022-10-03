#!/bin/bash
set -e
# Copy the config files into the Docker folder
rm -rf config && cp -r ../config ./config

# Get number of cores available on the machine. Build with all cores but one, to prevent RAM choking
cores_available=$(nproc --all)
n_build_cores=$(( cores_available -1 ))

# Get CUDA architecture, if GPUs are available
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
  gpu_arch_codes=$(./get_gpu_arch)
fi
gpu_arch_codes=$(echo "$gpu_arch_codes" | xargs -n1 | sort -u | xargs)
gpu_arch_codes=$(echo ${gpu_arch_codes// /,})
if [ -n "$gpu_arch_codes" ]; then
  echo "Host machine has GPUs with architecture codes: $gpu_arch_codes"
  echo "Configuring FlexFlow to build for the $gpu_arch_codes code(s)."
  sed -i "/FF_CUDA_ARCH/c\FF_CUDA_ARCH=${gpu_arch_codes}" ./config/config.linux
else
  echo "Could not detect any GPU on the host machine."
  echo "Letting FlexFlow build for a default GPU architecture: code=70"
  sed -i "/FF_CUDA_ARCH/c\FF_CUDA_ARCH=70" ./config/config.linux
fi
rm -f ./get_gpu_arch.cu ./get_gpu_arch

# Build Docker image
docker build --build-arg n_build_cores=$n_build_cores  -t flexflow .
