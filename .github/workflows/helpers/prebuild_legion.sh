#! /usr/bin/env bash
set -euo pipefail

# Parse input params
python_version=${python_version:-"empty"}
gpu_backend=${gpu_backend:-"empty"}
gpu_backend_version=${gpu_backend_version:-"empty"}

if [[ "${gpu_backend}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of gpu_backend (${gpu_backend}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
else
  echo "Pre-building Legion with GPU backend: ${gpu_backend}"
fi

if [[ "${gpu_backend}" == "cuda" || "${gpu_backend}" == "hip_cuda" ]]; then
    # Check that CUDA version is supported. Versions above 12.0 not supported because we don't publish docker images for it yet.
    if [[ "$gpu_backend_version" != @(11.1|11.2|11.3|11.4|11.5|11.6|11.7|11.8|12.0) ]]; then
        echo "cuda_version is not supported, please choose among {11.1|11.2|11.3|11.4|11.5|11.6|11.7|11.8|12.0}"
        exit 1
    fi
    export cuda_version="$gpu_backend_version"
elif [[ "${gpu_backend}" == "hip_rocm" ]]; then
    # Check that HIP version is supported
    if [[ "$gpu_backend_version" != @(5.3|5.4|5.5|5.6) ]]; then
        echo "hip_version is not supported, please choose among {5.3, 5.4, 5.5, 5.6}"
        exit 1
    fi
    export hip_version="$gpu_backend_version"
else
    echo "gpu backend: ${gpu_backend} and gpu_backend_version: ${gpu_backend_version} not yet supported."
    exit 1
fi

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

export FF_GPU_BACKEND="${gpu_backend}"
export FF_CUDA_ARCH=all
export FF_HIP_ARCH=all
export BUILD_LEGION_ONLY=ON
export INSTALL_DIR="/usr/legion"
export python_version="${python_version}"

# Build Docker Flexflow Container
echo "building docker"
../../../docker/build.sh flexflow

# Cleanup any existing container with the same name
docker rm prelegion || true

# Create container to be able to copy data from the image
docker create --name prelegion flexflow-"${gpu_backend}"-"${gpu_backend_version}":latest

# Copy legion libraries to host
echo "extract legion library assets"
mkdir -p ../../../prebuilt_legion_assets
rm -rf ../../../prebuilt_legion_assets/tmp || true
docker cp prelegion:$INSTALL_DIR ../../../prebuilt_legion_assets/tmp


# Create the tarball file
cd ../../../prebuilt_legion_assets/tmp
export LEGION_TARBALL="legion_ubuntu-20.04_${gpu_backend}-${gpu_backend_version}_py${python_version}.tar.gz"

echo "Creating archive $LEGION_TARBALL"
tar -zcvf "../$LEGION_TARBALL" ./
cd ..
echo "Checking the size of the Legion tarball..."
du -h "$LEGION_TARBALL"


# Cleanup
rm -rf tmp/*
docker rm prelegion
