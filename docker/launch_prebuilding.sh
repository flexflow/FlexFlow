#! /usr/bin/env bash
set -euo pipefail

# Parse input params
python_version=${1:-latest}
cuda_version=${2:-11.8}
gpu_backend=${3:-cuda}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

export FF_CUDA_ARCH=all
export BUILD_LEGION_ONLY=ON
export INSTALL_DIR="/usr/legion"

# Build Docker Flexflow Container
echo "building docker"
./build.sh flexflow

# Cleanup any existing container with the same name
docker rm prelegion || true

# Create container to be able to copy data from the image
docker create --name prelegion flexflow-"${gpu_backend}"-"${cuda_version}":"${python_version}"

# Copy legion libraries to host
echo "extract legion library assets"
mkdir -p ../prebuild_legion
rm -rf ../prebuild_legion/tmp || true
docker cp prelegion:$INSTALL_DIR ../prebuild_legion/tmp


# Create the tarball file
cd ../prebuild_legion/tmp
export LEGION_TARBALL="legion_ubuntu-20.04_${gpu_backend}.tar.gz"
echo "Creating archive $LEGION_TARBALL"
tar -zcvf "../$LEGION_TARBALL" ./
cd ..
echo "Checking the size of the Legion tarball..."
du -h "$LEGION_TARBALL"


# Cleanup
rm -rf tmp/*
docker rm prelegion