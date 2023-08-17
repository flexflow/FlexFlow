#! /usr/bin/env bash
set -euo pipefail

# Parse input params
python_version=${1:-lastest}
cuda_version=${2:-11.8}
gpu_backend=${3:-cuda}

export FF_CUDA_ARCH=all
export BUILD_LEGION_ONLY=ON

# Build Docker Flexflow Container
echo "building docker"
./docker/build.sh flexflow
./docker/run.sh flexflow
exit

# Copy legion libraries to host
container_id=$(docker ps | grep your-image-name | awk '{print $1}')
docker cp "$container_id":/usr/FlexFlow/build/deps ~/buildlegion

# Create the tarball file
cd ~/buildlegion
export LEGION_TARBALL="legion_ubuntu-20.04_${gpu_backend}.tar.gz"
echo "Creating archive $LEGION_TARBALL"
touch "$LEGION_TARBALL"
tar --exclude="$LEGION_TARBALL" -zcvf "$LEGION_TARBALL" .
echo "Checking the size of the Legion tarball..."
du -h "$LEGION_TARBALL"
