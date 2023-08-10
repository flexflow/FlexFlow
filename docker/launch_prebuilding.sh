#! /usr/bin/env bash
set -euo pipefail

# Parse input params
python_version=${1:-lastest}
cuda_version=${2:-11.8}
gpu_backend=${3:-cuda}

# Running in Docker Container
docker run -it --name local-test "nvidia/cuda:${cuda_version}-cudnn8-devel-ubuntu20.04"
# Copy to Docker Container
docker cp ./test.sh local-test:/tmp/prebuilding.sh
# Call the Bash script in Docker Container
docker exec -it local-test bash -c ". /tmp/prebuilding.sh ${python_version} ${cuda_version} ${gpu_backend}"
# Extract the legion binary files of tar.gz file from Docker container
docker cp "local-test:/tmp/build/export/legion_ubuntu-20.04_${gpu_backend}.tar.gz ~/Desktop/"