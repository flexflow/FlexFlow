#! /usr/bin/env bash
set -euo pipefail

# Parse input params
python_version=${1:-empty}
cuda_version=${2:-empty}
gpu_backend=${3:-empty}

# Running in Docker Container
docker run -it --name local-test "nvidia/cuda:${cuda_version}-cudnn8-devel-ubuntu20.04"
# Copy to Docker Container
docker cp ./test.sh local-test:/tmp/prebuilding.sh
# Call the Bash script in Docker Container
docker exec -it local-test bash -c ". /tmp/prebuilding.sh ${python_version} ${cuda_version} ${gpu_backend}"
s