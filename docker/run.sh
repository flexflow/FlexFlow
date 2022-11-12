#! /usr/bin/env bash
set -euo pipefail

# Parameter controlling whether to attach GPUs to the Docker container
ATTACH_GPUS=true

gpu_arg=""
if $ATTACH_GPUS ; then gpu_arg="--gpus all" ; fi
image=${1:-flexflow}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"


if [[ "$image" == "environment" ]]; then
    eval docker run -it "$gpu_arg" flexflow-environment:latest
elif [[ "$image" == "flexflow" ]]; then
    eval docker run -it "$gpu_arg" flexflow:latest
elif [[ "$image" == "mt5" ]]; then
    # Backward compatibility
    eval docker run -it "$gpu_arg" \
    -v "$(pwd)"/../examples/python/pytorch/mt5/data:/usr/FlexFlow/examples/python/pytorch/mt5/data \
    -v "$(pwd)"/../examples/python/pytorch/mt5/eng-sin.tar:/usr/FlexFlow/examples/python/pytorch/mt5/eng-sin.tar \
    flexflow:latest
else
    echo "Docker image name not valid"
fi
