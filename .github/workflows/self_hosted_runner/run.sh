#! /usr/bin/env bash
set -euo pipefail

ORGANIZATION="flexflow/FlexFlow"
REGISTRATION_TOKEN=""
GPUS=""

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Check that the ORGANIZATION and REGISTRATION_TOKEN are set
if [[ -z "${ORGANIZATION+1}" ]]; then 
    echo "Please set the ORGANIZATION before starting the runner"
    exit
fi
if [[ -z "${REGISTRATION_TOKEN+1}" ]]; then 
    echo "Please set the REGISTRATION_TOKEN before starting the runner"
    exit
fi

# Ensure that one or more GPU indexes are passed
if [[ -z "${GPUS+1}" ]]; then 
    echo "Please set the GPU index(es) before starting the runner"
    exit
fi

docker run --detach \
  --env ORGANIZATION=$ORGANIZATION \
  --env REGISTRATION_TOKEN=$REGISTRATION_TOKEN \
  --gpus $GPUS \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --name flexflow-gpu-runner \
  ff-runner-image
