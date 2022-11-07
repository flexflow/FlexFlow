#! /usr/bin/env bash
set -euo pipefail

RUNNER_VERSION="2.298.2"

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Check that RUNNER_VERSION is defined
if [[ -z "${RUNNER_VERSION+1}" ]]; then 
    echo "Please set the RUNNER_VERSION variable before building the runner"
    exit
fi

echo "Building FlexFlow self-hosted runner, version ${RUNNER_VERSION}"
docker build --build-arg RUNNER_VERSION=$RUNNER_VERSION -t ff-runner-image -f ./Dockerfile .
