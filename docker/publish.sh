#! /usr/bin/env bash
set -euo pipefail

image=${1:-flexflow-environment}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Check that image exists
image_exists=$(docker image inspect ${image}:latest)

# Log into container registry
FLEXFLOW_CONTAINER_TOKEN=${FLEXFLOW_CONTAINER_TOKEN:-}
if [ -z "$FLEXFLOW_CONTAINER_TOKEN" ]; then echo "FLEXFLOW_CONTAINER_TOKEN secret is not available, cannot publish the docker image to ghrc.io"; exit; fi
echo "$FLEXFLOW_CONTAINER_TOKEN" | docker login ghcr.io -u flexflow --password-stdin

# Tag image to be uploaded
git_sha=${GITHUB_SHA:-$(git rev-parse HEAD)}
if [ -z "$git_sha" ]; then echo "Commit hash cannot be detected, cannot publish the docker image to ghrc.io"; exit; fi
docker tag "$image":latest ghcr.io/flexflow/"$image":"$git_sha"

# Upload image
docker push ghcr.io/flexflow/"$image":"$git_sha"
