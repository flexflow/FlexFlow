#! /usr/bin/env bash
set -euo pipefail

image=${1:-flexflow-environment}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Download image
docker pull ghcr.io/flexflow/"$image"

# Tag downloaded image
docker tag ghcr.io/flexflow/"$image":latest "$image":latest 

# Check that image exists
image_exists=$(docker image inspect ${image}:latest)
