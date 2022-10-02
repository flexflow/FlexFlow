#!/bin/bash

# Copy the config files into the Docker folder
cp -r ../config/ .

# Get number of cores available on the machine. Build with all cores but one, to prevent RAM choking
cores_available=$(nproc --all)
n_build_cores=$(( cores_available -1 ))

# Get CUDA architecture, if GPUs are available
# TODO

# Build Docker image
docker build --build-arg path=$n_build_cores -t flexflow .
