#! /usr/bin/env bash
set -euo pipefail

# Cd into directory holding this script
cd "$( echo "${BASH_SOURCE[0]%/*}" )"

# (Idemponent command) Build FlexFlow docker container
../../../../docker/build.sh

# Build Docker-MT5 container
docker build -t flexflow-mt5 .
