#! /usr/bin/env bash
set -euo pipefail

# Cd into directory holding this script
cd "$( echo "${BASH_SOURCE[0]%/*}" )"

docker build -t flexflow-mt5 .