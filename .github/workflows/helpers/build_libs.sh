#! /usr/bin/env bash

set -euo pipefail

cd build
make -j $(( $(nproc) < 2 ? 1 : $(nproc)-1 )) "$@"
