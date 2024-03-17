#! /usr/bin/env bash

set -euo pipefail

DIR="$(realpath -- "$(dirname "${BASH_SOURCE[0]}")")"
REPO="$(realpath -- "$DIR/../../../")"

cd "$REPO/build-ci"
make -j $(( $(nproc) < 2 ? 1 : $(nproc)-1 )) "$@"
