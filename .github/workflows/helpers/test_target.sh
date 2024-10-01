#! /usr/bin/env bash

set -euo pipefail
set -x

DIR="$(realpath -- "$(dirname "${BASH_SOURCE[0]}")")"
REPO="$(realpath -- "$DIR/../../../")"

TEST_LIBS=("${@/%/-tests}")
REGEX="^($(IFS='|'; echo "${TEST_LIBS[*]}"))\$"

cd "$REPO/build-ci"
make -j $(( $(nproc) < 2 ? 1 : $(nproc)-1 )) "${TEST_LIBS[@]}"
ctest --progress --output-on-failure -L "$REGEX"
