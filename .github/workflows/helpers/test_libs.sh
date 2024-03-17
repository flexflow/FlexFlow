#! /usr/bin/env bash

set -euo pipefail
set -x

TEST_LIBS=("${@/%/-tests}")
REGEX="^$(IFS='|'; echo "${TEST_LIBS[*]}")\$"

cd build
make -j $(( $(nproc) < 2 ? 1 : $(nproc)-1 )) "${TEST_LIBS[@]}"
ctest --progress --output-on-failure -L "$REGEX"
