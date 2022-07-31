#! /usr/bin/env bash

GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT"

FILES=($(git ls-files | grep -E '\.(cc|cpp|cu)$'))

clang-format -i "${FILES[@]}"
