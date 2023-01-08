#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

./align/add/gen_tensors.sh
./align/conv2d/gen_tensors.sh
./align/embedding/gen_tensors.sh
./align/getitem/gen_tensors.sh
./align/layernorm/gen_tensors.sh
./align/linear/gen_tensors.sh
./align/mt5_encoder/gen_tensors.sh
./align/multiply/gen_tensors.sh
./align/subtract/gen_tensors.sh
./align/view_embedding/gen_tensors.sh

python -m pytest align/align_test.py

