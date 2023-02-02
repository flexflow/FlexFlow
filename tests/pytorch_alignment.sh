#! /usr/bin/env bash

#Run operators
./align/add/gen_tensors.sh
./align/conv2d/gen_tensors.sh
./align/cos/gen_tensors.sh
./align/concat/gen_tensors.sh
./align/embedding/gen_tensors.sh
./align/exp/gen_tensors.sh
./align/flat/gen_tensors.sh
./align/getitem/gen_tensors.sh
./align/identity/gen_tensors.sh
./align/multiply/gen_tensors.sh
./align/pool2d/gen_tensors.sh
./align/reducesum/gen_tensors.sh
./align/relu/gen_tensors.sh
./align/reshape/gen_tensors.sh
./align/scalar_add/gen_tensors.sh
./align/scalar_multiply/gen_tensors.sh
./align/scalar_sub/gen_tensors.sh
./align/scalar_truediv/gen_tensors.sh
./align/sigmoid/gen_tensors.sh
./align/sin/gen_tensors.sh
./align/subtract/gen_tensors.sh
./align/tanh/gen_tensors.sh
./align/transpose/gen_tensors.sh
./align/view_embedding/gen_tensors.sh


#Run tests
python -m pytest align/align_test.py

