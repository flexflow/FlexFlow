#!/bin/bash
make app="$FF_HOME"/tests/ops/batch_matmul_test -j -f Makefile
make app="$FF_HOME"/tests/ops/transpose_test -j -f Makefile
make app="$FF_HOME"/tests/ops/reshape_test -j -f Makefile
make app="$FF_HOME"/tests/ops/flat_test -j -f Makefile
make app="$FF_HOME"/tests/ops/tanh_test -j -f Makefile
make app="$FF_HOME"/tests/ops/concat_test -j -f Makefile
make app="$FF_HOME"/tests/ops/linear_test -j -f Makefile
