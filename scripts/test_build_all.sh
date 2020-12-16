#!/bin/bash
make app=src/ops/tests/batch_matmul_test -j -f Makefile
make app=src/ops/tests/transpose_test -j -f Makefile
make app=src/ops/tests/reshape_test -j -f Makefile
make app=src/ops/tests/flat_test -j -f Makefile
make app=src/ops/tests/tanh_test -j -f Makefile
make app=src/ops/tests/concat_test -j -f Makefile
make app=src/ops/tests/linear_test -j -f Makefile
