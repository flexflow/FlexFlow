
# Flexflow operator unit test
1. To build test targets:
  - BatchMatmul: `cd ~/DLRM_FlexFlow && make app=src/ops/tests/batch_matmul_test -j 20 -f Makefile`
  - Transpose: `cd ~/DLRM_FlexFlow && make app=src/ops/tests/transpose -j 20 -f Makefile`
2. To run unit test
  - `cd ~/DLRM_FlexFlow/src/ops/tests/ && ./test_bootstrap.sh` # run bootstrap to request 8 gpus resource
  - `cd ~/DLRM_FlexFlow/src/ops/tests/ && ./test_run.sh` # run unit tests