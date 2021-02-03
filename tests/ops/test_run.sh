#!/bin/bash

# this is to silience MPI warning http://www.open-mpi.org/faq/?category=openfabrics#ofa-default-subnet-gid
DIR=~/.openmpi
FILE=~/.openmpi/mca-params.conf
if [ ! -d "$DIR" ]; then
  mkdir ~/.openmpi
fi
if [ ! -f "$FILE" ]; then
  touch ~/.openmpi/mca-params.conf
  echo "btl_openib_warn_default_gid_prefix=0" >> ~/.openmpi/mca-params.conf
fi

cd ~/DLRM_FlexFlow/src/ops/tests/ && python -m unittest test_harness.TransposeTest
cd ~/DLRM_FlexFlow/src/ops/tests/ && python -m unittest test_harness.BatchMatmulTest
cd ~/DLRM_FlexFlow/src/ops/tests/ && python -m unittest test_harness.ReshapeTest
cd ~/DLRM_FlexFlow/src/ops/tests/ && python -m unittest test_harness.TanhTest
cd ~/DLRM_FlexFlow/src/ops/tests/ && python -m unittest test_harness.DotCompressorTest

