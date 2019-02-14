#!/bin/bash

# set GPU_ARCH to one of fermi, k20, k80, pascal, and volta
export GPU_ARCH=volta

# set CUDA_HOME to the directoy of cuda, which should have subdirectories bin/, include/, and lib/
export CUDA_HOME=/usr/local/cuda

APP="$1"

if [ -z "$APP" ]; then echo "Usage: ./ffcompile app"; exit; fi

make -j 8 APP="${APP}"

