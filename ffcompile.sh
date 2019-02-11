#!/bin/bash

# GPU_ARCH must be one of fermi, k20, pascal, volta
export GPU_ARCH=volta

# set CUDA_HOME to the directoy of cuda, which should have subdirectories bin/, include/, and lib/
export CUDA_HOME=/usr/local/cuda
git submodule init
git submodule update

APP="$1"

if [ -z "$APP" ]; then echo "Usage: ./ffcompile app"; exit; fi

make -j 8 APP="${APP}"

