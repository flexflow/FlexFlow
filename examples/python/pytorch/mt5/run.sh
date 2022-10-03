#!/bin/bash

docker run -it --gpus all \
    -v $(pwd)/data:/usr/FlexFlow/examples/python/pytorch/mt5/data \
    -v $(pwd)/eng-sin.tar:/usr/FlexFlow/examples/python/pytorch/mt5/eng-sin.tar \
    flexflow-mt5:latest