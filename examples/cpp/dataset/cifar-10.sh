#!/bin/bash

# If not already downloaded
if [ ! -f ./cifar-10-batches-bin/data_batch_1.bin ]; then
    # If the archive does not exist, download it
    if [ ! -f ./cifar-10-binary.tar.gz ]; then
        wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
    fi

    # Extract all the files
    tar xf cifar-10-binary.tar.gz
fi
