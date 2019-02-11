#!/bin/bash

# AlexNet experiments
./ffcompile.sh alexnet

./alexnet -b 64 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 5000 --strategy strategies/alexnet_data_parallel.strategy
./alexnet -b 64 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 5000 --strategy strategies/alexnet_optimized.strategy
./alexnet -b 128 -ll:gpu 2 -ll:fsize 5000 -ll:zsize 5000 --strategy strategies/alexnet_data_parallel.strategy
./alexnet -b 128 -ll:gpu 2 -ll:fsize 5000 -ll:zsize 5000 --strategy strategies/alexnet_optimized.strategy
./alexnet -b 256 -ll:gpu 4 -ll:fsize 5000 -ll:zsize 5000 --strategy strategies/alexnet_data_parallel.strategy
./alexnet -b 256 -ll:gpu 4 -ll:fsize 5000 -ll:zsize 5000 --strategy strategies/alexnet_optimized.strategy

# ResNet experiments
./ffcompile.sh resnet
./resnet -b 64 -ll:gpu 1 -ll:fsize 9000 -ll:zsize 5000 --strategy strategies/resnet121_data_parallel.strategy
./resnet -b 64 -ll:gpu 1 -ll:fsize 9000 -ll:zsize 5000 --strategy strategies/resnet121_optimized.strategy
./resnet -b 128 -ll:gpu 2 -ll:fsize 9000 -ll:zsize 5000 --strategy strategies/resnet121_data_parallel.strategy
./resnet -b 128 -ll:gpu 2 -ll:fsize 9000 -ll:zsize 5000 --strategy strategies/resnet121_optimized.strategy
./resnet -b 256 -ll:gpu 4 -ll:fsize 9000 -ll:zsize 5000 --strategy strategies/resnet121_data_parallel.strategy
./resnet -b 256 -ll:gpu 4 -ll:fsize 9000 -ll:zsize 5000 --strategy strategies/resnet121_optimized.strategy


