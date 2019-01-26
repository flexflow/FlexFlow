#!/bin/bash

# AlexNet experiments
./ffcompile.sh alexnet

./alexnet -b 64 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 5000 --strategy dp
./alexnet -b 64 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 5000 --strategy opt
./alexnet -b 128 -ll:gpu 2 -ll:fsize 5000 -ll:zsize 5000 --strategy dp
./alexnet -b 128 -ll:gpu 2 -ll:fsize 5000 -ll:zsize 5000 --strategy opt
./alexnet -b 256 -ll:gpu 4 -ll:fsize 5000 -ll:zsize 5000 --strategy dp
./alexnet -b 256 -ll:gpu 4 -ll:fsize 5000 -ll:zsize 5000 --strategy opt

# ResNet experiments
./ffcompile.sh resnet
./resnet -b 64 -ll:gpu 1 -ll:fsize 9000 -ll:zsize 5000 --strategy dp
./resnet -b 64 -ll:gpu 1 -ll:fsize 9000 -ll:zsize 5000 --strategy opt
./resnet -b 128 -ll:gpu 2 -ll:fsize 9000 -ll:zsize 5000 --strategy dp
./resnet -b 128 -ll:gpu 2 -ll:fsize 9000 -ll:zsize 5000 --strategy opt
./resnet -b 256 -ll:gpu 4 -ll:fsize 9000 -ll:zsize 5000 --strategy dp
./resnet -b 256 -ll:gpu 4 -ll:fsize 9000 -ll:zsize 5000 --strategy opt


