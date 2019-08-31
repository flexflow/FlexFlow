#!/bin/bash

./examples/dlrm -ll:gpu 1 -ll:cpu 4 -ll:fsize 12000 -ll:zsize 10000 --arch-sparse-feature-sie 13 --arch-embedding-size 1396-550-1761917-507795-290-21-11948-608-3-58176-5237-1497287-3127-26-12153-1068715-10-4836-2085-4-1312273-17-15-110946-91-72655 --arch-mlp-bot 13-512-256-64-16 --arch-mlp-top 512-256-1 --dataset /home/ubuntu/kaggle_dataset/kaggle_day_1.h5 --epochs 100 --batch-size 128

#./examples/dlrm -ll:gpu 1 -ll:fsize 12000 -ll:zsize 10000 --arch-sparse-feature-sie 13 --arch-embedding-size 1461-584-10131227-2202608-306-24-12518-634-4-93146-5684-8351593-3195-28-14993-5461306-11-5653-2173-4-7046547-18-16-286181-105-142572 --arch-mlp-bot 13-512-256-64-16 --arch-mlp-top 512-256-1
