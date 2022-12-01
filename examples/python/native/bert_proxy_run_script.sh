#!/bin/bash

ulimit -l unlimited

fs=12000
zs=12000
b=1
g=1
budget=50000

#<7.5B
script='bert_proxy_native.py -ll:py 1 --iterations 100 --seq-length 128 --num-heads 32 --hidden-size 4096 --num_layers 36'
ss=7.5b.l18

#$FF_HOME/python/flexflow_python $script -ll:gpu 1 -ll:fsize $fs -ll:zsize $zs -b $b --budget 1000 --export ./$ss.b$b.g1.bg$budget
#$FF_HOME/python/flexflow_python $script -ll:gpu $g -ll:fsize $fs -ll:zsize $zs -b $b --budget $budget --enable-parameter-parallel --enable-attribute-parallel --export ./$ss.b$b.g$g.bg$budget --import ./$ss.b$b.g1.bg$budget --taskgraph ./tg.$ss.b$b.g$g.bg$budget

"$FF_HOME"/python/flexflow_python "$script" -ll:gpu "$g" -ll:fsize "$fs" -ll:zsize "$zs" -b "$b"  --enable-parameter-parallel --enable-attribute-parallel --import "./$ss.b$b.g$g.bg$budget"
#-lg:prof 1 -logfile spy_$ss.%.log -lg:spy -lg:prof_logfile prof_$ss.%.gz
