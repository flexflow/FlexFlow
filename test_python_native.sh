#!/bin/bash
py_script="mnist_mlp.py mnist_cnn.py"

for script in $py_script; do
	./python/flexflow_python ./examples/python/native/$script -a -ll:py 1 -ll:gpu 1 -ll:fsize 10000 -ll:zsize 10000	
done
