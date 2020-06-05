set -x
set -e
mpirun -np 1 ./flexflow_python example/print_layers.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 1
mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 2
mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 3
mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 5
mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 6
mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 7
mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 8
mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 9
mpirun -np 1 ./flexflow_python example/keras/mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/mnist_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/mnist_net2net.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/cifar10_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/reuters_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/alexnet.py -ll:py 1 -ll:gpu 2 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/cnn_attach.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/mlp_attach.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/cnn_cifar.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 4