set -x
set -e
# mpirun -np 1 ./flexflow_python example/print_layers.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 1
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 2
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 3
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 5
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 6
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 7
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 8
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 9
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 12
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 13

# #Sequantial model tests
# mpirun -np 1 ./flexflow_python example/keras/seq_mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/seq_mnist_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/seq_reuters_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/seq_cifar10_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/seq_mnist_mlp_net2net.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/seq_mnist_cnn_net2net.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/seq_mnist_cnn_nested.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
#
# #Keras other
# mpirun -np 1 ./flexflow_python example/keras/callback.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/unary.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

#Functional API
# mpirun -np 1 ./flexflow_python example/keras/func_mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_mnist_mlp_concat.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_mnist_mlp_concat2.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_mnist_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_mnist_cnn_concat.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_nested.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_cifar10_alexnet.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_mnist_mlp_net2net.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_net2net.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
#Possible crash
# mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_concat.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_concat_model.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_concat_seq_model.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

# mpirun -np 1 ./flexflow_python example/alexnet.py -ll:py 1 -ll:gpu 2 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/cnn_attach.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/mlp_attach.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# #mpirun -np 1 ./flexflow_python example/cnn_cifar.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 4
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 10
# mpirun -np 1 ./flexflow_python example/keras/mnist_func.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192 --type 11
