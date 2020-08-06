set -x
set -e
#Sequantial model tests
mpirun -np 1 ./flexflow_python example/keras/seq_mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/seq_mnist_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/seq_reuters_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/seq_cifar10_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/seq_mnist_mlp_net2net.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/seq_mnist_cnn_net2net.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/seq_mnist_cnn_nested.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

#Keras other
mpirun -np 1 ./flexflow_python example/keras/callback.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/unary.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

#Functional API
mpirun -np 1 ./flexflow_python example/keras/func_mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_mnist_mlp_concat.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_mnist_mlp_concat2.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_mnist_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_mnist_cnn_concat.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_nested.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_cifar10_alexnet.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_mnist_mlp_net2net.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_net2net.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

#Python
mpirun -np 1 ./flexflow_python example/print_layers.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/alexnet.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/mnist_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/cifar10_cnn.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/cifar10_cnn_attach.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/mnist_mlp_attach.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

#Possible crash
mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_concat.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_concat_model.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/keras/func_cifar10_cnn_concat_seq_model.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
mpirun -np 1 ./flexflow_python example/cifar10_cnn_concat.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

