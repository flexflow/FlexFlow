set -x
set -e

GPUS=$1

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi

#Sequantial model tests
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_reuters_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_cifar10_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_mlp_net2net.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_cnn_net2net.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_cnn_nested.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192

#Keras other
./flexflow_python $FF_HOME/examples/python/keras/callback.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/unary.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192

#Functional API
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_mlp_concat.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_mlp_concat2.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_cnn_concat.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_nested.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_alexnet.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_mlp_net2net.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_net2net.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192

#Python
./flexflow_python $FF_HOME/examples/python/native/print_layers.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5
./flexflow_python $FF_HOME/examples/python/native/alexnet.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 40
./flexflow_python $FF_HOME/examples/python/native/mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5
./flexflow_python $FF_HOME/examples/python/native/mnist_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5
./flexflow_python $FF_HOME/examples/python/native/cifar10_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 40
./flexflow_python $FF_HOME/examples/python/native/cifar10_cnn_attach.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5
./flexflow_python $FF_HOME/examples/python/native/mnist_mlp_attach.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5

#Possible crash
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_concat.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_concat_model.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_concat_seq_model.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/native/cifar10_cnn_concat.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 40

