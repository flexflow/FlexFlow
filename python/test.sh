set -x
set -e

GPUS=$1

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi

#torch
#./flexflow_python $FF_HOME/examples/python/pytorch/cifar10_cnn_torch.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
#./flexflow_python $FF_HOME/examples/python/pytorch/cifar10_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
#./flexflow_python $FF_HOME/examples/python/pytorch/mnist_mlp_torch.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192
#./flexflow_python $FF_HOME/examples/python/pytorch/mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192

#ONNX
#python3 $FF_HOME/examples/python/onnx/mnist_mlp_pt.py
#./flexflow_python $FF_HOME/examples/python/onnx/mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5 --test_type 1
#python3 $FF_HOME/examples/python/onnx/mnist_mlp_keras.py
#./flexflow_python $FF_HOME/examples/python/onnx/mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5 --test_type 0 
#python3 $FF_HOME/examples/python/onnx/cifar10_cnn_pt.py
#./flexflow_python $FF_HOME/examples/python/onnx/cifar10_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 40 --test_type 1
#python3 $FF_HOME/examples/python/onnx/cifar10_cnn_keras.py
#./flexflow_python $FF_HOME/examples/python/onnx/cifar10_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 40 --test_type 0
 
#Sequantial model tests
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_reuters_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_cifar10_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_mlp_net2net.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 4096 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_cnn_net2net.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 4096 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/seq_mnist_cnn_nested.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192

#Keras other
./flexflow_python $FF_HOME/examples/python/keras/callback.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/unary.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 4096 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/reshape.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192

#Functional API
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_mlp_concat.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_mlp_concat2.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_cnn_concat.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_nested.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_alexnet.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 4096 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_mnist_mlp_net2net.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 4096 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_net2net.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 4096 -ll:zsize 12192

#Python
./flexflow_python $FF_HOME/examples/python/native/split.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/native/alexnet.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 40
./flexflow_python $FF_HOME/examples/python/native/mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5 --search-budget 10000 --export-strategy strategy.txt
./flexflow_python $FF_HOME/examples/python/native/mnist_mlp.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5 --import-strategy strategy.txt
./flexflow_python $FF_HOME/examples/python/native/mnist_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5
./flexflow_python $FF_HOME/examples/python/native/cifar10_cnn.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 40
./flexflow_python $FF_HOME/examples/python/native/cifar10_cnn_attach.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5
./flexflow_python $FF_HOME/examples/python/native/mnist_mlp_attach.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5

#Only work with CFFI
./flexflow_python $FF_HOME/examples/python/native/print_layers.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 5

#Possible crash
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_concat.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_concat_model.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/keras/func_cifar10_cnn_concat_seq_model.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192
./flexflow_python $FF_HOME/examples/python/native/cifar10_cnn_concat.py -ll:py 1 -ll:gpu $GPUS -ll:fsize 2048 -ll:zsize 12192 --epochs 40

