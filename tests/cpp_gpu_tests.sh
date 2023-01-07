#! /usr/bin/env bash
set -x
set -e

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi
GPUS=$1
BATCHSIZE=$((GPUS * 64))
FSIZE=14048
ZSIZE=12192

# Check if the AlexNet/alexnet example exists in the build folder. If so, run the tests out of the build folder
# Otherwise, look for the example binaries in the folders in the PATH, plus in the subdirectory of the flexflow
# Python package (if it exists)
if [[ -f "$FF_HOME/build/examples/cpp/AlexNet/alexnet" ]]; then
	echo "Running C++ tests from folder: $FF_HOME/build/examples/cpp"
	"$FF_HOME"/build/examples/cpp/AlexNet/alexnet -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	# TODO: fix DLRM test
	# "$FF_HOME"/build/examples/cpp/DLRM/dlrm -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/InceptionV3/inception -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/MLP_Unify/mlp_unify -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/ResNet/resnet -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/Transformer/transformer -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b $((GPUS * 8)) --only-data-parallel
	"$FF_HOME"/build/examples/cpp/XDL/xdl -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/candle_uno/candle_uno -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	wget -P "$FF_HOME"/build/examples/cpp/mixture_of_experts http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	wget -P "$FF_HOME"/build/examples/cpp/mixture_of_experts http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	gzip -d "$FF_HOME"/build/examples/cpp/mixture_of_experts/train-images-idx3-ubyte.gz
	gzip -d "$FF_HOME"/build/examples/cpp/mixture_of_experts/train-labels-idx1-ubyte.gz
	"$FF_HOME"/build/examples/cpp/mixture_of_experts/moe -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b 64 --only-data-parallel
	"$FF_HOME"/build/examples/cpp/resnext50/resnext50 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/split_test/split_test -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/split_test_2/split_test_2 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
else
	python_packages=$(python -c "from distutils import sysconfig; print(sysconfig.get_python_lib(plat_specific=False,standard_lib=False))")
	export PATH="${python_packages}/flexflow/bin:${PATH}"
	IFS=:
	for path in $PATH; do
		if [[ -f "$path/alexnet" ]]; then
			echo "Running C++ tests from folder: $path"
			alexnet -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			# TODO: fix DLRM test
			# dlrm -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			inception -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			mlp_unify -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			resnet -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			transformer -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b $((GPUS * 8)) --only-data-parallel
			xdl -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			candle_uno -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			wget -P "$path" http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
			wget -P "$path" http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
			gzip -d "$path"/train-images-idx3-ubyte.gz
			gzip -d "$path"/train-labels-idx1-ubyte.gz
			moe -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b 64 --only-data-parallel
			resnext50 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			split_test -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			split_test_2 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
		fi
	done
fi


