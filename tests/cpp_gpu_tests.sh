#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

FF_HOME="$(realpath ..)"
export FF_HOME

GPUS=$1
BATCHSIZE=$((GPUS * 64))
FSIZE=13800
ZSIZE=12192

GPU_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ $(( GPUS )) -gt $(( GPU_AVAILABLE )) ]; then echo "The test requires $GPUS GPUs, but only $GPU_AVAILABLE are available. Try reducing the number of nodes, or the number of gpus/node." ; exit; fi

remove_mnist() {
	rm -f train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz train-images-idx3-ubyte train-labels-idx1-ubyte
}

download_mnist() {
	if [[ ! -f train-images-idx3-ubyte || ! -f train-labels-idx1-ubyte ]]; then
		remove_mnist
		wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
		wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
		gzip -d train-images-idx3-ubyte.gz
		gzip -d train-labels-idx1-ubyte.gz
	fi
}

# Check if the AlexNet/alexnet example exists in the build folder. If so, run the tests out of the build folder
# Otherwise, look for the example binaries in the folders in the PATH, plus in the subdirectory of the flexflow
# Python package (if it exists)
if [[ -f "$FF_HOME/build/examples/cpp/AlexNet/alexnet" ]]; then
	echo "Running C++ tests from folder: $FF_HOME/build/examples/cpp"
	"$FF_HOME"/build/examples/cpp/AlexNet/alexnet -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	# TODO: fix DLRM test
	# "$FF_HOME"/build/examples/cpp/DLRM/dlrm -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	#"$FF_HOME"/build/examples/cpp/InceptionV3/inception -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/MLP_Unify/mlp_unify -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/ResNet/resnet -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/Transformer/transformer -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b $((GPUS * 8)) --only-data-parallel
	"$FF_HOME"/build/examples/cpp/XDL/xdl -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	"$FF_HOME"/build/examples/cpp/candle_uno/candle_uno -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	download_mnist
	"$FF_HOME"/build/examples/cpp/mixture_of_experts/moe -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b 64 --only-data-parallel
	remove_mnist
	"$FF_HOME"/build/examples/cpp/resnext50/resnext50 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	# TODO: fix split tests
	# "$FF_HOME"/build/examples/cpp/split_test/split_test -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	# "$FF_HOME"/build/examples/cpp/split_test_2/split_test_2 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
else
	python_packages=$(python -c "from distutils import sysconfig; print(sysconfig.get_python_lib(plat_specific=False,standard_lib=False))")
	OLD_PATH="$PATH"
	OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
	export PATH="${python_packages}/flexflow/bin:${PATH}"
	export LD_LIBRARY_PATH="${python_packages}/flexflow/lib:${LD_LIBRARY_PATH}"
	IFS=:
	found=false
	for path in $PATH; do
		if [[ -f "$path/alexnet" ]]; then
			echo "Running C++ tests from folder: $path"
			found=true
			alexnet -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			# TODO: fix DLRM test
			# dlrm -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			#inception -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			mlp_unify -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			resnet -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			transformer -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b $((GPUS * 8)) --only-data-parallel
			xdl -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			candle_uno -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			download_mnist
			moe -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b 64 --only-data-parallel
			remove_mnist
			resnext50 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			# TODO: fix split tests 
			# split_test -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
			# split_test_2 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
		fi
	done
	export PATH="$OLD_PATH"
	export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
	if [ ! $found ]; then echo "C++ test binaries not found"; exit 1; fi
fi


