echo "Running ResNet-50 with a parallelization strategy discovered by Unity"
$FF_HOME/build/examples/cpp/ResNet/resnet -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 -b 16 --budget 20

echo "Running ResNet-50 with data parallelism"
$FF_HOME/build/examples/cpp/ResNet/resnet -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 -b 16 --budget 20 --only-data-parallel
