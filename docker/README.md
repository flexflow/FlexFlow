# FlexFlow Docker
This folder contains a Docker container that you can use to quickly run FlexFlow with no manual installation required. To use it, follow the steps below.

## Prerequisites
You will need a machine with a NVIDIA GPU, with drivers installed. You will also need to have Docker and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started) installed on the host machine.

## Build and run instructions
1. Apply any desired configurations by editing the [config.linux](../config/config.linux) file in the `FlexFlow/configs` folder. Leave the file unchanged if you want to run with the default options
2. Build the Docker image with the following command (replace `base` with `mt5` if you are looking to build a container supporting the mt5 model in `examples/python/pytorch/mt5`):
```
./build.sh base
```
3. Run the Docker container with the following command (again replacing `base` with `mt5` if needed). If you don't have GPUs available on the machine, edit the `run.sh` script and set `ATTACH_GPUS=false` before running it.
```
./run.sh base
```
