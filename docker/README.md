# FlexFlow Docker
This folder contains a Docker container that you can use to quickly run FlexFlow with no manual installation required. To use it, follow the steps below.

## Prerequisites
You will need a machine with a NVIDIA GPU, with drivers installed. You will also need to have Docker and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started) installed on the host machine.

## Build and run instructions
1. Apply any desired configurations by editing the [config.linux](../config/config.linux) file in the `FlexFlow/configs` folder. Leave the file unchanged if you want to run with the default options
2. Build the Docker image with the following command (run from the `FlexFlow/docker` folder):
```bash
./build.sh
```
3. Run the Docker container with the following command. If you don't have GPUs available on the machine, remove the `--gpus all` flag.
```bash
docker run -it --gpus all flexflow:latest
```
4. After exiting the container, you can take note of the container ID and log back in (without losing your changes) with:
```
docker start <container ID> && docker attach <container ID>
```
In alternative, you can commit your changes to a new image for better permanence:
```
docker commit <container ID> new_image_name:tag_name(optional)
```
