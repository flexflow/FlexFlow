# FlexFlow Docker
This folder contains the Dockerfiles and scripts that you can use to quickly run FlexFlow with no manual installation required. To use the containers, follow the steps below.

## Prerequisites
You will need a machine with a NVIDIA GPU, with drivers installed. You will also need to have Docker and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started) installed on the host machine.

## Downloading a pre-built package
The fastest way to run FlexFlow is to use one of the pre-built containers, which we update for each commit to the `master` branch. The available containers are the following, and can be found [at this link](https://github.com/orgs/flexflow/packages?repo_name=FlexFlow):

* [flexflow-cuda](https://github.com/orgs/flexflow/packages/container/package/flexflow-cuda): the pre-built version of FlexFlow targeting GPUs with a CUDA backend. N.B.: currently, this container is only fully compatible with host machines that have CUDA 11.7 installed.
* [flexflow-hip_rocm](https://github.com/orgs/flexflow/packages/container/package/flexflow-hip_rocm): the pre-built version of FlexFlow targeting GPUs with a HIP-ROCM backend.
* [flexflow-environment-cuda](https://github.com/orgs/flexflow/packages/container/package/flexflow-environment-cuda) and [flexflow-environment-hip_rocm](https://github.com/orgs/flexflow/packages/container/package/flexflow-environment-hip_rocm): these are the base layers for `flexflow-cuda` and `flexflow-hip_rocm`. The packages are used in CI or for internal use, and contain all the dependencies needed to build/run Flexflow. N.B.: currently, the `flexflow-environment-cuda` container is only fully compatible with host machines that have CUDA 11.7 installed.

The easiest way to download any of the Docker containers above is to call:

```
./docker/pull.sh <CONTAINER_NAME>
```

After downloading a container you can use the `run.sh` script to run it by following the instructions in the section below.

## Building a Docker container from scratch
If you prefer to build one of the Docker containers from scratch, you can do so with the help of the `build.sh` script. You can configure the build via the same environment variables that you'd use to configure a CMake build (refer to the [Installation guide](../INSTALL.md) and to the `config/config.linux` file). For example, to build for a CUDA backend, you can export `FF_GPU_BACKEND=cuda` (you can also omit this since `cuda` is the default value for `FF_GPU_BACKEND`).

To build the FlexFlow container, run (the `flexflow` argument of the build script can be omitted):

```
FF_GPU_BACKEND=<YOUR_GPU_BACKEND> ./docker/build.sh flexflow
```

If you only want to build the `flexflow-environment` image (the base layers of the `flexflow` container, used in CI and for other internal purposes), run:

```
FF_GPU_BACKEND=<YOUR_GPU_BACKEND> ./docker/build.sh flexflow-environment
``` 

## Running a Docker container
After having either built or downloaded a Docker container by following the instructions above, you can run it with the following command (the `--image_name flexflow` argument of the run script can be omitted). Moreover, you can choose the CUDA version to build using optional flag `--cuda_version`; otherwise, it will autodetect the cuda_version installed on the current machine and run the Docker container with it if available.

```
FF_GPU_BACKEND=<YOUR_GPU_BACKEND> ./docker/run.sh --image_name flexflow --cuda_version 11.1
```

If you wish to run the `flexflow-environment` container, run:

```
FF_GPU_BACKEND=<YOUR_GPU_BACKEND> ./docker/run.sh --image_name flexflow-environment --cuda_version 11.1
```

Once again, if your backend is CUDA, you can omit the `FF_GPU_BACKEND` environment variable, since `cuda` is used as the default value.

N.B.: If you don't have GPUs available on the machine, edit the `run.sh` script and set `ATTACH_GPUS=false` before running it.
