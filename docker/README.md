# FlexFlow Docker
This folder contains the Dockerfiles and scripts that you can use to quickly run FlexFlow with no manual installation required. To use the containers, follow the steps below.

## Prerequisites
You can build and run the FlexFlow Docker images on any machine, but if you want to train or serve a model, you will need a machine with a NVIDIA or AMD GPU, with drivers installed. You will also need to have Docker and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started) installed on the host machine. If using an AMD GPU, follow the [Deploy ROCm Docker containers](https://rocm.docs.amd.com/en/latest/deploy/docker.html) instructions.

## Downloading a pre-built package
The fastest way to run FlexFlow is to use one of the pre-built containers, which we update for each commit to the `inference` branch (the `inference` branch is currently ahead of the `master` branch). The available containers are the following, and can be found [at this link](https://github.com/orgs/flexflow/packages?repo_name=FlexFlow):

* `flexflow`: the pre-built version of FlexFlow. We currently publish four version targeting AMD GPUs (ROCm versions: 5.3, 5.4, 5.5 and 5.6 ), and several versions for CUDA GPUs (CUDA versions: 11.1, 11.6, 11.7, 11.8, 12.0, 12.1, and 12.2). The CUDA images are named `flexflow-<GPU backend>-<GPU software version>`, e.g. [flexflow-hip_rocm-5.6](https://github.com/flexflow/FlexFlow/pkgs/container/flexflow-hip_rocm-5.6) or [flexflow-cuda-12.0](https://github.com/orgs/flexflow/packages/container/package/flexflow-cuda-12.0) or 
* `flexflow-environment`: this is the base layer for `flexflow`. The packages are used in CI or for internal use, and contain all the dependencies needed to build/run Flexflow. You may find them useful if you want to build FlexFlow yourself. We also publish four version of `flexflow-environment` for AMD GPUs and, for NVIDIA GPUs, one for each CUDA version in the list above. The naming convention is similar, too. For example, the `flexflow-environment` image for CUDA 12.0 is tagged [flexflow-environment-cuda-12.0](https://github.com/orgs/flexflow/packages/container/package/flexflow-environment-cuda-12.0).

The easiest way to download any of the Docker containers above is to call:

```
./docker/pull.sh <CONTAINER_NAME>
```

where `CONTAINER_NAME` is `flexflow` (or `flexflow-environment`). By default, the script will assume a NVIDIA backend and attempt to detect the CUDA version on your machine, to download the relevant container. If your machine has AMD GPUs, or no GPUs, or if you want to specify  the CUDA/ROCM version to download, set the environment variables below:

* `FF_GPU_BACKEND` (supported options: `cuda`, `hip_rocm`) to specify the GPU backend of the Docker container to be downloaded.
* `cuda_version` (supported options: 11.1, 11.6, 11.7, 11.8, 12.0, 12.1 and 12.2) to specify the CUDA version, when using a `cuda` backend. If `FF_GPU_BACKEND` is set to `hip_rocm`, the `cuda_version` env will be ignored
* `hip_version` (supported options: 5.3, 5.4, 5.5, 5.6) to specify the ROCm version, when using a HIP backend. If `FF_GPU_BACKEND` is set to `cuda`, the `hip_version` env will be ignored.


After downloading a container you can use the `run.sh` script to run it by following the instructions in the section below.

## Building a Docker container from scratch
If you prefer to build one of the Docker containers from scratch, you can do so with the help of the `build.sh` script. You can configure the build via the same environment variables that you'd use to configure a CMake build (refer to the [Installation guide](https://flexflow.readthedocs.io/en/latest/installation.html) and to the `config/config.linux` file). For example, to build for a CUDA backend, you can export `FF_GPU_BACKEND=cuda` (you can also omit this since `cuda` is the default value for `FF_GPU_BACKEND`). When building for the `cuda` backend, you can pick the CUDA version by setting the optional environment variable `cuda_version`, e.g.: `export cuda_version=12.0`. Leaving the `cuda_version` variable blank will let the script autodetect the CUDA version installed on the host machine, and build for that version. Setting the `cuda_version` env will have no effect when building for a GPU backend other than CUDA. Similarly, you can pick the ROCm version by setting `hip_version` when the backend is `FF_GPU_BACKEND=hip_rocm`, whereas the env will be ignored for non-HIP backends.

To build the FlexFlow container, run (the `flexflow` argument of the build script can be omitted):

```
./docker/build.sh flexflow
```

If you only want to build the `flexflow-environment` image (the base layers of the `flexflow` container, used in CI and for other internal purposes), run:

```
./docker/build.sh flexflow-environment
``` 

## Running a Docker container
After having either built or downloaded a Docker container by following the instructions above, you can run it with the following command (image name argument of the run script can be omitted). Once again, you can set the `FF_GPU_BACKEND`, `cuda_version` and `hip_version` optional environment variables to run the docker image with the desired GPU backend and CUDA/HIP version:

* `FF_GPU_BACKEND` (supported options: `cuda`, `hip_rocm`) to specify the GPU backend of the Docker container to be run.
* `cuda_version` (supported options: 11.1, 11.6, 11.7, 11.8, 12.0, 12.1, 12.2) to specify the CUDA version, when using a `cuda` backend. If `FF_GPU_BACKEND` is set to `hip_rocm`, the `cuda_version` env will be ignored
* `hip_version` (supported options: 5.3, 5.4, 5.5, 5.6) to specify the ROCm version, when using a HIP backend. If `FF_GPU_BACKEND` is set to `cuda`, the `hip_version` env will be ignored.

Leaving these variables unset will assume a GPU backend, and instruct the script to autodetect the CUDA version installed on the current machine and run the Docker container with it if available.

```
./docker/run.sh --image_name flexflow
```

If you wish to run the `flexflow-environment` container, run:

```
./docker/run.sh --image_name flexflow-environment
```

N.B.: If you don't have GPUs available on the machine, or you wish to run the docker image without attaching GPUs, you can set the environment variable `ATTACH_GPUS=false` before running the script.
