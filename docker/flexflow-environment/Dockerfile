ARG cuda_version
ARG python_version
FROM nvidia/cuda:${cuda_version}-cudnn8-devel-ubuntu20.04

LABEL org.opencontainers.image.source=https://github.com/flexflow/FlexFlow
LABEL org.opencontainers.image.description="FlexFlow environment container"

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git zlib1g-dev lsb-release nano libhdf5-dev && \
    rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential apt-utils \
    ca-certificates libssl-dev curl unzip htop && DEBIAN_FRONTEND=noninteractive \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:ubuntu-toolchain-r/test && \
        apt-get update -y && \
        apt-get upgrade -y libstdc++6

# Install Python3 with Miniconda corresponding version (assume python_version format is either "latest" or "3.11")
ARG conda_package="Miniconda3-${python_version}-Linux-x86_64.sh"
RUN  if [ "$python_version" != "lastest" ]; then \
        ARG conda_package="Miniconda3-py${python_version//./}_23.5.2-0-Linux-x86_64.sh" \
    fi
RUN wget -c -q https://repo.continuum.io/miniconda/${conda_packge} && \
            mv ${conda_packge} ~/${conda_packge} && \
            chmod +x ~/${conda_packge} && \
            bash ~/${conda_packge} -b -p /opt/conda && \
            rm ~/${conda_packge} && \
            /opt/conda/bin/conda upgrade --all && \
            /opt/conda/bin/conda install conda-build conda-verify && \
            /opt/conda/bin/conda clean -ya; \
# Optionally install HIP dependencies
# Note that amd's docs say to also install the `hip-runtime-nvidia` package. This
# package attempts to re-install cuda even though cuda is already installed
# in the container. It also attempts to install packages for a graphical install.
# For our container, we don't need `hip-runtime-nvidia`
ARG FF_GPU_BACKEND "cuda"
RUN  if [ "$FF_GPU_BACKEND" = "hip_cuda" ] || [ "$FF_GPU_BACKEND" = "hip_rocm" ]; then \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"; \
        wget https://repo.radeon.com/amdgpu-install/22.20.5/ubuntu/bionic/amdgpu-install_22.20.50205-1_all.deb; \
        apt-get install -y ./amdgpu-install_22.20.50205-1_all.deb; \
        rm ./amdgpu-install_22.20.50205-1_all.deb; \
        amdgpu-install -y --usecase=hip,rocm --no-dkms; \
        apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk; \
    else \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"; \
    fi
RUN rm -rf /var/lib/apt/lists/*

# Set env vars
ENV PATH /opt/conda/bin:$PATH
ENV CUDNN_DIR /usr/local/cuda
ENV CUDA_DIR /usr/local/cuda

# Install python packages and other dependencies
RUN conda install -c conda-forge cmake make pillow cmake-build-extension pybind11 numpy pandas keras-preprocessing
# Install CPU-only Pytorch and related dependencies
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN conda install -c conda-forge onnx transformers sentencepiece
RUN pip3 install tensorflow

ENTRYPOINT ["/bin/bash"]
